import torch
import torch.nn as nn
import torch.nn.functional as F

from verification.nn_verification import find_preactivation_bounds, output_Lp_bounds_LiRPA
from verification.optimization import add_gurobi_constr_for_MLP

from utils.sampling import find_bounding_box
import numpy as np
from pympc.geometry.polyhedron import Polyhedron
from utils.training import get_nn_info, setup_relu
from utils.sampling import uniform_random_sample_from_Polyhedron

import gurobipy as gp
import matplotlib.pyplot as plt
from torch.nn.utils.parametrizations import spectral_norm

class Barrier_Fcn(nn.Module):
    def __init__(self, net, domain):
        super().__init__()
        self.n_in, self.n_out = net[0].in_features, net[-1].out_features

        self.net = net
        self.A = nn.Parameter(torch.zeros(self.n_out, self.n_out))

        # WLOG, we select the index for the unsafe set constraint to be 0
        self.unsafe_index = 0

        # domain of the state
        self.domain = domain

        self.pre_act_lb, self.pre_act_ub = None, None

        self.basis_fcn_dim = net[-1].in_features

    def forward(self, x):
        return self.net(x)
    
    @property
    def device(self):
        return next(self.net.parameters()).device

    @property
    def basis_net(self):
        return nn.Sequential(*list(self.net)[:-1])

    @property
    def last_linear_layer(self):
        layer = self.net[-1]
        return layer

    def initialize_pre_act_bounds(self, domain=None, net=None):
        if domain is None:
            domain = self.domain

        if net is None:
            net = self.net

        pre_act_bounds, output_bounds = find_preactivation_bounds(net, domain)

        self.pre_act_lb = [item['lb'] for item in pre_act_bounds]
        self.pre_act_ub = [item['ub'] for item in pre_act_bounds]
        return pre_act_bounds, output_bounds

    def evaluate_basis(self, x):
        return self.basis_net(x)
    
    def initial_set_loss(self, x0_samples, leaky_relu=False):
        N0 = x0_samples.size(0)
        x0_samples = x0_samples.to(self.device)
        if leaky_relu:
            return F.leaky_relu(self.net(x0_samples)).sum() / N0
        else:
            return F.relu(self.net(x0_samples)).sum() / N0

    def unsafe_set_loss(self, xu_samples, leaky_relu=False):
        Nu = xu_samples.size(0)
        xu_samples = xu_samples.to(self.device)
        if leaky_relu:
            return F.leaky_relu(-self.net(xu_samples)[:, self.unsafe_index]).sum()/Nu
        else:
            return F.relu(-self.net(xu_samples)[:, self.unsafe_index]).sum()/Nu

    def decrease_loss(self, x_samples, xn_samples, leaky_relu=False):
        N = x_samples.size(0)
        x_samples, xn_samples = x_samples.to(self.device), xn_samples.to(self.device)
        if leaky_relu:
            return F.leaky_relu(self.net(xn_samples) - self.net(x_samples) @ self.A.T).sum() / N
        else:
            return F.relu(self.net(xn_samples) - self.net(x_samples) @ self.A.T).sum() / N

    def loss(self, x0_samples, xu_samples, x_samples, xn_samples, l1_lambda=None, leaky_relu=False):
        # x_samples: N x nx tensor
        # # TODO: using leaky relu may improve the training performance
        # N0, Nu, N = x0_samples.size(0), xu_samples.size(0), x_samples.size(0)
        #
        # # TODO: adding margins to improve numerical stability of training
        # epsilon = 1e-4
        # cost = F.relu(self.net(x0_samples)).sum()/N0 + F.relu(-self.net(xu_samples)[:, self.unsafe_index]).sum()/Nu + \
        #         F.relu(self.net(xn_samples)-self.net(x_samples)@self.A.T).sum()/N

        cost = self.initial_set_loss(x0_samples, leaky_relu) + self.unsafe_set_loss(xu_samples, leaky_relu) + \
               self.decrease_loss(x_samples, xn_samples, leaky_relu)

        if l1_lambda is not None:
            l1_norm = sum(p.abs().sum() for p in self.net.parameters())
            cost = cost + l1_lambda*l1_norm
        return cost

    def add_gurobi_constr(self, gurobi_model, input_var, output_var, net=None, domain=None, mark = ''):
        # add the constraint output_var = B(input_var)
        if domain is not None:
            self.initialize_pre_act_bounds(domain, net)

        if self.pre_act_ub is None or self.pre_act_lb is None:
            self.initialize_pre_act_bounds(domain, net)

        if net is None:
            net = self.net

        # add constraints to the gurobi model to enforce the following constraints: output_var = NN(input_var)
        nx = self.n_in
        ny = self.n_out

        x = list()
        for i in range(nx):
            x.append(gurobi_model.getVarByName(input_var + '[' + str(i) + ']'))

        y = list()
        for i in range(ny):
            y.append(gurobi_model.getVarByName(output_var + '[' + str(i) + ']'))

        gurobi_model = add_gurobi_constr_for_MLP(gurobi_model, net, x, y, self.pre_act_lb, self.pre_act_ub, mark=mark)
        gurobi_model.update()
        return gurobi_model

class Polyhedral_Set:
     # TODO: extend to union of polyhedra.
    def __init__(self, A, b):
        # polytope defined by Ax <= b
        self.A = A
        self.b = b
        self.set = Polyhedron(A, b)
        self.nx = self.A.shape[1]

        self.lb, self.ub = None, None
        self.is_box = False

    def sample(self, N):
        samples = uniform_random_sample_from_Polyhedron(self.set, N)
        return samples.astype('float32')

    def add_gurobi_constr(self, gurobi_model, input_var, mark = ''):
        x = list()
        for i in range(self.nx):
            x.append(gurobi_model.getVarByName(input_var + '[' + str(i) + ']'))

        A = self.A
        b = self.b

        gurobi_model.addConstrs( (A[i]@x <= b[i] for i in range(A.shape[0])), name = 'set_constr_step_' + mark)
        gurobi_model.update()
        return gurobi_model

class Controller(nn.Module):
    def __init__(self, net, domain, u_lb=None, u_ub=None):
        super().__init__()
        self.net = net

        self.u_lb, self.u_ub = u_lb, u_ub

        self.x_dim = self.net[0].in_features
        self.u_dim = self.net[-1].out_features

        self.domain = domain

        self.pre_act_lb, self.pre_act_ub = None, None

        # bounds on the controller output before projection
        self.output_bounds = None
        self.output_domain = None

        if self.u_lb is not None and self.u_ub is not None:
            clamp_layer = self.generate_clamp_layer()
            self.clamp_layer = clamp_layer
        else:
            self.clamp_layer = None

    def generate_clamp_layer(self):
        u_dim = self.u_dim
        device = self.device
        assert self.u_lb is not None and self.u_ub is not None
        u_lb, u_ub = self.u_lb.to(device), self.u_ub.to(device)

        nn1 = nn.Linear(u_dim, u_dim).to(device)
        nn1.weight.data = torch.eye(u_dim).to(device)
        nn1.bias.data = -u_lb

        nn2 = nn.Linear(u_dim, u_dim).to(device)
        nn2.weight.data = -torch.eye(u_dim).to(device)
        nn2.bias.data = u_ub - u_lb

        nn3 = nn.Linear(u_dim, u_dim).to(device)
        nn3.weight.data = -torch.eye(u_dim).to(device)
        nn3.bias.data = u_ub

        clamp_layer = nn.Sequential(nn1, nn.ReLU(), nn2, nn.ReLU(), nn3)
        return clamp_layer

    @property
    def device(self):
        return next(self.net.parameters()).device

    def forward(self, x):
        if self.u_lb is not None and self.u_ub is not None:
            output = self.net(x)
            # u_lb, u_ub = self.u_lb.to(self.device), self.u_ub.to(self.device)
            # temp = F.relu(output-u_lb) + u_lb
            # clamped_output = -F.relu(u_ub - temp) + u_ub
            clamped_output = self.clamp_layer(output)
            return clamped_output
        else:
            return self.net(x)

    def initialize_pre_act_bounds(self, domain=None):
        if domain is None:
            domain = self.domain
        pre_act_bounds, output_bounds = find_preactivation_bounds(self.net, domain)
        self.pre_act_lb = [item['lb'] for item in pre_act_bounds]
        self.pre_act_ub = [item['ub'] for item in pre_act_bounds]
        self.output_bounds = output_bounds
        self.output_domain = Polyhedron.from_bounds(output_bounds['lb'], output_bounds['ub'])

        return pre_act_bounds, output_bounds

    def add_gurobi_constr(self, gurobi_model, input_var, output_var, mark = ''):
        # TODO: the projection operation has not been modeled yet.
        if self.pre_act_lb is None or self.pre_act_ub is None:
            self.initialize_pre_act_bounds()

        x = list()
        for i in range(self.x_dim):
            x.append(gurobi_model.getVarByName(input_var + '[' + str(i) + ']'))

        if self.u_lb is not None and self.u_ub is not None:
            y = gurobi_model.addVars(self.u_dim, lb=-gp.GRB.INFINITY, name='u_med_'+str(mark))
            gurobi_model.update()
        else:
            y = list()
            for i in range(self.u_dim):
                y.append(gurobi_model.getVarByName(output_var + '[' + str(i) + ']'))

        nx, ny = self.x_dim, self.u_dim

        gurobi_model = add_gurobi_constr_for_MLP(gurobi_model, self.net, x, y, self.pre_act_lb, self.pre_act_ub, mark=mark)

        gurobi_model.update()

        # encode the projection layer
        if self.u_lb is not None and self.u_ub is not None:
            u_lb, u_ub = self.u_lb.cpu().numpy(), self.u_ub.cpu().numpy()

            output_bounds = self.output_bounds
            output_lb, output_ub = output_bounds['lb'], output_bounds['ub']

            # extract the output variable
            u = list()
            for i in range(self.u_dim):
                u.append(gurobi_model.getVarByName(output_var + '[' + str(i) + ']'))

            z_l = gurobi_model.addVars(self.u_dim, lb=-gp.GRB.INFINITY, name='z_l_'+str(mark))
            t_l = gurobi_model.addVars(self.u_dim, name='t_l_' + str(mark), vtype=gp.GRB.BINARY)
            # z_u = gurobi_model.addVars(self.u_dim, lb=-gp.GRB.INFINITY, name='z_u_'+str(mark))
            t_u = gurobi_model.addVars(self.u_dim, name='t_u_' + str(mark), vtype=gp.GRB.BINARY)

            # add constraints: z_l = max(y, u_lb)
            # equivalently we have: z_l >= y, z_l >= u_lb, z_l <= y + M*(1-t_l), z_l<=lb+M*t_l

            # find the big-M parameter
            M_vec_lb = np.maximum(output_ub-u_lb, -output_lb+u_lb)
            assert np.all(M_vec_lb >= 0)

            gurobi_model.addConstrs((z_l[i] >= y[i] for i in range(self.u_dim)), name='proj_lb_binary_1_'+str(mark))
            gurobi_model.addConstrs((z_l[i] >= u_lb[i] for i in range(self.u_dim)), name='proj_lb_binary_2_'+str(mark))
            gurobi_model.addConstrs((z_l[i] <= y[i] + M_vec_lb[i]*(1-t_l[i]) for i in range(self.u_dim)), name='proj_lb_binary_3_'+str(mark))
            gurobi_model.addConstrs((z_l[i] <= u_lb[i] + M_vec_lb[i]*t_l[i] for i in range(self.u_dim)), name='proj_lb_binary_4_'+str(mark))

            # add constraints: u = min(z_l, u_ub)
            # equivalently, we have: u <= z_l, u <= u_ub, u >= z_l - M*(1-t_u), u >= u_ub - M*t_u

            # find the big-M parameter
            M_vec_ub = np.maximum(output_ub-u_ub, -output_lb+u_ub)
            assert np.all(M_vec_lb >= 0)

            gurobi_model.addConstrs((u[i] <= z_l[i] for i in range(self.u_dim)), name='proj_ub_binary_1_'+str(mark))
            gurobi_model.addConstrs((u[i] <= u_ub[i] for i in range(self.u_dim)), name='proj_ub_binary_2_'+str(mark))
            gurobi_model.addConstrs((u[i] >= z_l[i] - M_vec_ub[i]*(1-t_u[i]) for i in range(self.u_dim)), name='proj_ub_binary_3_'+str(mark))
            gurobi_model.addConstrs((u[i] >= u_ub[i] - M_vec_ub[i]*t_u[i] for i in range(self.u_dim)), name='proj_ub_binary_4_'+str(mark))

            gurobi_model.update()

        return gurobi_model


class OpenLoopDynamics(nn.Module):
    # Models x_+ = f(x, u) where f is an MLP
    def __init__(self, net, x_dim, u_dim, domain):
        super().__init__()
        self.net = net
        self.domain = domain

        self.x_dim, self.u_dim = x_dim, u_dim
        self.pre_act_lb, self.pre_act_ub = None, None

        self.output_bounds = None
        self.output_domain = None
    
    @property
    def device(self):
        return next(self.net.parameters()).device

    def forward(self, aug_x):
        return self.net(aug_x)

    def initialize_pre_act_bounds(self, domain=None):
        if domain is None:
            domain = self.domain
        pre_act_bounds, output_bounds = find_preactivation_bounds(self.net, domain)
        self.pre_act_lb = [item['lb'] for item in pre_act_bounds]
        self.pre_act_ub = [item['ub'] for item in pre_act_bounds]

        self.output_bounds = output_bounds
        self.output_domain = Polyhedron.from_bounds(output_bounds['lb'], output_bounds['ub'])

        return pre_act_bounds, output_bounds

    def add_gurobi_constr(self, gurobi_model, x_var, u_var, xn_var, mark=''):
        if self.pre_act_lb is None or self.pre_act_ub is None:
            self.initialize_pre_act_bounds()

        x = list()
        for i in range(self.x_dim):
            x.append(gurobi_model.getVarByName(x_var + '[' + str(i) + ']'))

        u = list()
        for i in range(self.u_dim):
            u.append(gurobi_model.getVarByName(u_var + '[' + str(i) + ']'))

        y = list()
        for i in range(self.x_dim):
            y.append(gurobi_model.getVarByName(xn_var + '[' + str(i) + ']'))

        aug_x = x + u
        gurobi_model = add_gurobi_constr_for_MLP(gurobi_model, self.net, aug_x, y, self.pre_act_lb, self.pre_act_ub, mark=mark)
        gurobi_model.update()
        return gurobi_model


class OpenLoopDynamicsResidual(nn.Module):
    # Models x_+ = [A B]@[x; u] + f(x, u) + c where f is an MLP
    # The linear part [A B]@[x; u] + c is given by a linear layer
    def __init__(self, lin_layer, res_net, x_dim, u_dim, domain):
        super().__init__()
        self.lin_layer = lin_layer
        self.res_net = res_net
        self.domain = domain

        self.x_dim, self.u_dim = x_dim, u_dim
        self.pre_act_lb, self.pre_act_ub = None, None

        self.output_bounds = None
        self.output_domain = None

    @property
    def device(self):
        return next(self.res_net.parameters()).device

    def forward(self, aug_x):
        return self.lin_layer(aug_x) + self.res_net(aug_x)

    def initialize_pre_act_bounds(self, domain=None):
        if domain is None:
            domain = self.domain
        pre_act_bounds, output_bounds = find_preactivation_bounds(self.res_net, domain)
        self.pre_act_lb = [item['lb'] for item in pre_act_bounds]
        self.pre_act_ub = [item['ub'] for item in pre_act_bounds]

        self.output_bounds = output_bounds
        self.output_domain = Polyhedron.from_bounds(output_bounds['lb'], output_bounds['ub'])

        return pre_act_bounds, output_bounds

    def add_gurobi_constr(self, gurobi_model, x_var, u_var, xn_var, mark=''):
        if self.pre_act_lb is None or self.pre_act_ub is None:
            self.initialize_pre_act_bounds()

        x = list()
        for i in range(self.x_dim):
            x.append(gurobi_model.getVarByName(x_var + '[' + str(i) + ']'))

        u = list()
        for i in range(self.u_dim):
            u.append(gurobi_model.getVarByName(u_var + '[' + str(i) + ']'))

        y = list()
        for i in range(self.x_dim):
            y.append(gurobi_model.getVarByName(xn_var + '[' + str(i) + ']'))

        res_dyn_output = 'r_0'
        res = gurobi_model.addVars(self.x_dim, lb=-gp.GRB.INFINITY, name=res_dyn_output)
        gurobi_model.update()

        aug_x = x + u
        gurobi_model = add_gurobi_constr_for_MLP(gurobi_model, self.res_net, aug_x, res, self.pre_act_lb, self.pre_act_ub,
                                                 mark=mark)

        lin_weight = self.lin_layer.weight.data.detach().cpu().numpy()
        lin_bias = self.lin_layer.bias.data.detach().cpu().numpy()

        aug_x_dict = gp.tupledict([(i, k) for i, k in enumerate(aug_x)])

        gurobi_model.addConstrs((y[i] == aug_x_dict.prod(dict(zip(range(lin_weight.shape[1]), lin_weight[i, :]))) + lin_bias[i]
                                 for i in range(self.x_dim)), name='res_dyn_' + str(mark))
        gurobi_model.update()

        return gurobi_model

class ClosedLoopDynamics(nn.Module):
    def __init__(self, forward_dyn, controller):
        super().__init__()
        self.forward_dyn = forward_dyn
        self.controller = controller
        self.x_dim = self.forward_dyn.x_dim
        self.u_dim = self.forward_dyn.u_dim

    @property
    def device(self):
        return self.forward_dyn.device

    def forward(self, x):
        u = self.controller(x)
        aug_x = torch.cat((x, u), dim=-1)
        return self.forward_dyn(aug_x)

    def simulate_trajectory(self, x, step):
        bs = x.size(0)
        traj = x
        for i in range(step):
            y = self.forward(x)
            traj = torch.cat((traj, y), dim=-1)
            x = y

        return traj.reshape(bs, -1, self.x_dim)

    @property
    def forward_dyn_domain(self):
        return self.forward_dyn.domain

    @property
    def controller_domain(self):
        return self.controller.domain

    @property
    def output_domain(self):
        return self.forward_dyn.output_domain

    def initialize_pre_act_bounds(self):
        self.forward_dyn.initialize_pre_act_bounds()
        self.controller.initialize_pre_act_bounds()

    def add_gurobi_constr(self, gurobi_model, x_var, xn_var, mark = ''):
        # y = f(x, u), u = pi(x)
        control_input_name = 'u_0'
        gurobi_model.addVars(self.u_dim, lb=-gp.GRB.INFINITY, name=control_input_name)
        gurobi_model.update()

        gurobi_model = self.controller.add_gurobi_constr(gurobi_model, x_var, control_input_name, 'controller_' + str(mark))
        gurobi_model = self.forward_dyn.add_gurobi_constr(gurobi_model, x_var, control_input_name, xn_var, mark = 'forward_dyn_' + str(mark))
        gurobi_model.update()

        return gurobi_model


class Auto_Dynamics(nn.Module):
    # system described by x_+ = f(x) where f is a multi-layer perception
    def __init__(self, net, domain):
        super().__init__()
        self.net = net
        self.n_in = net[0].in_features
        self.n_out = net[-1].out_features

        self.x_dim = self.n_in

        self.domain = domain

        self.pre_act_lb = None
        self.pre_act_ub = None

        self.output_bounds = None
        self.output_domain = None

        # equilibrium point
        self.x_eq = None

    def forward(self, x):
        return self.net(x)

    def loss(self, x, xn, l1_lambda=None):
        err = self.forward(x) - xn
        loss = (err**2).sum(dim=-1).mean()
        if l1_lambda is not None:
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda*l1_norm

        return loss

    def set_equibrium(self, x_eq):
        output = self.forward(x_eq)
        err = x_eq - output
        self.net[-1].bias.data = self.net[-1].bias.data + err.detach()
        self.x_eq = x_eq

    def initialize_pre_act_bounds(self):
        pre_act_bounds, output_bounds = find_preactivation_bounds(self.net, self.domain)
        self.pre_act_lb = [item['lb'] for item in pre_act_bounds]
        self.pre_act_ub = [item['ub'] for item in pre_act_bounds]
        self.output_bounds = output_bounds

        self.output_domain = Polyhedron.from_bounds(output_bounds['lb'], output_bounds['ub'])
        return pre_act_bounds, output_bounds

    def set_pre_act_bounds_big_M(self, M = 100.0):
        # using big-M to initialize the pre-activation bounds, mainly for debugging
        if self.pre_act_lb is None:
            self.initialize_pre_act_bounds()

        pre_act_lb = [-M*np.ones(item.shape) for item in self.pre_act_lb]
        pre_act_ub = [ M*np.ones(item.shape) for item in self.pre_act_ub]
        self.pre_act_lb, self.pre_act_ub = pre_act_lb, pre_act_ub

    def add_gurobi_constr(self, gurobi_model, input_var, output_var, mark = ''):
        # add constraints to the gurobi model to enforce the following constraints: output_var = NN(input_var)
        if self.pre_act_lb is None or self.pre_act_ub is None:
            self.initialize_pre_act_bounds()

        nx = self.n_in
        ny = self.n_out

        x = list()
        for i in range(nx):
            x.append(gurobi_model.getVarByName(input_var + '[' + str(i) + ']'))

        y = list()
        for i in range(ny):
            y.append(gurobi_model.getVarByName(output_var + '[' + str(i) + ']'))

        gurobi_model = add_gurobi_constr_for_MLP(gurobi_model, self.net, x, y, self.pre_act_lb, self.pre_act_ub, mark=mark)

        return gurobi_model
