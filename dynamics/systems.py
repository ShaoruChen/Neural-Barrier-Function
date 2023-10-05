import torch
import numpy as np
import torch.nn as nn

from verification.nn_verification import find_preactivation_bounds
from verification.optimization import add_gurobi_constr_for_MLP
from utils.training import get_nn_info
import gurobipy as gp
from pympc.geometry.polyhedron import Polyhedron


class NN_Dynamics(nn.Module):
    # system described by x_+ = f(x) where f is a multi-layer perception
    def __init__(self, n_in, n_out, domain):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 40), nn.ReLU(), nn.Linear(40, 20), nn.ReLU(), nn.Linear(20, n_out))
        self.n_in = n_in
        self.n_out = n_out

        self.x_dim = n_in

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

    def add_gurobi_constr(self, gurobi_model, input_var, output_var, mark = ''):
        # add constraints to the gurobi model to enforce the following constraints: output_var = NN(input_var)
        nx = self.n_in 
        ny = self.n_out

        x = list()
        for i in range(nx):
            x.append(gurobi_model.getVarByName(input_var + '[' + str(i) + ']'))

        y = list()
        for i in range(nx):
            y.append(gurobi_model.getVarByName(output_var + '[' + str(i) + ']'))

        gurobi_model = add_gurobi_constr_for_MLP(gurobi_model, self.net, x, y, self.pre_act_lb, self.pre_act_ub, mark=mark)

        # # dims = [input dim, dims of hidden layers, output dim], L = num. of hidden layers
        # # TODO: replace the gurobi constraints by the new functions
        # dims, L, weights_list, negative_slope = get_nn_info(self.net)
        # ub_neurons = self.pre_act_ub
        # lb_neurons = self.pre_act_lb
        #
        # # add auxiliary continuous variables
        # z = []
        # for idx in range(L + 1):
        #     z.append(gurobi_model.addVars(dims[idx], lb=-gp.GRB.INFINITY, name='z_' + str(mark) + '_' + str(idx)))
        #
        # # add binary variables
        # t = []
        # for idx in range(L):
        #     t.append(gurobi_model.addVars(dims[idx + 1], name='t_' + str(mark) + '_' + str(idx), vtype=gp.GRB.BINARY))
        #
        # gurobi_model.update()
        #
        # # encode the ReLU network using mixed integer linear constraints
        # gurobi_model.addConstrs((z[0][i] == x[i] for i in range(nx)), name='initialization')
        #
        # gurobi_model.addConstrs((y[i] == z[L].prod(
        #     dict(zip(range(weights_list[L][0].shape[1]), weights_list[L][0][i, :]))) + weights_list[L][1][i]
        #                          for i in range(ny)), name='outputConstr' + '_step_' + str(mark))
        #
        # gurobi_model.addConstrs((z[ell + 1][i] >= z[ell].prod(
        #     dict(zip(range(weights_list[ell][0].shape[1]), weights_list[ell][0][i, :]))) + weights_list[ell][1][i]
        #                          for ell in range(L) for i in range(len(z[ell + 1]))),
        #                         name='binary_1' + '_step_' + str(mark))
        #
        # gurobi_model.addConstrs((z[ell + 1][i] >= 0 for ell in range(L) for i in range(len(z[ell + 1]))),
        #                         name='binary_3' + '_step_' + str(mark))
        #
        # gurobi_model.addConstrs((z[ell + 1][i] <= z[ell].prod(
        #     dict(zip(range(weights_list[ell][0].shape[1]), weights_list[ell][0][i, :]))) + weights_list[ell][1][i]
        #                          - lb_neurons[ell][i] * (1 - t[ell][i]) for ell in range(L) for i in
        #                          range(len(z[ell + 1]))), name='binary_2' + '_step_' + str(mark))
        #
        # gurobi_model.addConstrs(
        #     (z[ell + 1][i] <= ub_neurons[ell][i] * t[ell][i] for ell in range(L) for i in range(len(z[ell + 1]))),
        #     name='binary_4' + '_step_' + str(mark) )
        #
        # gurobi_model.update()

        return gurobi_model

######## test examples ##############
def predator_prey(x):
    # x: torch tensor
    cross_terms = x[:,0]*x[:,1]
    y = x@torch.tensor([[0.5, 0.0], [0.0,-0.5]]) + cross_terms.view(-1, 1)@torch.tensor([[1.0, -1.0]])
    return y





        

