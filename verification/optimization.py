import gurobipy as gp
import numpy as np
from utils.training import get_nn_info
import torch.optim as optim
import torch
def add_gurobi_constr_for_MLP(gurobi_model, net, x, y, pre_act_lb, pre_act_ub, mark=''):
    nx, ny = len(x), len(y)

    dims, L, weights_list, negative_slope = get_nn_info(net)
    ub_neurons = pre_act_ub
    lb_neurons = pre_act_lb

    # add auxiliary continuous variables
    z = []
    for idx in range(L + 1):
        z.append(gurobi_model.addVars(dims[idx], lb=-gp.GRB.INFINITY, name='z_' + str(mark) + '_' + str(idx)))

    # add binary variables
    t = []
    for idx in range(L):
        t.append(gurobi_model.addVars(dims[idx + 1], name='t_' + str(mark) + '_' + str(idx), vtype=gp.GRB.BINARY))

    gurobi_model.update()

    # encode the ReLU network using mixed integer linear constraints
    gurobi_model.addConstrs((z[0][i] == x[i] for i in range(nx)), name='initialization')

    gurobi_model.addConstrs((y[i] == z[L].prod(
        dict(zip(range(weights_list[L][0].shape[1]), weights_list[L][0][i, :]))) + weights_list[L][1][i]
                             for i in range(ny)), name='outputConstr' + '_step_' + str(mark))

    gurobi_model.addConstrs((z[ell + 1][i] >= z[ell].prod(
        dict(zip(range(weights_list[ell][0].shape[1]), weights_list[ell][0][i, :]))) + weights_list[ell][1][i]
                             for ell in range(L) for i in range(len(z[ell + 1]))),
                            name='binary_1' + '_step_' + str(mark))

    gurobi_model.addConstrs((z[ell + 1][i] >= z[ell].prod(
        dict(zip(range(weights_list[ell][0].shape[1]), negative_slope * weights_list[ell][0][i, :]))) + negative_slope *
                             weights_list[ell][1][i]
                             for ell in range(L) for i in range(len(z[ell + 1]))),
                            name='binary_3' + '_step_' + str(mark))

    gurobi_model.addConstrs((z[ell + 1][i] <= z[ell].prod(
        dict(zip(range(weights_list[ell][0].shape[1]), weights_list[ell][0][i, :]))) + weights_list[ell][1][i]
                             - lb_neurons[ell][i] * (1 - t[ell][i]) for ell in range(L) for i in
                             range(len(z[ell + 1]))), name='binary_2' + '_step_' + str(mark))

    gurobi_model.addConstrs(
        (z[ell + 1][i] <= z[ell].prod(
            dict(zip(range(weights_list[ell][0].shape[1]), negative_slope * weights_list[ell][0][i, :]))) \
         + negative_slope * weights_list[ell][1][i] + ub_neurons[ell][i] * t[ell][i] for ell in range(L) for i in
         range(len(z[ell + 1]))),
        name='binary_4' + '_step_' + str(mark))

    gurobi_model.update()
    return gurobi_model

def search_counterexamples(problem, type, net=None, samples=None, num_iter=100, num_samples=100):
    # search counterexampels locally through gradient descent
    # type = 'x0', 'xu', or 'x' depending on the desired type of counterexamples
    # samples: given initial points to start the search
    
    barrier_fcn = problem.barrier_fcn
    ori_net = barrier_fcn.net
    if net is not None:
        barrier_fcn.net = net

    device = next(barrier_fcn.net.parameters()).device

    if type == 'x0':
        # look for counterexamples that falsify the initial set constraint
        X0 = problem.X0
        if samples is None:
            samples = torch.from_numpy(X0.sample(num_samples)).to(device)

        if X0.lb is not None:
            lb, ub = torch.from_numpy(X0.lb).to(device), torch.from_numpy(X0.ub).to(device)
        else:
            lb, ub = None, None

        loss_fcn, sol = gradient_descent_attack(barrier_fcn.initial_set_loss, samples, lb, ub, num_iter)

        output_val = barrier_fcn(sol)
        max_output_val, _ = torch.max(output_val, dim=-1)

        ce = sol[max_output_val > 1e-8]

        if net is not None:
            barrier_fcn.net = ori_net
        return ce, sol, output_val

    elif type == 'xu':
        # look for counterexamples that falsify the unsafe set constraint
        Xu = problem.Xu
        if samples is None:
            samples = torch.from_numpy(Xu.sample(num_samples)).to(device)

        if Xu.lb is not None:
            lb, ub = torch.from_numpy(Xu.lb).to(device), torch.from_numpy(Xu.ub).to(device)
        else:
            lb, ub = None, None

        loss_fcn, sol = gradient_descent_attack(barrier_fcn.unsafe_set_loss, samples, lb, ub, num_iter)

        output_val = barrier_fcn(sol)
        unsafe_output_val = output_val[:, barrier_fcn.unsafe_index]

        ce = sol[unsafe_output_val < -1e-8]
        if net is not None:
            barrier_fcn.net = ori_net
        return ce, sol, unsafe_output_val

    elif type == 'x':
        # look for counterexamples that falsify the decrease constraint
        X = problem.X
        if samples is None:
            samples = torch.from_numpy(X.sample(num_samples)).to(device)

        if X.lb is not None:
            lb, ub = torch.from_numpy(X.lb).to(device), torch.from_numpy(X.ub).to(device)
        else:
            lb, ub = None, None

        system = problem.system
        loss_fcn, sol = gradient_descent_attack_barrier_decrease(barrier_fcn.decrease_loss, system, samples, lb, ub, num_iter)

        output_val = barrier_fcn(system(sol)) - barrier_fcn(sol)@barrier_fcn.A.T
        max_output_val, _ = torch.max(output_val, dim=-1)

        ce = sol[max_output_val > 1e-8]

        if net is not None:
            barrier_fcn.net = ori_net
        return ce, sol, output_val

def gradient_descent_attack(loss_fcn, var, lb=None, ub=None, num_iter=100,):
    # use projected gradient descent to find adversarial examples
    # var: tensor optimization variable

    var.requires_grad = True
    opt = optim.Adam([var])

    for i in range(num_iter):
        opt.zero_grad()
        loss = -loss_fcn(var)
        loss.backward()
        opt.step()

        if lb is not None:
            var.data.clamp_(min=lb)

        if ub is not None:
            var.data.clamp_(max=ub)

    return -loss, var

def gradient_descent_attack_barrier_decrease(loss_fcn, forward_model, var, lb=None, ub=None, num_iter=100):
    # specified for finding counterexamples of the barrier function decrease condition
    # var: tensor optimization variable

    var.requires_grad = True
    opt = optim.Adam([var])

    for i in range(num_iter):
        opt.zero_grad()

        loss = -loss_fcn(var, forward_model(var))
        loss.backward()
        opt.step()

        if lb is not None:
            var.data.clamp_(min=lb)

        if ub is not None:
            var.data.clamp_(max=ub)

    return -loss, var


