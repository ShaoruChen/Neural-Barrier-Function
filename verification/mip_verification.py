import gurobipy as gp
from dynamics.models import Auto_Dynamics, Polyhedral_Set
import numpy as np
def mip_nn_output_bounds(net, domain, mode='min'):
    net_model = Auto_Dynamics(net, domain)
    net_model.initialize_pre_act_bounds()
    net_model.set_pre_act_bounds_big_M()

    poly_set = Polyhedral_Set(domain.A, domain.b)

    input_dim, output_dim = net[0].in_features, net[-1].out_features

    gurobi_model = gp.Model('test')
    gurobi_model.Params.NonConvex = 2

    var_dict = {'x': input_dim, 'y': output_dim}

    for name, dim in var_dict.items():
        gurobi_model.addVars(dim, lb = -gp.GRB.INFINITY, name = name)
    gurobi_model.update()

    gurobi_model = net_model.add_gurobi_constr(gurobi_model, 'x', 'y', mark='dynamics')
    gurobi_model = poly_set.add_gurobi_constr(gurobi_model, 'x', mark='input_set')
    gurobi_model.update()

    outputs = list()
    for i in range(output_dim):
        outputs.append(gurobi_model.getVarByName('y[' + str(i) + ']'))

    inputs = list()
    for i in range(input_dim):
        inputs.append(gurobi_model.getVarByName('x[' + str(i) + ']'))

    bounds_lst =[]
    mip_sol_lst = []
    for i in range(output_dim):
        obj = outputs[i]
        if mode == 'min':
            gurobi_model.setObjective(obj, gp.GRB.MINIMIZE)
        elif mode == 'max':
            gurobi_model.setObjective(obj, gp.GRB.MAXIMIZE)
        else:
            raise ValueError('MIP optimization mode not recognized.')

        gurobi_model.optimize()
        solver_time = gurobi_model.Runtime
        gurobi_status = gurobi_model.Status

        value = gurobi_model.objVal
        bounds_lst.append(value)

        y_sol = np.array([outputs[i].X for i in range(output_dim)]).reshape(1, -1)
        x_sol = np.array([inputs[i].X for i in range(input_dim)]).reshape(1, -1)

        mip_sol_lst.append({'value': value, 'x_sol': x_sol, 'y_sol': y_sol, 'solver_time': solver_time, 'status': gurobi_status})

    return mip_sol_lst, bounds_lst