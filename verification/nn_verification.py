import torch
import torch.nn as nn

from utils.sampling import find_bounding_box
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np

def find_preactivation_bounds(net, domain):
    device = next(net.parameters()).device

    input_lb, input_ub = find_bounding_box(domain)
    input_lb, input_ub = np.array(input_lb).astype('float32'), np.array(input_ub).astype('float32')
    input_lb, input_ub = torch.from_numpy(input_lb).unsqueeze(0).to(device), torch.from_numpy(input_ub).unsqueeze(0).to(device)

    pre_act_bounds = []

    layer_list = []
    for layer in net:
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.Sigmoid):
            temp_net = nn.Sequential(*layer_list)
            pre_act_lb, pre_act_ub = output_Lp_bounds_LiRPA(temp_net, input_lb, input_ub, method = 'backward')
            pre_act_bounds.append({'lb': pre_act_lb.squeeze(0).detach().cpu().numpy(), 'ub': pre_act_ub.squeeze(0).detach().cpu().numpy()})

            layer_list.append(layer)
        else:
            layer_list.append(layer)

    temp_net = nn.Sequential(*layer_list)
    output_lb, output_ub = output_Lp_bounds_LiRPA(temp_net, input_lb, input_ub, method='backward')
    output_bounds = {'lb': output_lb.squeeze(0).detach().cpu().numpy(), 'ub': output_ub.squeeze(0).detach().cpu().numpy()}

    return pre_act_bounds, output_bounds


def output_Lp_bounds_LiRPA(nn_model, lb, ub, method = 'backward', C = None):
    center = (lb + ub)/2
    radius = (ub - lb)/2

    model = BoundedModule(nn_model, center)
    ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
    # Make the input a BoundedTensor with perturbation
    my_input = BoundedTensor(center, ptb)
    # Compute LiRPA bounds
    if 'optimized' in method or 'alpha' in method:
        model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})

    output = model.compute_bounds(x=(my_input,),C = C, method=method)
    output_lb, output_ub = output[0], output[1]
    return output_lb, output_ub
    