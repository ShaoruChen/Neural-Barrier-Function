import copy

import torch
import torch.nn as nn

from complete_verifier.complete_verification import incomplete_verifier, complete_verifier
from complete_verifier.attack_pgd import attack
from complete_verifier.read_vnnlib import batch_vnnlib

import complete_verifier.arguments as arguments
import yaml
import time
import numpy as np

class DecCondition(nn.Module):
    def __init__(self, barrier_net, A_mat, dyn_net):
        super().__init__()
        device = A_mat.device
        n = A_mat.size(0)
        self.lin_layer = nn.Linear(n, n, bias = False).to(device)
        self.barrier_net = barrier_net
        self.dyn_net = dyn_net

    def forward(self, x):
        return self.lin_layer(self.barrier_net(x)) - self.barrier_net(self.dyn_net(x))
    
def bab_verification(nn_model, data_lb, data_ub, C, rhs, yaml_file_path):
    """
    nn_model: neural network
    data_lb, data_ub: (B, nx) tensors denoting a box input domain
    C: (N, ny) tensor, rhs: (N, 1) tensor, C@y<=rhs denotes the unsafe region
    yaml_file_path: path to the yaml file that stores the BaB parameters
    """
    device = data_lb.device
    data = (data_lb + data_ub)/2
    x_dim = data.size(-1)

    data_range = torch.cat((data_lb.view(-1, 1), data_ub.view(-1, 1)), dim=-1)
    data_range = data_range.cpu().numpy()

    C = C.cpu().numpy()
    rhs = rhs.cpu().numpy()

    spec = [(C[i].reshape(1, -1), rhs[i]) for i in range(rhs.shape[0])]
    vnnlib = [(data_range, spec)]

    # pgd attack
    start_time = time.time()
    verified_status = 'unknown'
    verified_success = False
    verified_status, verified_success, attack_images, attack_margins, all_adv_candidates = attack(
        nn_model, data, data_lb, data_ub, vnnlib,
        verified_status, verified_success)

    # incompete verifier
    verified_status, global_lb, saved_bounds, saved_slopes, activation_opt_params = \
        incomplete_verifier(nn_model, data, data_ub=data_ub, data_lb=data_lb, vnnlib=vnnlib)
    incomplete_verifier_solver_time = time.time() - start_time

    verified_success = verified_status != "unknown"
    if not verified_success:
        model_incomplete, lower_bounds, upper_bounds = saved_bounds[:3]
        lA = saved_bounds[-1]

    batched_vnnlib = batch_vnnlib(vnnlib)
    timeout_threshold = arguments.Config["bab"]["timeout"]

    cplex_processes = None
    refined_betas = None

    vnnlib_shape = [-1, x_dim]
    new_idx = 0
    bab_ret = []

    if not verified_success:
        start_time = time.time()
        bab_sol_lst, is_time_out = complete_verifier(
            nn_model, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
            global_lb, lower_bounds, upper_bounds, new_idx,
            timeout_threshold=timeout_threshold,
            bab_ret=bab_ret, lA=lA, cplex_processes=cplex_processes,
            reference_slopes=saved_slopes, activation_opt_params=activation_opt_params,
            refined_betas=refined_betas, attack_images=all_adv_candidates, attack_margins=attack_margins)
        bab_runtime = time.time() - start_time
        bab_lb = [item['lb'] for item in bab_sol_lst]
        bab_ub = [item['ub'] for item in bab_sol_lst]
        status_lst = [item['result'] for item in bab_sol_lst]
        adv_samples = [item['adv_samples'] for item in bab_sol_lst]

        verified_status = 'safe'
        for status in status_lst:
            if status != 'safe':
                verified_status = 'unsafe'
                break

        bab_result = {'incomplete_lb': global_lb, 'bab_lb': bab_lb, 'bab_ub': bab_ub, 'status': verified_status,
                  'status_lst': status_lst, 'solver_time': bab_runtime,
                  'adv_samples': adv_samples,
                  'bab_sol': bab_sol_lst, 'bab_time_out': is_time_out,
                  'data_lb': data_lb, 'data_ub': data_ub}

        return verified_status, bab_result
    else:
        bab_result = {'incomplete_lb': global_lb, 'bab_lb': None, 'bab_ub': None, 'status': verified_status,
                  'status_lst': None, 'solver_time': incomplete_verifier_solver_time,
                  'adv_samples': None,
                  'bab_sol': None, 'bab_time_out': None,
                  'data_lb': data_lb, 'data_ub': data_ub}
        return verified_status, bab_result

def bab_barrier_fcn_verification(problem, type, yaml_file_path, net=None):
    system = problem.system
    B = problem.barrier_fcn
    device = problem.device

    if net is None:
        net = B.net

    if type == 'xu':
        Xu = problem.Xu
        assert Xu.is_box
        data_lb = torch.from_numpy(Xu.lb.astype('float32')).unsqueeze(0).to(device)
        data_ub = torch.from_numpy(Xu.ub.astype('float32')).unsqueeze(0).to(device)

        C = torch.zeros(1, B.n_out)
        C[0][B.unsafe_index] = 1.0
        rhs = torch.tensor([[0.0]])

        verified_status, bab_result = bab_verification(net, data_lb, data_ub, C, rhs, yaml_file_path)

        if bab_result['adv_samples'] is not None:
            adv_sample_filter_radius_xu = arguments.Config['alg_options']['bab']['adv_sample_filter_radius_xu']

            filtered_samples = [filter_adversarial_samples(item[1], filter_radius=adv_sample_filter_radius_xu) 
                                for item in bab_result['adv_samples'] if item is not None]
            filtered_samples = [item for item in filtered_samples if item is not None]
            if len(filtered_samples) > 0:
                adv_samples = torch.cat(filtered_samples, dim=0)
                adv_samples = adv_samples.detach().cpu().numpy()
            else:
                adv_samples = None
        else:
            adv_samples = None

        return verified_status, adv_samples, bab_result
    elif type == 'x0':
        X0 = problem.X0
        assert X0.is_box
        data_lb = torch.from_numpy(X0.lb.astype('float32')).unsqueeze(0).to(device)
        data_ub = torch.from_numpy(X0.ub.astype('float32')).unsqueeze(0).to(device)

        C = -torch.eye(B.n_out)
        rhs = torch.zeros(B.n_out, 1)

        verified_status, bab_result = bab_verification(net, data_lb, data_ub, C, rhs, yaml_file_path)
                                
        adv_sample_filter_radius_x0 = arguments.Config['alg_options']['bab']['adv_sample_filter_radius_x0']
        if bab_result['adv_samples'] is not None:
            filtered_samples = [filter_adversarial_samples(item[1], filter_radius=adv_sample_filter_radius_x0) 
                                for item in bab_result['adv_samples'] if item is not None]
            filtered_samples = [item for item in filtered_samples if item is not None]
            if len(filtered_samples) > 0:
                adv_samples = torch.cat(filtered_samples, dim=0)
                adv_samples = adv_samples.detach().cpu().numpy()
            else:
                adv_samples = None
        else:
            adv_samples = None

        return verified_status, adv_samples, bab_result

    elif type == 'x':
        # the unsafe region is given by A@B(x) - B(x_+) <= 0
        X = problem.X
        assert X.is_box
        data_lb = torch.from_numpy(X.lb.astype('float32')).unsqueeze(0).to(device)
        data_ub = torch.from_numpy(X.ub.astype('float32')).unsqueeze(0).to(device)

        C = torch.eye(B.n_out)
        rhs = torch.zeros(B.n_out, 1)

        barrier_net = copy.deepcopy(net)
        dyn_net = copy.deepcopy(system)
        A_mat = copy.deepcopy(B.A)

        dec_net = DecCondition(barrier_net, A_mat, dyn_net)

        verified_status, bab_result = bab_verification(dec_net, data_lb, data_ub, C, rhs, yaml_file_path)
        
        adv_sample_filter_radius_x = arguments.Config['alg_options']['bab']['adv_sample_filter_radius_x']
        if bab_result['adv_samples'] is not None:
            filtered_samples = [filter_adversarial_samples(item[1], filter_radius=adv_sample_filter_radius_x)
                                        for item in bab_result['adv_samples'] if item is not None]

            filtered_samples = [item for item in filtered_samples if item is not None]
            if len(filtered_samples) > 0:
                adv_samples = torch.cat(filtered_samples, dim=0)
                adv_samples = adv_samples.detach().cpu().numpy()
            else:
                adv_samples = None
        else:
            adv_samples = None

        return verified_status, adv_samples, bab_result

def filter_adversarial_samples(samples, filter_radius = 1e-2):
    # samples: N x n tensor with the most advesarial samples coming first.
    if samples is None:
        return samples

    numpy_input=False
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)
        numpy_input=True

    new_samples = samples[0].unsqueeze(0)
    filtered_sample = samples[0]

    while True:
        index = torch.sum((samples - filtered_sample)**2, dim=1) > filter_radius**2
        samples = samples[index]
        if len(samples) == 0:
            break

        filtered_sample = samples[0]
        new_samples = torch.cat((new_samples, filtered_sample.unsqueeze(0)), dim=0)

    if numpy_input:
        new_samples = new_samples.cpu().numpy()

    return new_samples


