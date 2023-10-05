
import copy
import socket
import random
import pickle
import os
import time
import gc
import torch
import numpy as np
from collections import defaultdict

import complete_verifier.arguments as arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_min
from .jit_precompile import precompile_jit_kernels
from .beta_CROWN_solver import LiRPAConvNet
from .lp_mip_solver import FSB_score
from .attack_pgd import attack
from .utils import Customized, default_onnx_and_vnnlib_loader, parse_run_mode
from .nn4sys_verification import nn4sys_verification
from .batch_branch_and_bound import relu_bab_parallel
from .batch_branch_and_bound_input_split import input_bab_parallel

from .read_vnnlib import batch_vnnlib, read_vnnlib
from .cut_utils import terminate_mip_processes, terminate_mip_processes_by_c_matching

def incomplete_verifier(model_ori, data, data_ub=None, data_lb=None, vnnlib=None):
    norm = arguments.Config["specification"]["norm"]

    # Generally, c should be constructed from vnnlib
    assert len(vnnlib) == 1
    vnnlib = vnnlib[0]
    c = torch.tensor(np.array([item[0] for item in vnnlib[1]])).to(data)
    c_transposed = False
    if c.shape[0] != 1 and data.shape[0] == 1:
        # TODO need a more general solution.
        # transpose c to share intermediate bounds
        c = c.transpose(0, 1)
        c_transposed = True
    arguments.Config["bab"]["decision_thresh"] = torch.tensor(np.array([item[1] for item in vnnlib[1]])).to(data)

    model = LiRPAConvNet(model_ori, in_size=data.shape, c=c)
    print('Model prediction is:', model.net(data))

    eps = arguments.Globals[
        "lp_perturbation_eps"]  # Perturbation value for non-Linf perturbations, None for all other cases.
    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    bound_prop_method = arguments.Config["solver"]["bound_prop_method"]

    _, global_lb, _, _, _, mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, attack_images = model.build_the_model(
        domain, x, data_lb, data_ub, vnnlib,
        stop_criterion_func=stop_criterion_min(arguments.Config["bab"]["decision_thresh"]))

    if (global_lb > arguments.Config["bab"]["decision_thresh"]).all():
        print("verified with init bound!")
        return "safe-incomplete", global_lb, None, None, None

    if arguments.Config["attack"]["pgd_order"] == "middle":
        if attack_images is not None:
            return "unsafe-pgd", None, None, None, None

    # Save the alpha variables during optimization. Here the batch size is 1.
    saved_slopes = defaultdict(dict)
    for m in model.net.relus:
        for spec_name, alpha in m.alpha.items():
            # each slope size is (2, spec, 1, *shape); batch size is 1.
            saved_slopes[m.name][spec_name] = alpha.detach().clone()

    if bound_prop_method == 'alpha-crown':
        # obtain and save relu alphas
        activation_opt_params = dict([(relu.name, relu.dump_optimized_params()) for relu in model.net.relus])
    else:
        activation_opt_params = None

    if c_transposed:
        lower_bounds[-1] = lower_bounds[-1].t()
        upper_bounds[-1] = upper_bounds[-1].t()
        global_lb = global_lb.t()

    saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

    return "unknown", global_lb, saved_bounds, saved_slopes, activation_opt_params


def mip(saved_bounds, labels_to_verify=None):
    lirpa_model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA = saved_bounds
    refined_betas = None

    if arguments.Config["general"]["complete_verifier"] == "mip":
        mip_global_lb, mip_global_ub, mip_status, mip_adv = lirpa_model.build_the_model_mip(
            labels_to_verify=labels_to_verify, save_adv=True)

        if mip_global_lb.ndim == 1:
            mip_global_lb = mip_global_lb.unsqueeze(-1)  # Missing batch dimension.
        if mip_global_ub.ndim == 1:
            mip_global_ub = mip_global_ub.unsqueeze(-1)  # Missing batch dimension.
        print(f'MIP solved lower bound: {mip_global_lb}')
        print(f'MIP solved upper bound: {mip_global_ub}')

        verified_status = "safe-mip"
        # Batch size is always 1.
        labels_to_check = labels_to_verify if labels_to_verify is not None else range(len(mip_status))
        for pidx in labels_to_check:
            if mip_global_lb[pidx] >= 0:
                # Lower bound > 0, verified.
                continue
            # Lower bound < 0, now check upper bound.
            if mip_global_ub[pidx] <= 0:
                # Must be 2 cases: solved with adv example, or early terminate with adv example.
                assert mip_status[pidx] in [2, 15]
                print("verified unsafe-mip with init mip!")
                return "unsafe-mip", mip_global_lb, None, None, None
            # Lower bound < 0 and upper bound > 0, must be a timeout.
            assert mip_status[pidx] == 9 or mip_status[pidx] == -1, "should only be timeout for label pidx"
            verified_status = "unknown-mip"

        print(f"verified {verified_status} with init mip!")
        return verified_status, mip_global_lb, None, None, None

    elif arguments.Config["general"]["complete_verifier"] == "bab-refine":
        print("Start solving intermediate bounds with MIP...")
        score = FSB_score(lirpa_model.net, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

        refined_lower_bounds, refined_upper_bounds, refined_betas = lirpa_model.build_the_model_mip_refine(lower_bounds,
                                                                                                           upper_bounds,
                                                                                                           score=score,
                                                                                                           stop_criterion_func=stop_criterion_min(
                                                                                                               1e-4))

        # shape of the last layer should be (batch, 1) for verified-acc
        refined_lower_bounds[-1] = refined_lower_bounds[-1].reshape(lower_bounds[-1].shape)
        refined_upper_bounds[-1] = refined_upper_bounds[-1].reshape(upper_bounds[-1].shape)

        lower_bounds, upper_bounds, = refined_lower_bounds, refined_upper_bounds
        refined_global_lb = lower_bounds[-1]
        print("refined global lb:", refined_global_lb, "min:", refined_global_lb.min())
        if refined_global_lb.min() >= 0:
            print("Verified safe using alpha-CROWN with MIP improved bounds!")
            return "safe-incomplete-refine", refined_global_lb, lower_bounds, upper_bounds, None

        return "unknown", refined_global_lb, lower_bounds, upper_bounds, refined_betas
    else:
        return "unknown", -float("inf"), lower_bounds, upper_bounds, refined_betas


def bab(unwrapped_model, data, targets, y, data_ub, data_lb,
        lower_bounds=None, upper_bounds=None, reference_slopes=None,
        attack_images=None, c=None, all_prop=None, cplex_processes=None,
        activation_opt_params=None, reference_lA=None, rhs=None,
        model_incomplete=None, timeout=None, refined_betas=None):
    norm = arguments.Config["specification"]["norm"]
    eps = arguments.Globals["lp_perturbation_eps"]  # epsilon for non Linf perturbations, None for all other cases.
    if norm != float("inf"):
        # For non Linf norm, upper and lower bounds do not make sense, and they should be set to the same.
        assert torch.allclose(data_ub, data_lb)

    # This will use the refined bounds if the complete verifier is "bab-refine".
    # FIXME do not repeatedly create LiRPAConvNet which creates a new BoundedModule each time.
    model = LiRPAConvNet(unwrapped_model,
                         in_size=data.shape if not targets.size > 1 else [len(targets)] + list(data.shape[1:]),
                         c=c, cplex_processes=cplex_processes)

    data = data.to(model.device)
    data_lb, data_ub = data_lb.to(model.device), data_ub.to(model.device)
    output = model.net(data).flatten()
    print('Model prediction is:', output)

    # TODO: remove clean prediction test
    # if arguments.Config['attack']['check_clean']:
    #     clean_rhs = c.matmul(output)
    #     print(f'Clean RHS: {clean_rhs}')
    #     if (clean_rhs < rhs).any():
    #         return -np.inf, np.inf, None, None, 'unsafe'

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    cut_enabled = arguments.Config["bab"]["cut"]["enabled"]
    if cut_enabled:
        model.set_cuts(model_incomplete.A_saved, x, lower_bounds, upper_bounds)

    if arguments.Config["bab"]["branching"]["input_split"]["enable"]:
        min_lb, min_ub, glb_record, nb_states, verified_ret = input_bab_parallel(
            model, domain, x, model_ori=unwrapped_model, all_prop=all_prop,
            rhs=rhs, timeout=timeout, branching_method=arguments.Config["bab"]["branching"]["method"])
    else:
        min_lb, min_ub, glb_record, nb_states, adv_samples, verified_ret = relu_bab_parallel(
            model, domain, x,
            refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds,
            activation_opt_params=activation_opt_params, reference_lA=reference_lA,
            reference_slopes=reference_slopes, attack_images=attack_images,
            timeout=timeout, refined_betas=refined_betas, rhs=rhs)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    if min_lb is None:
        min_lb = -np.inf
    if isinstance(min_ub, torch.Tensor):
        min_ub = min_ub.item()
    if min_ub is None:
        min_ub = np.inf

    return min_lb, min_ub, nb_states, glb_record, adv_samples, verified_ret


def update_parameters(model, data_min, data_max):
    if 'vggnet16_2022' in arguments.Config['general']['root_path']:
        perturbed = (data_max - data_min > 0).sum()
        if perturbed > 10000:
            print('WARNING: prioritizing attack due to too many perturbed pixels on VGG')
            print('Setting arguments.Config["attack"]["pgd_order"] to "before"')
            arguments.Config['attack']['pgd_order'] = 'before'


def sort_targets_cls(batched_vnnlib, init_global_lb, init_global_ub, scores, reference_slopes, lA, final_node_name,
                     reverse=False):
    # TODO need minus rhs
    # To sort targets, this must be a classification task, and initial_max_domains
    # is set to 1.
    assert len(batched_vnnlib) == init_global_lb.shape[0] and init_global_lb.shape[1] == 1
    sorted_idx = scores.argsort(descending=reverse)
    batched_vnnlib = [batched_vnnlib[i] for i in sorted_idx]
    init_global_lb = init_global_lb[sorted_idx]
    init_global_ub = init_global_ub[sorted_idx]

    if reference_slopes is not None:
        for m, spec_dict in reference_slopes.items():
            for spec in spec_dict:
                if spec == final_node_name:
                    if spec_dict[spec].size()[1] > 1:
                        # correspond to multi-x case
                        spec_dict[spec] = spec_dict[spec][:, sorted_idx]
                    else:
                        spec_dict[spec] = spec_dict[spec][:, :, sorted_idx]

    if lA is not None:
        lA = [lAitem[:, sorted_idx] for lAitem in lA]

    return batched_vnnlib, init_global_lb, init_global_ub, lA, sorted_idx


def complete_verifier(
        model_ori, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
        init_global_lb, lower_bounds, upper_bounds, index,
        timeout_threshold, bab_ret=None, lA=None, cplex_processes=None,
        reference_slopes=None, activation_opt_params=None, refined_betas=None, attack_images=None,
        attack_margins=None):
    start_time = time.time()
    cplex_cuts = arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]
    sort_targets = arguments.Config["bab"]["sort_targets"]
    bab_attack_enabled = arguments.Config["bab"]["attack"]["enabled"]

    if arguments.Config["general"]["enable_incomplete_verification"]:
        init_global_ub = upper_bounds[-1]
        print('lA shape:', [lAitem.shape for lAitem in lA])
        if bab_attack_enabled:
            # Sort specifications based on adversarial attack margins.
            batched_vnnlib, init_global_lb, init_global_ub, lA, sorted_idx = sort_targets_cls(batched_vnnlib,
                                                                                              init_global_lb,
                                                                                              init_global_ub,
                                                                                              scores=attack_margins.flatten(),
                                                                                              reference_slopes=reference_slopes,
                                                                                              lA=lA,
                                                                                              final_node_name=model_incomplete.net.final_node().name)
            attack_images = attack_images[:, :, sorted_idx]
        else:
            if sort_targets:
                assert not cplex_cuts
                # Sort specifications based on incomplete verifier bounds.
                batched_vnnlib, init_global_lb, init_global_ub, lA, _ = sort_targets_cls(batched_vnnlib, init_global_lb,
                                                                                         init_global_ub,
                                                                                         scores=init_global_lb.flatten(),
                                                                                         reference_slopes=reference_slopes,
                                                                                         lA=lA,
                                                                                         final_node_name=model_incomplete.net.final_node().name)
            if cplex_cuts:
                assert not sort_targets
                # need to sort pidx such that easier first according to initial alpha crown
                batched_vnnlib, init_global_lb, init_global_ub, lA, _ = sort_targets_cls(batched_vnnlib, init_global_lb,
                                                                                         init_global_ub,
                                                                                         scores=init_global_lb.flatten(),
                                                                                         reference_slopes=reference_slopes,
                                                                                         lA=lA,
                                                                                         final_node_name=model_incomplete.net.final_node().name,
                                                                                         reverse=True)
        if reference_slopes is not None:
            reference_slopes_cp = copy.deepcopy(reference_slopes)

    solved_c_rows = []

    sol_lst = []
    is_time_out = False
    for property_idx, properties in enumerate(batched_vnnlib):  # loop of x
        # batched_vnnlib: [x, [(c, rhs, y, pidx)]]
        print(f'\nProperties batch {property_idx}, size {len(properties[0])}')

        # Fixme: reset the timeout threshold for each property
        # timeout = timeout_threshold - (time.time() - start_time)
        timeout = timeout_threshold

        print(f'Remaining timeout: {timeout}')
        start_time_bab = time.time()

        x_range = torch.tensor(properties[0], dtype=torch.get_default_dtype())
        data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
        data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
        x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.

        target_label_arrays = list(properties[1])  # properties[1]: (c, rhs, y, pidx)

        assert len(target_label_arrays) == 1
        c, rhs, y, pidx = target_label_arrays[0]

        if bab_attack_enabled:
            if arguments.Config["bab"]["initial_max_domains"] != 1:
                raise ValueError('To run Bab-attack, please set initial_max_domains to 1. '
                                 f'Currently it is {arguments.Config["bab"]["initial_max_domains"]}.')
            # Attack images has shape (batch, restarts, specs, c, h, w). The specs dimension should already be sorted.
            # Reshape it to (restarts, c, h, w) for this specification.
            this_spec_attack_images = attack_images[:, :, property_idx].view(attack_images.size(1),
                                                                             *attack_images.shape[3:])
        else:
            this_spec_attack_images = None

        if arguments.Config["general"]["enable_incomplete_verification"]:
            # extract lower bound by (sorted) init_global_lb and batch size of initial_max_domains
            this_batch_start_idx = property_idx * arguments.Config["bab"]["initial_max_domains"]
            lower_bounds[-1] = init_global_lb[this_batch_start_idx: this_batch_start_idx + c.shape[0]]
            upper_bounds[-1] = init_global_ub[this_batch_start_idx: this_batch_start_idx + c.shape[0]]

            # trim reference slope by batch size of initial_max_domains accordingly
            if reference_slopes is not None:
                for m, spec_dict in reference_slopes.items():
                    for spec in spec_dict:
                        if spec == model_incomplete.net.final_node().name:
                            if reference_slopes_cp[m][spec].size()[1] > 1:
                                # correspond to multi-x case
                                spec_dict[spec] = reference_slopes_cp[m][spec][:,
                                                  this_batch_start_idx: this_batch_start_idx + c.shape[0]]
                            else:
                                spec_dict[spec] = reference_slopes_cp[m][spec][:, :,
                                                  this_batch_start_idx: this_batch_start_idx + c.shape[0]]

            # trim lA by batch size of initial_max_domains accordingly
            if lA is not None:
                lA_trim = [Aitem[:, this_batch_start_idx: this_batch_start_idx + c.shape[0]] for Aitem in lA]

        print('##### Instance {} first 10 spec matrices: {}\nthresholds: {} ######'.format(index, c[:10],
                                                                                           rhs.flatten()[:10]))

        if np.array(pidx == y).any():
            raise NotImplementedError

        torch.cuda.empty_cache()
        gc.collect()

        c = torch.tensor(c, dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
        rhs = torch.tensor(rhs, dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])

        sol = {'c': c, 'rhs': rhs, 'idx':property_idx, 'lb': None, 'ub': None, 'result': None,
               'nodes': None, 'glb_record': None,
               'runtime': None, 'method': None}

        # extract cplex cut filename
        if cplex_cuts:
            assert arguments.Config["bab"]["initial_max_domains"] == 1

        # Complete verification (BaB, BaB with refine, or MIP).
        if arguments.Config["general"]["enable_incomplete_verification"]:
            assert not arguments.Config["bab"]["branching"]["input_split"]["enable"]
            # Reuse results from incomplete results, or from refined MIPs.
            # skip the prop that already verified
            rlb = list(lower_bounds)[-1]
            if arguments.Config["data"]["num_outputs"] != 1:
                init_verified_cond = rlb.flatten() > rhs.flatten()
                init_verified_idx = np.array(torch.where(init_verified_cond)[0].cpu())
                if init_verified_idx.size > 0:
                    print(
                        f"Initial alpha-CROWN verified for spec index {init_verified_idx} with bound {rlb[init_verified_idx].squeeze()}.")
                    l, ret = init_global_lb[init_verified_idx].cpu().numpy().tolist(), 'safe'
                    bab_ret.append([index, l, 0, time.time() - start_time_bab, pidx])
                init_failure_idx = np.array(torch.where(~init_verified_cond)[0].cpu())
                if init_failure_idx.size == 0:
                    # This batch of x verified by init opt crown
                    sol['runtime'] = time.time() - start_time_bab
                    sol['method'] = 'incomplete'
                    sol['lb'] = rlb.item()
                    sol['result'] = 'safe'
                    sol['ub'] = None
                    sol['adv_samples'] = None
                    sol_lst.append(sol)
                    continue
                print(f"Remaining spec index {init_failure_idx} with "
                      f"bounds {rlb[init_failure_idx]} need to verify.")
                assert len(np.unique(y)) == 1 and len(rhs.unique()) == 1
            else:
                init_verified_cond, init_failure_idx, y = torch.tensor([1]), np.array(0), np.array(0)

            if reference_slopes is not None:
                LiRPAConvNet.prune_reference_slopes(reference_slopes, ~init_verified_cond,
                                                    model_incomplete.net.final_node().name)
            if lA is not None:
                lA_trim = LiRPAConvNet.prune_lA(lA_trim, ~init_verified_cond)

            lower_bounds[-1] = lower_bounds[-1][init_failure_idx]
            upper_bounds[-1] = upper_bounds[-1][init_failure_idx]
            # TODO change index [0:1] to [torch.where(~init_verified_cond)[0]] can handle more general vnnlib for multiple x
            l, u, nodes, glb_record, adv_samples, ret = bab(
                model_ori, x[0:1], init_failure_idx, y=np.unique(y),
                data_ub=data_max[0:1], data_lb=data_min[0:1],
                lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                c=c[torch.where(~init_verified_cond)[0]],
                reference_slopes=reference_slopes, cplex_processes=cplex_processes, rhs=rhs[0:1],
                activation_opt_params=activation_opt_params, reference_lA=lA_trim,
                model_incomplete=model_incomplete, timeout=timeout, refined_betas=refined_betas,
                attack_images=this_spec_attack_images)
            bab_ret.append([index, float(l), nodes, time.time() - start_time_bab, init_failure_idx.tolist()])
            sol['runtime'] = time.time() - start_time_bab
            sol['method'] = 'bab'
            sol['adv_samples'] = adv_samples
            sol['lb'], sol['ub'], sol['nodes'], sol['glb_record'], sol['result'] = float(l), float(u), nodes, glb_record, ret
        else:
            assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
            assert not arguments.Config["bab"]["attack"]["enabled"], "BaB-attack must be used with incomplete verifier."
            # input split also goes here directly
            l, u, nodes, glb_record, adv_samples, ret = bab(
                model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min, c=c,
                all_prop=target_label_arrays, cplex_processes=cplex_processes,
                rhs=rhs, timeout=timeout, attack_images=this_spec_attack_images)
            bab_ret.append([index, l, nodes, time.time() - start_time_bab, pidx])
            sol['runtime'] = time.time() - start_time_bab
            sol['method'] = 'bab'
            sol['adv_samples'] = adv_samples
            sol['lb'], sol['ub'], sol['nodes'], sol['glb_record'], sol['result'] = float(l), float(u), nodes, glb_record, ret

        sol_lst.append(sol)

        # terminate the corresponding cut inquiry process if exists
        if cplex_cuts:
            solved_c_rows.append(c)
            terminate_mip_processes_by_c_matching(cplex_processes, solved_c_rows)

        timeout = timeout_threshold - (time.time() - start_time)

        if timeout < 0:
            is_time_out = True
            break

    return sol_lst, is_time_out
