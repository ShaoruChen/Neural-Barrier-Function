import sys
# Append the path to the directory containing your module to sys.path
import os
project_dir = '/home/shaoruchen/Desktop/learning_basis/learning-basis-functions/codes/learning_barrier'
sys.path.append(project_dir)

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

import torch
import torch.nn as nn
import numpy as np

from utils.sampling import scale_polyhedron

from dynamics.models import Barrier_Fcn, Polyhedral_Set, OpenLoopDynamics, ClosedLoopDynamics, Controller

from dynamics.systems import predator_prey, NN_Dynamics
from dynamics.model_training import train_autonomous_nn_dynamics
from training.train_barrier import Training_Options, Trainer, CE_Sampling_Options
from pympc.geometry.polyhedron import Polyhedron

import os

from cutting_plane.ACCPM import Problem, ACCPM_Options
import time
import random

import complete_verifier.arguments as arguments

import argparse
import yaml

from utils.sampling import set_seed
import json
import warnings

from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    arguments.Config.parse_config()

    seed = arguments.Config['general']['seed']
    device = torch.device(arguments.Config['general']['device'])
    set_seed(seed, device=arguments.Config['general']['device'])

    data_dir = 'data_0926'
    dataset_path = os.path.join(script_directory, data_dir, 'dataset.p')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    B_dataset_path = os.path.join(script_directory, data_dir, 'B_train_dataset.p')
    nn_model_path = os.path.join(script_directory, data_dir, 'nn_model.p')
    B_model_path = os.path.join(script_directory, data_dir, 'B_model.p')

    result_path = os.path.join(script_directory, data_dir, 'ACCPM_result.p')

    #################### initialize the verification problem ##############################
    x_dim = 6
    u_dim = 3

    # state space
    # x_lb = np.array([4.0, 3.5,  2.5,  -1.0, -1.0, -1.0]).astype('float32')
    # x_ub = np.array([5.5, 5.0, 3.5, 1.0, 1.0, 1.0]).astype('float32')
    # x_lb = np.array([-5.0, -5.0, -5.0, -1.0, -1.0, -1.0]).astype('float32')
    # x_ub = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]).astype('float32')

    # working example
    # x_lb = np.array([4.0, 4.0, 2.7, 0.9, -0.01, -0.01]).astype('float32')
    # x_ub = np.array([5.0, 5.0, 3.2, 1.0, 0.01, 0.01]).astype('float32')

    x_lb = np.array([4.0, 4.0, 2.5, -1.0, -1.0, -1.0]).astype('float32')
    x_ub = np.array([5.0, 5.0, 3.5, 1.0, 1.0, 1.0]).astype('float32')

    domain = Polyhedron.from_bounds(x_lb, x_ub)
    domain = scale_polyhedron(domain, 1.0)

    X = Polyhedral_Set(domain.A, domain.b)
    X.lb, X.ub = x_lb, x_ub
    X.is_box = True

    # the initial region
    x0_min = np.array([4.69, 4.65,  2.975,  0.9499, -0.0001, -0.0001]).astype('float32')
    x0_max = np.array([4.71, 4.75, 3.025, 0.9501, 0.0001, 0.0001]).astype('float32')
    X0 = Polyhedron.from_bounds(x0_min, x0_max)
    X0 = Polyhedral_Set(X0.A, X0.b)
    X0.lb, X0.ub = x0_min, x0_max
    X0.is_box = True

    # the unsafe region
    # xu_min = np.array([4.6, 4.3, 2.5, -1.0, -1.0, -1.0]).astype('float32')
    # xu_max = np.array([4.7, 4.4, 3.5, 1.0, 1.0, 1.0]).astype('float32')

    # working example
    # xu_min = np.array([4.4, 4.3, 2.9, 0.95, -0.0001, -0.0001]).astype('float32')
    # xu_max = np.array([4.45, 4.35, 3.0, 1.0, 0.0001, 0.0001]).astype('float32')

    xu_min = np.array([4.4, 4.3, 2.9, 0.95, -0.1, -0.1]).astype('float32')
    xu_max = np.array([4.45, 4.35, 3.0, 1.0, 0.1, 0.1]).astype('float32')

    Xu = Polyhedron.from_bounds(xu_min, xu_max)
    Xu = Polyhedral_Set(Xu.A, Xu.b)
    Xu.lb, Xu.ub = xu_min, xu_max
    Xu.is_box = True

    #################### generate autonomous nn dynamics ##############################
    # load dynamics model
    model_config_path = os.path.join(script_directory, 'model_data/quadRotor.json')
    with open(model_config_path, 'r') as file:
        config = json.load(file)

    A = torch.Tensor(config['A']).to(device)
    B = torch.Tensor(config['B']).to(device)
    c = torch.Tensor(config['c']).to(device)

    nn_controller_path = os.path.join(script_directory, 'model_data/quadRotorNormalV3.0.pth')
    quad_dyn = NeuralNetwork(nn_controller_path, A, B, c)
    quad_dyn = quad_dyn.to(device)

    # create open loop dynamics

    # TODO: need to figure out the input constraints
    # u_lb = np.array([-0.42, -0.05, 6.0]).astype('float32')
    # u_ub = np.array([0.01, 0.45, 10.5]).astype('float32')

    u_lb = np.array([-0.364, -0.364, 0.0]).astype('float32')
    u_ub = np.array([0.364, 0.364, 19.62]).astype('float32')

    aug_x_lb, aug_x_ub = np.concatenate((x_lb, u_lb)), np.concatenate((x_ub, u_ub))
    aug_x_domain = Polyhedron.from_bounds(aug_x_lb, aug_x_ub)

    open_loop_dyn_layer = nn.Sequential(nn.Linear(x_dim + u_dim, x_dim)).to(device)
    open_loop_dyn_layer[0].weight.data = torch.cat((A, B), dim=-1).to(device)
    # Note that the bias term c provided has the wrong dimension.
    open_loop_dyn_layer[0].bias.data = c.to(device)
    open_loop_dyn = OpenLoopDynamics(open_loop_dyn_layer, x_dim, u_dim, aug_x_domain)

    controller_net = quad_dyn.Linear
    u_lb, u_ub = torch.from_numpy(u_lb).to(device), torch.from_numpy(u_ub).to(device)
    controller = Controller(controller_net, domain, u_lb=u_lb, u_ub=u_ub)

    dynamics = ClosedLoopDynamics(open_loop_dyn, controller)

    # simulate trajectories
    # init_states = X0.sample(1000)
    # init_states = torch.from_numpy(init_states).to(device)
    # num_step = 100
    # traj = dynamics.simulate_trajectory(init_states, num_step)
    # is_safe = True
    # for i in range(num_step+1):
    #     state = traj[:,i,:]
    #     control_input = controller(state)
    #     x_l, _ = state.min(dim=0)
    #     x_u, _ = state.max(dim=0)
    #     x_l, x_u = x_l.detach().cpu().numpy(), x_u.detach().cpu().numpy()
    #     reach_set_over = Polyhedron.from_bounds(x_min = x_l, x_max = x_u)
    #     intersection = Xu.set.intersection(reach_set_over)
    #     if not intersection.empty:
    #         is_safe = False
    #         print(i)
    #         warnings.warn('Reachable set over-approximation intersects the unsafe region!')
    #
    # estimate the controller outout lower and upper bounds
    # traj = torch.cat([traj[:, i, :] for i in range(traj.size(1))])
    # lb, _ = controller(traj).min(dim=0)
    # ub, _ =  controller(traj).max(dim=0)
    #
    # x_l = traj.min(dim=0)
    # x_u = traj.max(dim=0)

    # import matplotlib.pyplot as plt
    # traj = traj.detach().numpy()
    # plt.figure()
    # for i in range(num_step+1):
    #     plt.scatter(traj[:,i,0], traj[:,i,1], marker='.', alpha=0.5)
    # Xu.set.plot()
    # plt.show()

    #################### train a barrier function ##############################
    barrier_dim = arguments.Config['alg_options']['barrier_fcn']['barrier_output_dim']
    # barrier_net = nn.Sequential(nn.Linear(x_dim, 200),
    #                             nn.ReLU(),
    #                             nn.Linear(200, 120),
    #                             nn.ReLU(),
    #                             nn.Linear(120, 80),
    #                             nn.ReLU(),
    #                             nn.Linear(80, 40),
    #                             nn.ReLU(),
    #                             nn.Linear(40, 20),
    #                             nn.ReLU(),
    #                             nn.Linear(20, barrier_dim))

    barrier_net = nn.Sequential(nn.Linear(x_dim, 100),
                            nn.ReLU(),
                            nn.Linear(100, 80),
                            nn.ReLU(),
                            nn.Linear(80, 60),
                            nn.ReLU(),
                            nn.Linear(60, 40),
                            nn.ReLU(),
                            nn.Linear(40, 20),
                            nn.ReLU(),
                            nn.Linear(20, barrier_dim))

    barrier_fcn = Barrier_Fcn(barrier_net, domain).to(device)

    problem = Problem(dynamics, barrier_fcn, X, X0, Xu)

    verification_method = arguments.Config['alg_options']['verification_method']
    trainer = Trainer(problem, verification_method=verification_method)

    if arguments.Config['alg_options']['barrier_fcn']['dataset']['collect_samples']:
        num_samples_x0 = arguments.Config['alg_options']['barrier_fcn']['dataset']['num_samples_x0']
        num_samples_xu = arguments.Config['alg_options']['barrier_fcn']['dataset']['num_samples_xu']
        num_samples_x = arguments.Config['alg_options']['barrier_fcn']['dataset']['num_samples_x']

        sampling_options = {'num_samples_x0': num_samples_x0, 'num_samples_xu': num_samples_xu,
                            'num_samples_x': num_samples_x}
        trainer.generate_training_samples(B_dataset_path, sampling_options)
    else:
        trainer.load_sample_set(B_dataset_path)

    method = arguments.Config['alg_options']['train_method']

    if method == 'fine-tuning':
        train_result_path = os.path.join(script_directory, data_dir, 'train_iter_accpm_' + str(seed) + '.p')
        B_iter_model_path = os.path.join(script_directory, data_dir, 'B_iter_model_accpm_' + str(seed) + '.p')
    elif method == 'verification-only':
        train_result_path = os.path.join(script_directory, data_dir, 'train_iter_verify_' + str(seed) + '.p')
        B_iter_model_path = os.path.join(script_directory, data_dir, 'B_iter_model_verify_' + str(seed) + '.p')
    else:
        raise ValueError(f'Method {method} is not supported.')

    start_time = time.time()
    num_iter = arguments.Config['alg_options']['barrier_fcn']['train_options']['num_iter']

    l1_lambda = arguments.Config['alg_options']['barrier_fcn']['train_options']['l1_lambda']
    num_epochs = arguments.Config['alg_options']['barrier_fcn']['train_options']['num_epochs']
    early_stop_tol = arguments.Config['alg_options']['barrier_fcn']['train_options']['early_stopping_tol']
    update_A_freq = arguments.Config['alg_options']['barrier_fcn']['train_options']['update_A_freq']
    training_opt = Training_Options(l1_lambda=l1_lambda, num_epochs=num_epochs, early_stop_tol=early_stop_tol,
                                    update_A_freq=update_A_freq)

    num_ce_samples = arguments.Config['alg_options']['ce_sampling']['num_ce_samples']
    radius = arguments.Config['alg_options']['ce_sampling']['radius']
    opt_iter = arguments.Config['alg_options']['ce_sampling']['opt_iter']
    num_ce_samples_accpm = arguments.Config['alg_options']['ce_sampling']['num_ce_samples_accpm']
    ce_sampling_opt = CE_Sampling_Options(num_ce_samples=num_ce_samples, num_ce_samples_accpm=num_ce_samples_accpm,
                                          radius=radius, opt_iter=opt_iter)

    accpm_opt = ACCPM_Options(max_iter=arguments.Config['alg_options']['ACCPM']['max_iter'],
                              sample_ce_opt=ce_sampling_opt)
    train_method = arguments.Config['alg_options']['train_method']

    batch_size = arguments.Config['alg_options']['barrier_fcn']['train_options']['B_batch_size']
    train_timeout = arguments.Config['alg_options']['barrier_fcn']['train_options']['train_timeout']

    status, num_queries, results = trainer.train_and_verify(num_iter,
                                                            method=train_method,
                                                            batch_size=batch_size,
                                                            save_model_path=B_iter_model_path,
                                                            training_opt=training_opt,
                                                            ce_sampling_opt=ce_sampling_opt,
                                                            accpm_opt=accpm_opt,
                                                            timeout=train_timeout)

    runtime = time.time() - start_time
    print(
        f'Method: {train_method}. Verification status: {status}. Num. of verifier queries: {num_queries}, wall-clock time:{runtime} s.')

    data_to_save = {'training_results': results, 'seed': seed, 'method': method}

    torch.save(data_to_save, train_result_path)





