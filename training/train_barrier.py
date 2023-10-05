import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling import generate_training_data, sample_batch, generate_traj_samples, plot_multiple_traj
from utils.sampling import uniform_random_sample_from_Polyhedron, scale_polyhedron

import pickle
import numpy as np

from cutting_plane.ACCPM import ACCPM_Alg, ACCPM_Options, gurobi_options
from verification.optimization import search_counterexamples
import copy
from pympc.geometry.polyhedron import Polyhedron
from tqdm import tqdm
import time
from utils.training import setup_relu, extract_relu_structure

import complete_verifier.arguments as arguments
class Training_Options:
    def __init__(self, l1_lambda=None, num_epochs=50, early_stop_tol=1e-10, update_A_freq=1):
        self.l1_lambda = l1_lambda
        self.num_epochs = num_epochs
        self.early_stop_tol = early_stop_tol
        self.update_A_freq = update_A_freq

class CE_Sampling_Options:
    def __init__(self, num_ce_samples=20, num_ce_samples_accpm=5, radius=0.1,opt_iter=100):
        self.num_ce_samples = num_ce_samples
        self.radius = radius
        self.opt_iter = opt_iter
        # number of samples to augment the sample set in ACCPM
        self.num_ce_samples_accpm = num_ce_samples_accpm

class Trainer:
    # Given a dynamical system and (workspace, initial set, unsafe set), we train a vector barrier function to certify safety
    def __init__(self, problem, verification_method = 'bab', options=None):
        # problem contains all parameters describing the verification problem
        self.problem = problem

        # TODO: not sure if accpm_alg will share use the same barrier function
        bab_yaml_path = arguments.Config['alg_options']['bab']['bab_yaml_path']

        mip_solver_options = gurobi_options(time_limit=arguments.Config['alg_options']['mip']['time_limit'],
                                            MIPFocus=arguments.Config['alg_options']['mip']['MIPFocus'])

        self.accpm_alg = ACCPM_Alg(problem, method = verification_method, mip_solver_options=mip_solver_options, bab_yaml_path=bab_yaml_path)

        self.system = problem.system
        self.barrier_fcn = problem.barrier_fcn
        self.X, self.X0, self.Xu = problem.X, problem.X0, problem.Xu
        self.x_dim = problem.system.x_dim

        self.sample_set = None

        # maximum number of training samples for each condition
        self.samples_pool_size = arguments.Config['alg_options']['barrier_fcn']['train_options']['samples_pool_size']

        self.device = next(self.barrier_fcn.net.parameters()).device

    def load_sample_set(self, load_data_path):
        self.sample_set = pickle.load(open(load_data_path, 'rb'))

    def generate_training_samples(self, save_data_path, sampling_options=None):
        if sampling_options is not None:
            num_samples_x0 = sampling_options['num_samples_x0'] if 'num_samples_x0' in sampling_options.keys() else 2000
            num_samples_xu = sampling_options['num_samples_xu'] if 'num_samples_xu' in sampling_options.keys() else 2000
            num_samples_x = sampling_options['num_samples_x'] if 'num_samples_x' in sampling_options.keys() else 5000
        else:
            num_samples_x0 = arguments.Config['alg_options']['barrier_fcn']['dataset']['num_samples_x0']
            num_samples_xu = arguments.Config['alg_options']['barrier_fcn']['dataset']['num_samples_xu']
            num_samples_x = arguments.Config['alg_options']['barrier_fcn']['dataset']['num_samples_x']


        x0_samples = self.X0.sample(num_samples_x0)
        xu_samples = self.Xu.sample(num_samples_xu)

        x_samples, xn_samples = generate_training_data(self.system, self.X.set, num_samples_x)

        dataset = {'x': x_samples.astype('float32'), 'xn': xn_samples.astype('float32'), 'x0': x0_samples.astype('float32'), 'xu': xu_samples.astype('float32')}
        pickle.dump(dataset, open(save_data_path, 'wb'))

        self.sample_set = dataset

    def train_barrier_fcn_from_samples(self, save_model_path, l1_lambda=None, num_epochs=100,
                                       early_stop_tol=1e-10, dataset=None, update_A=True, batch_size=50):
        # generate training samples for learning the Lyapunov function
        device = self.device
        barrier_fcn = self.barrier_fcn

        if dataset is None:
            dataset = self.sample_set
            if self.sample_set is None:
                raise ValueError('Training data set is None.')

        # construct custom dataset
        x0_samples, xu_samples = torch.from_numpy(dataset['x0'].astype('float32')).to(device), torch.from_numpy(dataset['xu'].astype('float32')).to(device)
        x_samples, xn_samples = torch.from_numpy(dataset['x'].astype('float32')).to(device), torch.from_numpy(dataset['xn'].astype('float32')).to(device)
        aug_x_samples = torch.cat((x_samples, xn_samples), dim=-1)

        x0_labels = torch.ones((x0_samples.size(0),), dtype=x0_samples.dtype).to(device)
        xu_labels = torch.ones((xu_samples.size(0),), dtype=xu_samples.dtype).to(device)
        x_labels = torch.ones((x_samples.size(0),), dtype=x_samples.dtype).to(device)

        x0_dataset = torch.utils.data.TensorDataset(x0_samples, x0_labels)
        xu_dataset = torch.utils.data.TensorDataset(xu_samples, xu_labels)
        x_dataset = torch.utils.data.TensorDataset(aug_x_samples, x_labels)

        num_batches = int(np.ceil(max(len(x0_dataset), len(xu_dataset), len(x_dataset))/batch_size))

        x0_dataloader = torch.utils.data.DataLoader(x0_dataset, batch_size=int(np.ceil(len(x0_dataset) / num_batches)), shuffle=True)
        xu_dataloader = torch.utils.data.DataLoader(xu_dataset, batch_size=int(np.ceil(len(xu_dataset) / num_batches)), shuffle=True)
        x_dataloader = torch.utils.data.DataLoader(x_dataset, batch_size=int(np.ceil(len(x_dataset) / num_batches)), shuffle=True)

        best_loss = barrier_fcn.loss(x0_samples, xu_samples, x_samples, xn_samples)
        best_training_params = self._get_current_training_params()

        barrier_fcn.train()
        if update_A:
            opt = torch.optim.Adam(barrier_fcn.parameters())
        else:
            # do not update the A matrix
            opt = torch.optim.Adam(barrier_fcn.net.parameters())

        for epoch in range(num_epochs):
            iter_x0_samples = iter(x0_dataloader)
            iter_xu_samples = iter(xu_dataloader)
            iter_x_samples = iter(x_dataloader)

            for i in range(min([len(x0_dataloader), len(xu_dataloader), len(x_dataloader)])):
                x0_batch, _ = next(iter_x0_samples)
                xu_batch, _ = next(iter_xu_samples)
                aug_x_batch, _ = next(iter_x_samples)
                x_batch, xn_batch = aug_x_batch[:, :self.x_dim], aug_x_batch[:, self.x_dim:]

                opt.zero_grad()

                loss = barrier_fcn.loss(x0_batch, xu_batch, x_batch, xn_batch, l1_lambda=l1_lambda)
                loss.backward()

                opt.step()

                # project A back to the nonnegative cone
                barrier_fcn.A.data = barrier_fcn.A.clamp(min=0.0)

                print(f'epoch {epoch} iter. {i} loss: {loss.item():.8f}')

            total_loss = barrier_fcn.loss(x0_samples, xu_samples, x_samples, xn_samples)
            if total_loss <= early_stop_tol:
                if save_model_path is not None:
                    torch.save(barrier_fcn.state_dict(), save_model_path)
                return

            if total_loss <= best_loss:
                best_training_params = self._get_current_training_params()
                best_loss = total_loss

        self._set_barrier_fcn_params(best_training_params)
        if save_model_path is not None:
            torch.save(barrier_fcn.state_dict(), save_model_path)

    def load_barrier_fcn(self, model_path):
        self.barrier_fcn.load_state_dict(torch.load(model_path))

    def _get_current_training_params(self):
        params = {}
        params['barrier_network'] = copy.deepcopy(self.barrier_fcn.net.state_dict())
        params['A'] = self.barrier_fcn.A.data.clone()
        return params

    def _set_barrier_fcn_params(self, params):
        self.barrier_fcn.net.load_state_dict(params['barrier_network'])
        self.barrier_fcn.A.data = params['A'].clone()

    def verify_barrier_fcn(self, barrier_fcn=None):
        status, ce, sol_record = self.accpm_alg.verify_candidate_barrier(barrier_fcn)
        solver_time = sum([item['solver_time'] for item in sol_record])
        return status, ce, solver_time, sol_record

    def update_barrier_fcn_by_ACCPM(self, accpm_opt=None):
        if accpm_opt is None:
            accpm_opt = ACCPM_Options(max_iter=arguments.Config['alg_options']['ACCPM']['max_iter'])

        status, num_queries, results = self.accpm_alg.run(accpm_opt)

        if status != 'infeasible':
            final_weights = results['output_coeff']
            C, b = final_weights['C'], final_weights['b']

            self.barrier_fcn.net[-1].weight.data = torch.from_numpy(C.astype('float32')).to(self.device)
            self.barrier_fcn.net[-1].bias.data = torch.from_numpy(b.astype('float32')).to(self.device)

            ce = results['last_ce_set']
        else:
            # in case no valid last linear layer exists, we do not update the barrier function candidate
            ce = results['init_ce_set']

        # calculate solver time
        verifier_record = results['verifier_record']
        if len(verifier_record) > 0:
            verifier_solver_time = sum([verifier_record[i][j]['solver_time']
                                    for i in range(len(verifier_record)) for j in range(len(verifier_record[i]))])
        else:
            verifier_solver_time = 0.0

        learner_record = results['learner_record']
        if len(learner_record) > 0:
            learner_solver_time = sum([item['solver_time'] for item in learner_record if item['solver_time'] is not None])
        else:
            learner_solver_time = 0.0

        total_solver_time = verifier_solver_time + learner_solver_time
        return status, ce, num_queries, total_solver_time, results

    def sample_counterexamples(self, ce, num_samples, radius = 0.1, opt_iter=None):
        # given a set of counterexamples, sample nearby states and optimize to augment the counterexamples
        ce_x0 = ce['x0']
        new_ce_x0 = self.sample_ce_and_opt(ce_x0, num_samples, 'x0', radius, opt_iter)

        ce_xu = ce['xu']
        new_ce_xu = self.sample_ce_and_opt(ce_xu, num_samples, 'xu', radius, opt_iter)

        ce_x = ce['x']
        new_ce_x = self.sample_ce_and_opt(ce_x, num_samples, 'x', radius, opt_iter)

        if new_ce_x is None:
            new_ce_xn = None
        else:
            new_ce_xn = self.problem.system(torch.from_numpy(new_ce_x).to(self.device)).detach().cpu().numpy()

        new_ce = {'x0': new_ce_x0, 'xu': new_ce_xu, 'x': new_ce_x, 'xn': new_ce_xn}
        return new_ce
    
    def sample_ce_and_opt(self, ce, num_samples, type, radius=0.1, opt_iter=None):
        if ce is None:
            return ce

        num_ce = ce.shape[0]
        num_samples_batch = int(np.ceil(num_samples/num_ce))

        new_samples = None
        for i in range(num_ce):
            center = ce[i]
            local_poly = Polyhedron.from_bounds(center - radius, center + radius)
            new_ce = uniform_random_sample_from_Polyhedron(local_poly, num_samples_batch)
            new_ce = new_ce.astype('float32')

            if opt_iter is not None:
                new_ce = torch.from_numpy(new_ce).to(self.device)
                _, new_ce, _ = search_counterexamples(self.problem, type, samples=new_ce, num_iter=opt_iter)
                new_ce = new_ce.detach().cpu().numpy()

            if type == 'x0':
                idx_set = [self.problem.X0.set.contains(new_ce[i, :]) for i in range(new_ce.shape[0])]
            elif type == 'xu':
                idx_set = [self.problem.Xu.set.contains(new_ce[i, :]) for i in range(new_ce.shape[0])]
            elif type == 'x':
                idx_set = [self.problem.X.set.contains(new_ce[i, :]) for i in range(new_ce.shape[0])]
            else:
                raise ValueError('Type not recognized.')

            valid_samples = new_ce[idx_set]

            if new_samples is None:
                new_samples = valid_samples
            else:
                new_samples = np.concatenate((new_samples, valid_samples), axis=0)

        # concatenate the original counterexamples
        new_samples = np.concatenate((ce, new_samples), axis=0)
        return new_samples

    def train_and_verify(self, num_iter, method='verification-only', dataset=None, batch_size=50, save_model_path=None, training_opt=None,
                        ce_sampling_opt=None, accpm_opt=None, timeout=None):
        if dataset is None:
            dataset = self.sample_set 
            if self.sample_set is None:
                raise ValueError('Training data set is None.')

        if training_opt is None:
            training_opt = Training_Options()

        l1_lambda = training_opt.l1_lambda
        num_epochs = training_opt.num_epochs
        early_stop_tol = training_opt.early_stop_tol
        update_A_freq = training_opt.update_A_freq

        if ce_sampling_opt is None:
            ce_sampling_opt = CE_Sampling_Options()

        num_ce_samples = ce_sampling_opt.num_ce_samples
        radius = ce_sampling_opt.radius
        opt_iter = ce_sampling_opt.opt_iter

        barrier_status = 'unknown'
        num_verifier_queries = 0
        ce_record = []
        train_time = []
        verification_time = []

        training_start_time = time.time()
        if timeout is None:
            time_out_status = None
        else:
            time_out_status = False

        verification_sol_record = []
        for i in tqdm(range(num_iter),desc='training_iter'):
            if i% update_A_freq == 0:
                update_A = True
            else:
                update_A = False

            start_time = time.time()
            self.train_barrier_fcn_from_samples(save_model_path, l1_lambda=l1_lambda, num_epochs=num_epochs,
                                           early_stop_tol=early_stop_tol, dataset=dataset, update_A=update_A, batch_size=batch_size)

            train_time.append(time.time() - start_time)

            if method == 'verification-only':
                status, ce, solver_time, sol_record = self.verify_barrier_fcn()
                num_verifier_queries += 1
                ce_record.append(ce)
                verification_time.append(solver_time)
                verification_sol_record.append(sol_record)
            elif method == 'fine-tuning':
                status, ce, num_queries, solver_time, accpm_results = self.update_barrier_fcn_by_ACCPM(accpm_opt)
                num_verifier_queries += num_queries
                ce_record.append(ce)
                verification_time.append(solver_time)
                verification_sol_record.append(accpm_results)
            else:
                raise NotImplementedError

            if status == 'feasible':
                barrier_status = 'feasible'
                # save the updated barrier function
                if save_model_path is not None:
                    torch.save(self.barrier_fcn.state_dict(), save_model_path)

                verification_method = arguments.Config['alg_options']['verification_method']
                results = {'status': barrier_status, 'ce_record': ce_record, 'dataset': dataset,
                           'num_queries': num_verifier_queries, 'timeout_status': time_out_status,
                           'method': method, 'verification_method': verification_method, 'verification_sol_record': verification_sol_record,
                           'train_time': train_time, 'verification_time': verification_time}

                return barrier_status, num_verifier_queries, results

            new_ce = self.sample_counterexamples(ce, num_ce_samples, radius=radius, opt_iter=opt_iter)
            dataset = sample_set_union(dataset, new_ce)

            if timeout is not None:
                if timeout < sum(train_time) + sum(verification_time):
                    time_out_status = True
                    print(f'Barrier function training timeout! ({timeout} seconds)')
                    break

            if i < num_iter -1:
                # apply shrink and perturb
                scaling_factor = arguments.Config['alg_options']['barrier_fcn']['train_options']['scaling_factor']
                noise_weight = arguments.Config['alg_options']['barrier_fcn']['train_options']['noise_weight'] 
                shrink_and_perturb(self.barrier_fcn.net, scaling_factor=scaling_factor, noise_weight=noise_weight)
        
        verification_method = arguments.Config['alg_options']['verification_method']
        results = {'status': barrier_status, 'ce_record': ce_record, 'dataset': dataset, 
                   'num_queries': num_verifier_queries, 'timeout_status': time_out_status,
                   'method': method, 'verification_method': verification_method, 'verification_sol_record': verification_sol_record,
                   'train_time':train_time, 'verification_time': verification_time}
        return barrier_status, num_verifier_queries, results

def sample_set_union(set_1, set_2):
    sample_set = {'x': None, 'xn': None, 'x0': None, 'xu': None}

    if set_1['x0'] is None:
        x0_set = set_2['x0']
    elif set_2['x0'] is None:
        x0_set = set_1['x0']
    else:
        x0_set = np.concatenate((set_1['x0'], set_2['x0']), axis=0)
    sample_set['x0'] = x0_set

    if set_1['xu'] is None:
        xu_set = set_2['xu']
    elif set_2['xu'] is None:
        xu_set = set_1['xu']
    else:
        xu_set = np.concatenate((set_1['xu'], set_2['xu']), axis=0)
    sample_set['xu'] = xu_set

    if set_1['x'] is None:
        x_set = set_2['x']
        xn_set = set_2['xn']
    elif set_2['x'] is None:
        x_set = set_1['x']
        xn_set = set_1['xn']
    else:
        x_set = np.concatenate((set_1['x'], set_2['x']), axis=0)
        xn_set = np.concatenate((set_1['xn'], set_2['xn']), axis=0)

    sample_set['x'] = x_set
    sample_set['xn'] = xn_set
    return sample_set

def shrink_and_perturb(net, scaling_factor = 0.2, noise_weight=1.0):
    device = next(net.parameters()).device
    linear_layer_width, negative_slope, bias = extract_relu_structure(net)
    new_net = setup_relu(linear_layer_width, negative_slope=negative_slope, bias=bias)
    new_net = new_net.to(device)

    params_net = list(net.parameters())
    params_new_net = list(new_net.parameters())
    for i in range(len(params_net)):
        params_net[i].data = scaling_factor*params_net[i].data + noise_weight*params_new_net[i].data
