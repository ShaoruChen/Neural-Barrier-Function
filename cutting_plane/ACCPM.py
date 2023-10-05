import torch
import cvxpy as cp
import numpy as np
import gurobipy as gp
import pickle
import warnings
from tqdm import tqdm
import torch.nn as nn
from verification.optimization import search_counterexamples
import time
from pympc.geometry.polyhedron import Polyhedron
from utils.sampling import uniform_random_sample_from_Polyhedron, scale_polyhedron
from verification.bab_verification import bab_barrier_fcn_verification

import complete_verifier.arguments as arguments
from verification.bab_verification import filter_adversarial_samples

class gurobi_options:
    def __init__(self, time_limit = 1e-9, MIPFocus = None):
        self.time_limit = time_limit
        self.MIPFocus = MIPFocus

class ACCPM_Options:
    def __init__(self, max_iter = 30, sample_ce_opt = None):
        self.max_iter = max_iter
        self.sample_ce_opt = sample_ce_opt

class Problem:
    # record the data of the problem
    def __init__(self, system, barrier_fcn, X, X0, Xu):
        self.system = system
        self.barrier_fcn = barrier_fcn
        self.X, self.X0, self.Xu = X, X0, Xu
        self.x_dim = system.x_dim

    @property
    def device(self):
        return self.barrier_fcn.device

class Learner:
    def __init__(self, problem, M=5.0):
        self.problem = problem
        self.x_dim = problem.x_dim
        # bounds on the linear layer weights and bias
        self.M = M

    @property
    def device(self):
        return self.problem.device
    
    def find_analytic_center(self, samples, options=None):
        try:
            sol = self.analytic_center(samples, options=options)
        except Exception as e:
            warnings.warn('Solver failed when solving the analytic center problem. ')
            sol = {'c': None, 'status': 'infeasible', 'solver_time': 0.0}

        return sol

    def analytic_center(self, samples, options = None):
        # samples: {'x0': N0 x n numpy array, 'xu': Nu x n numpy array, 'x': N x n numpy array, 'xn': N x n numpy array}
        B = self.problem.barrier_fcn
        m = B.n_out
        n = B.basis_fcn_dim
        A = B.A.detach().cpu().numpy()

        # TODO: the coefficients (C_var, b_var) are subject to constraints -M <= C_var <= M, -M <= b_var <= M
        M = self.M

        C_var = cp.Variable((m, n))
        b_var = cp.Variable(m)

        constr = []

        obj = 0.0
        # bounded range of c_var
        obj += sum([-cp.log(C_var[i][j] + 1.1*M)-cp.log(M - C_var[i][j]) for i in range(m) for j in range(n)])
        obj += sum([-cp.log(b_var[i] + 1.1*M)-cp.log(M - b_var[i]) for i in range(m)])

        # Lyapunov decrease condition
        x_samples, xn_samples = samples['x'], samples['xn']
        x0_samples, xu_samples = samples['x0'], samples['xu']

        if x_samples is not None:
            x_basis = B.evaluate_basis(torch.from_numpy(x_samples.astype('float32')).to(self.device))
            xn_basis = B.evaluate_basis(torch.from_numpy(xn_samples.astype('float32')).to(self.device))

            x_basis, xn_basis = x_basis.detach().cpu().numpy(), xn_basis.detach().cpu().numpy()

            # decrease constraints
            LHS_x = (x_basis@C_var.T)@A.T - (xn_basis@C_var.T)
            bias_term = b_var@A.T - b_var
            obj += sum([-cp.log(LHS_x[i][j] + bias_term[j]) for i in range(x_basis.shape[0]) for j in range(m)])

        if x0_samples is not None:
            x0_basis = B.evaluate_basis(torch.from_numpy(x0_samples.astype('float32')).to(self.device))
            x0_basis = x0_basis.detach().cpu().numpy()

            # initial set constraints
            LHS_0 = x0_basis @ C_var.T
            obj += sum([-cp.log(-(LHS_0[i][j]+b_var[j])) for i in range(x0_basis.shape[0]) for j in range(m)])

        if xu_samples is not None:
            xu_basis = B.evaluate_basis(torch.from_numpy(xu_samples.astype('float32')).to(self.device))
            xu_basis = xu_basis.detach().cpu().numpy()

            # unsafe set constraints
            LHS_u = xu_basis@C_var.T
            obj += sum([-cp.log(LHS_u[i, B.unsafe_index]+b_var[B.unsafe_index]) for i in range(xu_basis.shape[0])])

        # construct the problem
        prob = cp.Problem(cp.Minimize(obj), constr)

        # select a solver
        solver_name = arguments.Config['alg_options']['ACCPM']['cvxpy_solver']

        if solver_name == 'ECOS':
            cvxpy_solver = cp.ECOS
        elif solver_name == 'SCS':
            cvxpy_solver = cp.SCS
        elif solver_name == 'MOSEK':
            cvxpy_solver = cp.MOSEK
        else:
            raise ValueError(f'Solver {solver_name} is not supported.')

        prob.solve(solver = cvxpy_solver, verbose = True)

        C_sol = C_var.value
        b_sol = b_var.value

        status = prob.status
        solver_time = prob.solver_stats.solve_time
        sol = {'C': C_sol, 'b': b_sol, 'status': status, 'solver_time': solver_time}

        return sol
        
class Verifier:
    def __init__(self, problem, method='bab', gurobi_model=None, yaml_file_path=None):
        # method = 'bab' or 'mip'
        self.problem = problem
        self.method = method
        self.x_dim = problem.system.x_dim
        # path of the yaml file that stores the BaB parameters
        self.yaml_file_path = yaml_file_path

        if gurobi_model is not None:
            self.gurob_base_model = gurobi_model
            self.gurobi_base_model_init = None
            self.gurobi_base_model_unsafe = None
            self.gurobi_base_model_dec = None

            self.initialize_gurobi_base_model(gurobi_model)

    @property
    def device(self):
        return self.problem.device
    
    def initialize_gurobi_base_model(self, gurobi_model):
        gp_model_init = gurobi_model.copy()
        gp_model_init = self.problem.X0.add_gurobi_constr(gp_model_init, 'x_0', mark='init_set_constr')
        self.gurobi_base_model_init = gp_model_init

        gp_model_unsafe = gurobi_model.copy()
        gp_model_unsafe = self.problem.Xu.add_gurobi_constr(gp_model_unsafe, 'x_0', mark='unsafe_constr')
        self.gurobi_base_model_unsafe = gp_model_unsafe

        gp_model_dec = gurobi_model.copy()
        gp_model_dec = self.problem.system.add_gurobi_constr(gp_model_dec, 'x_0', 'x_1', mark='dynamics')
        gp_model_dec = self.problem.X.add_gurobi_constr(gp_model_dec, 'x_0', mark='state_space_constr')
        self.gurobi_base_model_dec = gp_model_dec

    def verify_candidate(self, C, b):
        method = self.method
        if method == 'bab':
            yaml_file_path = self.yaml_file_path
            assert yaml_file_path is not None
            status, ce_dict, sol_record = self.verify_candidate_bab(C, b, yaml_file_path)
        elif method == 'mip':
            status, ce_dict, sol_record = self.verify_candidate_mip(C, b)
        else:
            raise NotImplementedError

        return status, ce_dict, sol_record

    def verify_candidate_bab(self, C, b, yaml_file_path):

        B = self.problem.barrier_fcn
        device = B.device

        net = B.net
        x_dim = self.x_dim
        A_mat = B.A
        output_dim = B.n_out

        last_layer = nn.Linear(B.basis_fcn_dim, B.n_out).to(self.device)
        last_layer.weight.data = torch.from_numpy(C.astype('float32')).to(self.device)
        last_layer.bias.data = torch.from_numpy(b.astype('float32')).to(self.device)

        layers_to_verify = list(net)[:-1] + [last_layer]
        net_to_verify = nn.Sequential(*layers_to_verify).to(self.device)

        sol_record = []

        # unsafe region constraint
        # gradient descent-based attack
        start_time = time.time()
        ce, sol, output_val = search_counterexamples(self.problem, 'xu', net=net_to_verify, samples=None, num_iter=300, num_samples=100)
        runtime = time.time() - start_time

        if len(ce) > 0:
            ce_unsafe = ce.detach().cpu().numpy()
            # remove repetitive samples
            ce_unsafe = filter_adversarial_samples(ce_unsafe, 0.001)

            sol_record.append({'solver_time': runtime, 'method': 'pgd'})
            solver_status_unsafe = 'unsafe'
        else:
            solver_status_unsafe, ce_unsafe, sol_unsafe = bab_barrier_fcn_verification(self.problem, 'xu', yaml_file_path, net=net_to_verify)
            sol_record.append(sol_unsafe)

        # initial region constraint
        # gradient descent-based attack
        start_time = time.time()
        ce, sol, output_val = search_counterexamples(self.problem, 'x0', net=net_to_verify, samples=None, num_iter=300,
                                                     num_samples=100)
        runtime = time.time() - start_time

        if len(ce) > 0:
            ce_init = ce.detach().cpu().numpy()
            ce_init = filter_adversarial_samples(ce_init, 0.001)

            sol_record.append({'solver_time': runtime, 'method': 'pgd'})
            solver_status_init = 'unsafe'
        else:
            solver_status_init, ce_init, sol_init = bab_barrier_fcn_verification(self.problem, 'x0', yaml_file_path,
                                                                                net=net_to_verify)
            sol_record.append(sol_init)

        # decrease constraint
        # gradient descent-based attack
        start_time = time.time()
        ce, sol, output_val = search_counterexamples(self.problem, 'x', net=net_to_verify, samples=None, num_iter=300,
                                                     num_samples=100)
        runtime = time.time() - start_time

        if len(ce) > 0:
            ce_dec_x = ce.detach().cpu().numpy()
            ce_dec_x = filter_adversarial_samples(ce_dec_x, 0.001)

            sol_record.append({'solver_time': runtime, 'method': 'pgd'})
            solver_status_dec = 'unsafe'
        else:
            solver_status_dec, ce_dec_x, sol_dec = bab_barrier_fcn_verification(self.problem, 'x', yaml_file_path,
                                                                                net=net_to_verify)
            sol_record.append(sol_dec)

        if ce_dec_x is None:
            ce_dec_xn = None
        else:
            ce_dec_xn = self.problem.system(torch.from_numpy(ce_dec_x).to(device)).detach().cpu().numpy()

        ce_dict = {'xu': ce_unsafe, 'x0': ce_init, 'x': ce_dec_x, 'xn': ce_dec_xn}

        if solver_status_unsafe in ['safe', 'safe-incomplete'] and solver_status_init in ['safe', 'safe-incomplete'] \
                and solver_status_dec in ['safe', 'safe-incomplete']:
            status = 'feasible'
        else:
            status = 'infeasible'

        return status, ce_dict, sol_record

    def verify_candidate_mip(self, C, b):
        # (C, b) are the candidate weights and bias
        B = self.problem.barrier_fcn
        net = B.net
        x_dim = self.x_dim
        A_mat = B.A
        output_dim = B.n_out

        last_layer = nn.Linear(B.basis_fcn_dim, B.n_out).to(self.device)
        last_layer.weight.data = torch.from_numpy(C.astype('float32')).to(self.device)
        last_layer.bias.data = torch.from_numpy(b.astype('float32')).to(self.device)

        layers_to_verify = list(net)[:-1] + [last_layer]
        net_to_verify = nn.Sequential(*layers_to_verify).to(self.device)

        sol_record = []

        # construct objective functions
        # unsafe region constraint

        # gradient descent-based attack
        # Fixme: the attack should be conducted on the new barrier function
        start_time = time.time()
        ce, sol, output_val = search_counterexamples(self.problem, 'xu', net=net_to_verify, samples=None, num_iter=300, num_samples=100)
        runtime = time.time() - start_time

        if len(ce) > 0:
            ce_unsafe = ce.detach().cpu().numpy()
            ce_unsafe = filter_adversarial_samples(ce_unsafe, 0.001)

            sol_record.append({'solver_time': runtime, 'method': 'pgd'})
        else:
            gp_model_unsafe = self.gurobi_base_model_unsafe.copy()
            gp_model_unsafe = B.add_gurobi_constr(gp_model_unsafe, 'x_0', 'B_0', net=net_to_verify, domain=self.problem.Xu.set)

            # extract relevant variables
            B_0 = list()
            for i in range(B.n_out):
                B_0.append(gp_model_unsafe.getVarByName('B_0[' + str(i) + ']'))

            obj = B_0[B.unsafe_index]

            tol_unsafe = 0.0
            gp_model_unsafe.Params.BestObjStop = tol_unsafe + 1e-8
            gp_model_unsafe.Params.BestBdStop = tol_unsafe - 1e-8

            gp_model_unsafe.setObjective(obj, gp.GRB.MINIMIZE)
            gp_model_unsafe.optimize()

            status_unsafe, ce_unsafe, sol_unsafe = gurobi_results_processing(gp_model_unsafe, 'x_0', x_dim, mode='min', tol=tol_unsafe)
            sol_record.append(sol_unsafe)

        # initial set constraint
        # gradient descent-based attack
        start_time = time.time()
        ce, sol, output_val = search_counterexamples(self.problem, 'x0', net=net_to_verify, samples=None, num_iter=300, num_samples=100)
        runtime = time.time() - start_time

        if len(ce) > 0:
            ce_init = ce.detach().cpu().numpy()
            ce_init = filter_adversarial_samples(ce_init, 0.001)

            sol_record.append({'solver_time': runtime, 'method': 'pgd'})
        else:
            gp_model_init = self.gurobi_base_model_init.copy()
            gp_model_init = B.add_gurobi_constr(gp_model_init, 'x_0', 'B_0', net=net_to_verify, domain=self.problem.X0.set)

            tol_init = 0.0
            gp_model_init.Params.BestObjStop = tol_init - 1e-8
            gp_model_init.Params.BestBdStop = tol_init + 1e-8

            # extract relevant variables
            B_0 = list()
            for i in range(B.n_out):
                B_0.append(gp_model_init.getVarByName('B_0[' + str(i) + ']'))

            ce_init = None
            for i in range(output_dim):
                # TODO: should we separate samples for each individual barrier function constraint?
                obj = B_0[i]
                gp_model_init.setObjective(obj, gp.GRB.MAXIMIZE)
                gp_model_init.optimize()

                status_init, ce_0, sol_init = gurobi_results_processing(gp_model_init, 'x_0', x_dim, mode='max', tol=tol_init)
                sol_record.append(sol_init)

                if ce_0 is not None:
                    if ce_init is None:
                        ce_init = ce_0
                    else:
                        ce_init = np.concatenate((ce_init, ce_0), axis=0)

        # decrease constraint
        # gradient descent-based attack
        start_time = time.time()
        ce, sol, output_val = search_counterexamples(self.problem, 'x', net=net_to_verify, samples=None, num_iter=300, num_samples=500)
        runtime = time.time() - start_time
        if len(ce) > 0:
            ce_dec_x = ce.detach().cpu().numpy()
            ce_dec_x = filter_adversarial_samples(ce_dec_x, 0.001)

            ce_dec_xn = self.problem.system(torch.from_numpy(ce_dec_x).to(self.problem.device)).detach().cpu().numpy()
            sol_record.append({'solver_time': runtime, 'method': 'pgd'})
        else:
            gp_model_dec = self.gurobi_base_model_dec.copy()

            gp_model_dec = B.add_gurobi_constr(gp_model_dec, 'x_0', 'B_0', net=net_to_verify, \
                                               domain=self.problem.X.set, mark='B_0')

            # TODO: this is a temporary fix
            if self.problem.system.output_domain is not None:
                gp_model_dec = B.add_gurobi_constr(gp_model_dec, 'x_1', 'B_1', net=net_to_verify, \
                                                   domain=self.problem.system.output_domain, mark='B_1')
            else:
                output_domain = scale_polyhedron(self.problem.X.set, 1.5)
                gp_model_dec = B.add_gurobi_constr(gp_model_dec, 'x_1', 'B_1', net=net_to_verify, \
                                                   domain=output_domain, mark='B_1')

            tol_dec = 0.0
            gp_model_dec.Params.BestObjStop = tol_dec + 1e-8
            gp_model_dec.Params.BestBdStop = tol_dec - 1e-8
            # extract relevant variables
            B_0 = list()
            for i in range(B.n_out):
                B_0.append(gp_model_dec.getVarByName('B_0[' + str(i) + ']'))

            B_1 = list()
            for i in range(B.n_out):
                B_1.append(gp_model_dec.getVarByName('B_1[' + str(i) + ']'))

            ce_dec = None
            for i in range(output_dim):
                a_vec = A_mat[i,:].detach().cpu().numpy()
                obj = B_0@a_vec - B_1[i]
                gp_model_dec.setObjective(obj, gp.GRB.MINIMIZE)
                gp_model_dec.optimize()

                status_dec, ce_dec_i, sol_dec = gurobi_results_processing(gp_model_dec, 'x_0', x_dim, mode='min',tol=tol_dec)
                sol_record.append(sol_dec)

                if ce_dec_i is not None:
                    if ce_dec is None:
                        ce_dec = ce_dec_i
                    else:
                        ce_dec = np.concatenate((ce_dec, ce_dec_i), axis=0)

            ce_dec_x = ce_dec
            if ce_dec_x is None:
                ce_dec_xn = None
            else:
                ce_dec_xn = self.problem.system(torch.from_numpy(ce_dec_x).to(self.problem.device)).detach().cpu().numpy()

        ce_dict = {'xu': ce_unsafe, 'x0': ce_init, 'x': ce_dec_x, 'xn': ce_dec_xn}

        if (ce_unsafe is None) and (ce_init is None) and (ce_dec_x is None):
            # the candidate barrier function is verified
            status = 'feasible'
        else:
            status = 'infeasible'

        return status, ce_dict, sol_record

class ACCPM_Alg:
    def __init__(self, problem, method='bab', result_path = None, mip_solver_options = None, bab_yaml_path=None):
        # method = 'bab' if using branch-and-bound, 'mip' if using gurobi
        self.problem = problem

        self.method = method
        self.solver_options = mip_solver_options
        self.x_dim = problem.x_dim
        # path to save the results
        self.result_path = result_path

        self.bab_yaml_path = bab_yaml_path

        self.result = 'unknown'
        self.num_iter = None

        self.sample_set = {'x': None, 'xn': None, 'x0': None, 'xu': None}

        # save the counterexamples for the initial barrier function candidate
        self.init_ce = None

        if self.method == 'mip':
            self.gurobi_model = None
            self.init_gurobi_model()
        
        self.learner_sol_record = []
        self.verifier_sol_record = []

        self.barrier_coeff ={'C': None, 'b': None}
        self.barrier_coeff_record = []


        C_candidate = problem.barrier_fcn.net[-1].weight.data.detach().cpu().numpy()
        b_candidate = problem.barrier_fcn.net[-1].bias.data.detach().cpu().numpy()
        self.M = 2.0*np.maximum(np.abs(C_candidate).max(), np.abs(b_candidate).max())

        self.learner = Learner(problem, M=self.M)
        if self.method == 'bab':
            self.verifier = Verifier(problem, method=self.method, yaml_file_path=self.bab_yaml_path)
        elif self.method == 'mip':
            self.verifier = Verifier(problem, method=self.method, gurobi_model=self.gurobi_model, yaml_file_path= self.bab_yaml_path)
        else:
            raise ValueError(f'Method {self.method} is not supported.')

    def clear_sample_set(self):
        self.sample_set = None

    def reset(self):
        self.result = 'unknown'
        self.num_iter = None

        self.sample_set = {'x': None, 'xn': None, 'x0': None, 'xu': None}
        self.init_ce = None

        self.learner_sol_record = []
        self.verifier_sol_record = []

        self.barrier_coeff = {'C': None, 'b': None}
        self.barrier_coeff_record = []

    def init_gurobi_model(self):
        gurobi_model = gurobi_model_initialization(self.problem, options = self.solver_options)
        self.gurobi_model = gurobi_model
        return gurobi_model

    def verify_candidate_barrier(self, barrier_fcn=None):
        if barrier_fcn is None:
            B = self.problem.barrier_fcn
        else:
            B = barrier_fcn

        C_candidate, b_candidate = B.net[-1].weight.data.detach().cpu().numpy(), B.net[-1].bias.data.detach().cpu().numpy()

        # self.M = 2.0*np.maximum(np.abs(C_candidate).max(), np.abs(b_candidate).max())

        self.barrier_coeff = {'C': C_candidate, 'b': b_candidate}
        self.barrier_coeff_record.append({'C': C_candidate,'b': b_candidate})

        verifier = self.verifier
        verifier_status, ce_set, sol_record = verifier.verify_candidate(C_candidate, b_candidate)

        self.verifier_sol_record.append(sol_record)

        if verifier_status == 'feasible':
            return verifier_status, ce_set, sol_record

        # add counterexample to the sample set
        ce_unsafe, ce_init, ce_x, ce_xn = ce_set['xu'], ce_set['x0'], ce_set['x'], ce_set['xn']

        if ce_x is not None:
            if self.sample_set['x'] is None:
                self.sample_set['x'] = ce_x
                self.sample_set['xn'] = ce_xn
            else:
                self.sample_set['x'] = np.vstack((self.sample_set['x'], ce_x))
                self.sample_set['xn'] = np.vstack((self.sample_set['xn'], ce_xn))

        if ce_unsafe is not None:
            if self.sample_set['xu'] is None:
                self.sample_set['xu'] = ce_unsafe
            else:
                self.sample_set['xu'] = np.vstack((self.sample_set['xu'], ce_unsafe))

        if ce_init is not None:
            if self.sample_set['x0'] is None:
                self.sample_set['x0'] = ce_init
            else:
                self.sample_set['x0'] = np.vstack((self.sample_set['x0'], ce_init))

        return verifier_status, ce_set, sol_record

    def ACCPM_iter(self, sample_ce=True):
        problem = self.problem
        alg_status = 'unknown'

        learner = self.learner
        sample_set = self.sample_set
        learner_sol = learner.find_analytic_center(sample_set)
        self.learner_sol_record.append(learner_sol)

        if learner_sol['status'] in ['infeasible', 'unbounded']:
            alg_status = 'infeasible'
            return alg_status, None

        C_candidate, b_candidate = learner_sol['C'], learner_sol['b']
        self.barrier_coeff = {'C': C_candidate,'b': b_candidate}
        self.barrier_coeff_record.append({'C': C_candidate,'b': b_candidate})

        verifier = self.verifier
        verifier_status, ce_set, sol_record = verifier.verify_candidate(C_candidate, b_candidate)

        self.verifier_sol_record.append(sol_record)

        if verifier_status == 'feasible':
            alg_status = 'feasible'
            return alg_status, ce_set

        # add counterexample to the sample set
        if self.method == 'mip':
            ori_ce_set = ce_set
            if sample_ce is not None:
                num_ce_samples = sample_ce.num_ce_samples_accpm
                radius = sample_ce.radius
                opt_iter = sample_ce.opt_iter
                new_ce_set = sample_counterexamples(self.problem, ce_set, num_ce_samples, radius=radius, opt_iter=opt_iter)
                ce_unsafe, ce_init, ce_x, ce_xn = new_ce_set['xu'], new_ce_set['x0'], new_ce_set['x'], new_ce_set['xn']  
            else:
                ce_unsafe, ce_init, ce_x, ce_xn = ce_set['xu'], ce_set['x0'], ce_set['x'], ce_set['xn']

            accpm_num_ce_thresh = arguments.Config['alg_options']['ACCPM']['num_ce_thresh']

            # ce_unsafe = filter_adversarial_samples(ce_unsafe, 0.01)

            if ce_unsafe is not None:
                ce_unsafe = ce_unsafe[:accpm_num_ce_thresh]

            # ce_init = filter_adversarial_samples(ce_init, 0.01)
            if ce_init is not None:
                ce_init = ce_init[:accpm_num_ce_thresh]

            # ce_x = filter_adversarial_samples(ce_x, 0.01)
            if ce_x is None:
                ce_xn = None
            else:
                ce_x = ce_x[:accpm_num_ce_thresh]
                ce_xn = problem.system(torch.from_numpy(ce_x).to(self.problem.device)).detach().cpu().numpy()

            ce_set['x0'], ce_set['xu'], ce_set['x'], ce_set['xn'] = ce_init, ce_unsafe, ce_x, ce_xn

        elif self.method == 'bab':
            # remove close examples
            ce_unsafe, ce_init, ce_x, ce_xn = ce_set['xu'], ce_set['x0'], ce_set['x'], ce_set['xn']

            accpm_num_ce_thresh = arguments.Config['alg_options']['ACCPM']['num_ce_thresh']

            ce_unsafe = filter_adversarial_samples(ce_unsafe, 0.01)
            if ce_unsafe is not None:
                ce_unsafe = ce_unsafe[:accpm_num_ce_thresh]
                
            ce_init = filter_adversarial_samples(ce_init, 0.01)
            if ce_init is not None:
                ce_init = ce_init[:accpm_num_ce_thresh]
                
            ce_x = filter_adversarial_samples(ce_x, 0.01)
            if ce_x is None:
                ce_xn = None
            else:
                ce_x = ce_x[:accpm_num_ce_thresh]
                ce_xn = problem.system(torch.from_numpy(ce_x).to(self.problem.device)).detach().cpu().numpy()

            ce_set['x0'], ce_set['xu'], ce_set['x'], ce_set['xn'] = ce_init, ce_unsafe, ce_x, ce_xn
        else:
            ce_unsafe, ce_init, ce_x, ce_xn = ce_set['xu'], ce_set['x0'], ce_set['x'], ce_set['xn']

        if ce_x is not None:
            if self.sample_set['x'] is None:
                self.sample_set['x'] = ce_x
                self.sample_set['xn'] = ce_xn
            else:
                self.sample_set['x'] = np.vstack((self.sample_set['x'], ce_x))
                self.sample_set['xn'] = np.vstack((self.sample_set['xn'], ce_xn))

        if ce_unsafe is not None:
            if self.sample_set['xu'] is None:
                self.sample_set['xu'] = ce_unsafe
            else:
                self.sample_set['xu'] = np.vstack((self.sample_set['xu'], ce_unsafe))

        if ce_init is not None:
            if self.sample_set['x0'] is None:
                self.sample_set['x0'] = ce_init
            else:
                self.sample_set['x0'] = np.vstack((self.sample_set['x0'], ce_init))

        return alg_status, ce_set

    def run(self, alg_opt=None):
        # TODO: not sure if this has any side effects
        self.reset()

        # first verify the given candidate
        alg_status, ce_set, _ = self.verify_candidate_barrier()
        self.init_ce = ce_set

        if alg_status == 'feasible':
            self.result = alg_status
            results = {'status': self.result, 'problem': self.problem,
                       'learner_record': self.learner_sol_record,
                       'verifier_record': self.verifier_sol_record,
                       'barrier_coeff_record': self.barrier_coeff_record, 'num_iter': self.num_iter,
                       'output_coeff': self.barrier_coeff, 'sample_set': self.sample_set,
                       'last_ce_set': ce_set, 'init_ce_set': self.init_ce}
            self.save_data(self.result_path)

            return alg_status, 1, results

        # execute the ACCPM
        alg_status = 'unknown'
        if alg_opt is None:
            max_iter = 30
            sample_ce_opt = None
        else:
            max_iter = alg_opt.max_iter
            sample_ce_opt = alg_opt.sample_ce_opt

        count = 1
        while count < max_iter:
            iter_status, ce_set = self.ACCPM_iter(sample_ce=sample_ce_opt)
            if iter_status != 'infeasible':
                count += 1

            self.num_iter = count

            if iter_status in ['infeasible', 'feasible', 'verifier_failure']:
                alg_status = iter_status
                self.result = alg_status
                self.save_data(self.result_path)

                results = {'status': self.result, 'problem': self.problem,
                                'learner_record': self.learner_sol_record,
                                'verifier_record': self.verifier_sol_record,
                                'barrier_coeff_record': self.barrier_coeff_record, 'num_iter': self.num_iter,
                                'output_coeff': self.barrier_coeff, 'sample_set': self.sample_set,
                            'last_ce_set': ce_set, 'init_ce_set': self.init_ce}

                return alg_status, count, results

        self.result = alg_status

        results = {'status': self.result, 'problem': self.problem, 'learner_record': self.learner_sol_record,
                        'verifier_record': self.verifier_sol_record, 'barrier_coeff_record': self.barrier_coeff_record,
                        'num_iter': self.num_iter,
                        'output_coeff': self.barrier_coeff, 'sample_set': self.sample_set,
                    'last_ce_set': ce_set, 'init_ce_set': self.init_ce}

        self.save_data(self.result_path)

        return alg_status, count, results

    def save_data(self, path):
        if path is not None:
            data_to_save = {'status': self.result, 'problem': self.problem, 'learner_record':self.learner_sol_record,
                            'verifier_record': self.verifier_sol_record, 'barrier_coeff_record': self.barrier_coeff_record, 'num_iter': self.num_iter,
                            'output_coeff': self.barrier_coeff, 'sample_set': self.sample_set,
                            'init_ce_set': self.init_ce}
            pickle.dump(data_to_save, open(path, 'wb'))

def gurobi_model_construction(name = 'barrier', options = None):
    gurobi_model = gp.Model(name)
    # gurobi_model.Params.FeasibilityTol = 1e-6
    # gurobi_model.Params.IntFeasTol = 1e-5
    # gurobi_model.Params.OptimalityTol = 1e-6

    # gurobi_model.Params.DualReductions = 0

    gurobi_model.Params.NonConvex = 2

    if options is not None:
        if options.time_limit > 1e-3:
            gurobi_model.setParam("TimeLimit", options.time_limit)

        if options.MIPFocus is not None:
            # TODO: figure out this option
            gurobi_model.Params.MIPFocus = options.MIPFocus

    gurobi_model.update()
    return gurobi_model

def gurobi_model_initialization(problem, options = None):
    system = problem.system
    B = problem.barrier_fcn

    gurobi_model = gurobi_model_construction(name='base_model', options=options)
    nx = system.x_dim
    B_output_dim = B.n_out

    var_dict = {'x_0': nx, 'x_1': nx, 'B_0': B_output_dim, 'B_1': B_output_dim}
    gurobi_model = gurobi_model_addVars(gurobi_model, var_dict)

    gurobi_model.update()

    return gurobi_model

def gurobi_model_addVars(gurobi_model, var_dict):
    for name, dim in var_dict.items():
        gurobi_model.addVars(dim, lb = -gp.GRB.INFINITY, name = name)
    gurobi_model.update()
    return gurobi_model

def gurobi_results_processing(gurobi_model, sol_var, dim, mode='min', tol=0.0):
    solver_time = gurobi_model.Runtime
    gurobi_status = gurobi_model.Status

    if gurobi_status not in [2, 15]:
        sol = {'obj': None, 'status': gurobi_status, 'sol': None, 'solver_time': solver_time}
        status = 'infeasible'
        ce = None
        return status, ce, sol

    obj_value = gurobi_model.objVal

    if mode == 'min' and obj_value > tol:
        ce = None
    elif mode == 'max' and obj_value < tol:
        ce = None
    else:
        x = [gurobi_model.getVarByName(sol_var + '[' + str(i) + ']') for i in range(dim)]
        x_sol = np.array([x[i].X for i in range(dim)]).reshape(1, -1)
        x_sol = x_sol.astype('float32')

        ce = x_sol

    sol = {'obj': obj_value, 'status': gurobi_status, 'sol': ce, 'solver_time': solver_time}
    status = 'feasible'

    # if gurobi_status == 15:
    #     ce = x_sol
    # else:
    #     if mode == 'min' and obj_value > tol:
    #         ce = None
    #     elif mode == 'max' and obj_value < tol:
    #         ce = None
    #     else:
    #         ce = x_sol

    return status, ce, sol

def sample_counterexamples(problem, ce, num_samples, radius = 0.1, opt_iter=None):
    # given a set of counterexamples, sample nearby states and optimize to augment the counterexamples
    ce_x0 = ce['x0']
    new_ce_x0 = sample_ce_and_opt(problem, ce_x0, num_samples, 'x0', radius, opt_iter)

    ce_xu = ce['xu']
    new_ce_xu = sample_ce_and_opt(problem, ce_xu, num_samples, 'xu', radius, opt_iter)

    ce_x = ce['x']
    new_ce_x = sample_ce_and_opt(problem, ce_x, num_samples, 'x', radius, opt_iter)

    device = problem.device

    if new_ce_x is None:
        new_ce_xn = None
    else:
        new_ce_xn = problem.system(torch.from_numpy(new_ce_x).to(device)).detach().cpu().numpy()

    new_ce = {'x0': new_ce_x0, 'xu': new_ce_xu, 'x': new_ce_x, 'xn': new_ce_xn}
    return new_ce

def sample_ce_and_opt(problem, ce, num_samples, type, radius=0.1, opt_iter=None):
    if ce is None:
        return ce

    device = problem.device

    num_ce = ce.shape[0]
    num_samples_batch = int(np.ceil(num_samples/num_ce))

    new_samples = None
    for i in range(num_ce):
        center = ce[i]
        local_poly = Polyhedron.from_bounds(center - radius, center + radius)
        new_ce = uniform_random_sample_from_Polyhedron(local_poly, num_samples_batch)
        new_ce = new_ce.astype('float32')

        if opt_iter is not None:
            new_ce = torch.from_numpy(new_ce).to(device)
            _, new_ce, _ = search_counterexamples(problem, type, samples=new_ce, num_iter=opt_iter)
            new_ce = new_ce.detach().cpu().numpy()

        # make sure the original counterexample is included as the first entry
        # new_ce = np.vstack((center, new_ce))

        if type == 'x0':
            idx_set = [problem.X0.set.contains(new_ce[i, :]) for i in range(new_ce.shape[0])]
        elif type == 'xu':
            idx_set = [problem.Xu.set.contains(new_ce[i, :]) for i in range(new_ce.shape[0])]
        elif type == 'x':
            idx_set = [problem.X.set.contains(new_ce[i, :]) for i in range(new_ce.shape[0])]
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
