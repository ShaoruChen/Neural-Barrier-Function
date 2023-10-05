import numpy as np
from pympc.geometry.polyhedron import Polyhedron
from pympc.plot import plot_state_space_trajectory

from pympc.optimization.programs import linear_program

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings
import matplotlib.pyplot as plt
import pickle
import random

# dyn_fcn has inputs and outputs as torch tensors
# griding or sampling in the input space gives numpy arrays

def generate_training_data(dyn_fcn, X, N, random = True, load_file_name = None):
    # sample one-step transitions of the states
    # if filename is not None, load the existing file
    if load_file_name is not None:
        data = load_pickle_file(load_file_name)
        input_samples = data['train_data']
        labels = data['label_data']
    else:
        if random:
            input_samples = uniform_random_sample_from_Polyhedron(X, N)
        else:
            input_samples = grid_sample_from_Polyhedron(X, N)
        labels = sample_vector_field(dyn_fcn, input_samples)
    return input_samples, labels

def sample_vector_field(dyn_fcn, samples):
    # num_samples, nx = samples.shape
    if hasattr(dyn_fcn, 'device'):
        device = dyn_fcn.device
        samples_tensor = torch.from_numpy(samples).to(torch.float).to(device)
    else:
        samples_tensor = torch.from_numpy(samples).to(torch.float)

    labels = dyn_fcn(samples_tensor)

    return labels.detach().cpu().numpy()

def generate_traj_samples(dyn_fcn, init_states, step = 1):
    N = init_states.shape[0]
    init_states_tensor = torch.from_numpy(init_states).to(torch.float)
    traj_list = [init_states]
    x = init_states_tensor
    for i in range(step):
        y = dyn_fcn(x)
        traj_list.append(y.detach().numpy())
        x = y

    return traj_list

def generate_training_data_traj(dyn_fcn, X, N, random = True, step = 1):
    if random:
        init_states = uniform_random_sample_from_Polyhedron(X, N)
    else:
        init_states = grid_sample_from_Polyhedron(X, N)
    traj_list = generate_traj_samples(dyn_fcn, init_states, step)
    return traj_list

def find_bounding_box(X):
    # find the smallest box that contains Polyhedron X
    A = X.A
    b = X.b

    nx = A.shape[1]

    lb_sol = [linear_program(np.eye(nx)[i], A, b) for i in range(nx)]
    lb_val = [lb_sol[i]['min'] for i in range(nx)]

    ub_sol = [linear_program(-np.eye(nx)[i], A, b) for i in range(nx)]
    ub_val = [-ub_sol[i]['min'] for i in range(nx)]

    return lb_val, ub_val

def grid_sample_from_Polyhedron(X, N_dim, epsilon=None, residual_dim=None):
    # grid sample from the Polyhedron X with N_dim points on each dimension with uniform space
    # if residual_dim (list) is not None, we only grid sample on the dimensions in residual_dim
    nx = X.A.shape[1]

    if residual_dim is not None:
        X = X.project_to(residual_dim)

    lb, ub = find_bounding_box(X)
    box_grid_samples = grid_sample_from_box(lb, ub, N_dim, epsilon)
    idx_set = [X.contains(box_grid_samples[i, :]) for i in range(box_grid_samples.shape[0])]
    valid_samples = box_grid_samples[idx_set]

    if residual_dim is not None:
        aux_samples = np.zeros((valid_samples.shape[0], 1))
        for i in range(nx):
            if i in residual_dim:
                aux_samples = np.hstack((aux_samples, valid_samples[:, i].reshape(-1, 1)))
            else:
                aux_samples = np.hstack((aux_samples, np.zeros((valid_samples.shape[0], 1))))

        aux_samples = aux_samples[:, 1:]
        return aux_samples

    return valid_samples

def grid_sample_from_box(lb, ub, Ndim, epsilon=None):
    # generate uniform grid samples from a box {lb <= x <= ub} with Ndim samples on each dimension
    nx = len(lb)
    assert nx == len(ub)

    if epsilon is not None:
        lb = [lb[i] + epsilon for i in range(nx)]
        ub = [ub[i] - epsilon for i in range(nx)]

    grid_samples = grid_sample(lb, ub, Ndim, nx)
    return grid_samples


def grid_sample(lb, ub, Ndim, idx):
    # generate samples using recursion
    nx = len(lb)
    cur_idx = nx - idx
    lb_val = lb[cur_idx]
    ub_val = ub[cur_idx]

    if idx == 1:
        cur_samples = np.linspace(lb_val, ub_val, Ndim)
        return cur_samples.reshape(-1, 1)

    samples = grid_sample(lb, ub, Ndim, idx - 1)
    n_samples = samples.shape[0]
    extended_samples = np.tile(samples, (Ndim, 1))

    cur_samples = np.linspace(lb_val, ub_val, Ndim)
    new_column = np.kron(cur_samples.reshape(-1, 1), np.ones((n_samples, 1)))

    new_samples = np.hstack((new_column, extended_samples))
    return new_samples

def uniform_random_sample_from_Polyhedron(X, N):
    # uniformly grid sample from the Polyhedron X with N_dim grid points on each dimension
    nx = X.A.shape[1]
    lb, ub = find_bounding_box(X)
    box = [[lb[i], ub[i]] for i in range(len(lb))]
    box_grid_samples = uniform_random_sample_from_box(box, N)
    idx_set = [X.contains(box_grid_samples[i, :]) for i in range(box_grid_samples.shape[0])]
    valid_samples = box_grid_samples[idx_set]
    return valid_samples

def uniform_random_sample_from_box(bounds_list, N):
    # box_list = [[lb_1, ub_1], [lb_2, ub_2], ..., [lb_n, ub_n]] where lb_i and ub_i denotes the box range in the i-th dim.
    # sample a total of N points randomly from the box described by bounds_list
    box_list = [[item[0], item[1] - item[0]] for item in bounds_list]
    nx = len(box_list)
    rand_matrix = np.random.rand(N, nx)
    samples = np.vstack([rand_matrix[:, i] * box_list[i][1] + box_list[i][0] for i in range(nx)])
    samples = samples.T
    return samples

# save and load data
def load_pickle_file(file_name):
    with open(file_name, 'rb') as config_dictionary_file:
        data = pickle.load(config_dictionary_file)
    return data

def pickle_file(data, file_name):
    with open(file_name, 'wb') as config_dictionary_file:
          pickle.dump(data, config_dictionary_file)

def plot_multiple_traj(x_traj_list, **kwargs):
    num_traj = x_traj_list[0].shape[0]
    horizon = len(x_traj_list)
    nx = x_traj_list[0].shape[1]

    traj = np.array(x_traj_list)
    for i in range(num_traj):
        plot_state_space_trajectory(traj[:,i,:], **kwargs)

def sample_batch(x, y = None, batchsize=10):
    # sample a batch from x, y
    # x, y are np.arrays
    N = x.shape[0]
    if batchsize > N:
        batchsize = N

    ind = np.random.choice(N, batchsize)
    if y is None:
        return x[ind]
    else:
        return x[ind], y[ind]


def scale_polyhedron(X, gamma):
    A, b = X.A, X.b
    Y = Polyhedron(A, gamma*b)
    return Y

def set_seed(seed=0, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

