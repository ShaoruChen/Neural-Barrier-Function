import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling import generate_training_data, sample_batch, generate_traj_samples, plot_multiple_traj
from utils.sampling import uniform_random_sample_from_Polyhedron, scale_polyhedron

import pickle
def train_autonomous_nn_dynamics(nn_model, ref_dynamics, domain, dataset_path, save_path, load_data_path=None, l1_lambda=None):
    # generate training dataset for the NN dynamics
    device = next(nn_model.parameters()).device

    if load_data_path is None:
        x_samples, xn_samples = generate_training_data(ref_dynamics, domain, 50000)
        pickle.dump({'x': x_samples, 'xn': xn_samples}, open(dataset_path, 'wb'))

        dataset = pickle.load(open(dataset_path, 'rb'))
    else:
        dataset = pickle.load(open(load_data_path, 'rb'))

    x_samples, xn_samples = dataset['x'], dataset['xn']
    nn_model.train()
    opt = torch.optim.Adam(nn_model.parameters())

    for j in range(100000):
        x_batch, xn_batch = sample_batch(x_samples, xn_samples, batchsize=500)
        x_batch, xn_batch = torch.from_numpy(x_batch.astype('float32')).to(device), torch.from_numpy(
            xn_batch.astype('float32')).to(device)

        opt.zero_grad()

        loss = nn_model.loss(x_batch, xn_batch, l1_lambda=l1_lambda)
        loss.backward()

        opt.step()

        if j % 100 == 99:
            print(f'iter {j} loss: {loss.item():.8f}')

    torch.save(nn_model.state_dict(), save_path)

def train_nn_barrier_function(problem, train_data_path, save_model_path, load_data_path=None, l1_lambda=None, iter_num=50000):
    # generate training samples for learning the Lyapunov function
    system = problem.system
    barrier_fcn = problem.barrier_fcn
    X, X0, Xu = problem.X, problem.X0, problem.Xu

    if load_data_path is None:
        x_samples, xn_samples = generate_training_data(system, X.set, 50000)
        x0_samples = X0.sample(10000)
        xu_samples = Xu.sample(10000)

        pickle.dump({'x': x_samples, 'xn': xn_samples, 'x0': x0_samples, 'xu':xu_samples}, open(train_data_path, 'wb'))

        dataset = pickle.load(open(train_data_path, 'rb'))
    else:
        dataset = pickle.load(open(load_data_path, 'rb'))

    x_samples, xn_samples = dataset['x'], dataset['xn']
    x0_samples, xu_samples = dataset['x0'], dataset['xu']

    barrier_fcn.train()
    opt = torch.optim.Adam(barrier_fcn.parameters())
    device = next(barrier_fcn.net.parameters()).device

    for j in range(iter_num):
        # TODO: does sampling in the loop make the training slow? How does it compare with using a dataloader?
        x_batch, xn_batch = sample_batch(x_samples, xn_samples, batchsize=500)
        x0_batch, xu_batch = sample_batch(x0_samples, xu_samples, batchsize=500)

        x_batch, xn_batch = torch.from_numpy(x_batch.astype('float32')).to(device), torch.from_numpy(
            xn_batch.astype('float32')).to(device)

        x0_batch, xu_batch = torch.from_numpy(x0_batch.astype('float32')).to(device), torch.from_numpy(
            xu_batch.astype('float32')).to(device)

        opt.zero_grad()

        loss = barrier_fcn.loss(x0_batch, xu_batch, x_batch, xn_batch, l1_lambda=l1_lambda)
        loss.backward()

        opt.step()

        # TODO: check if the PGD is done properly.
        barrier_fcn.A.data = barrier_fcn.A.clamp(min=0.0)

        if j % 100 == 99:
            print(f'iter {j} loss: {loss.item():.8f}')

        if j % 10000 == 9999:
            torch.save(barrier_fcn.state_dict(), save_model_path)

    torch.save(barrier_fcn.state_dict(), save_model_path)

