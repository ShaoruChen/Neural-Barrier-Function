import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils.sampling import generate_training_data, uniform_random_sample_from_Polyhedron
from utils.training import train_approximator
import copy
import pickle
import os

class Dynamics_Trainer:
    def __init__(self, dyn_fcn, nn_model, domain, x_dim, u_dim, save_dir):
        self.dyn_fcn = dyn_fcn
        self.nn_model = nn_model
        # augmented domain in the (state, action) space
        self.domain = domain

        self.x_dim, self.u_dim = x_dim, u_dim

        self.sample_set = None
        self.save_dir = save_dir

        self.device = next(self.nn_model.parameters()).device

    def generate_training_samples(self, num_samples=10000):
        input_data, labels = generate_training_data(self.dyn_fcn, self.domain, num_samples)
        dataset = {'x': input_data, 'y': labels}
        self.sample_set = dataset
        sample_path = os.path.join(self.save_dir, 'dynamics_samples.p')
        torch.save(self.sample_set, sample_path)
        return dataset

    def load_sample_set(self, path):
        data = torch.load(path)
        self.sample_set = data

    def train_nn_dynamics(self, batch_size=50, num_epochs=500, lr=0.001, l1_lambda=None, save_path=None):
        samples = self.sample_set
        x_samples = torch.from_numpy(samples['x'].astype('float32')).to(self.device),
        y_samples = torch.from_numpy(samples['y'].astype('float32')).to(self.device)

        dyn_dataset = torch.utils.data.TensorDataset(x_samples, y_samples)
        nn_model = self.nn_model
        loss_fcn = torch.nn.MSELoss()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'nn_dyn_model.p')

        trained_model_state_dict = train_nn_approximator(dyn_dataset, nn_model, loss_fcn,
                                                         batch_size=batch_size, num_epochs=num_epochs, lr=lr, l1_lambda=l1_lambda, save_path=save_path)

        self.nn_model.load_state_dict(trained_model_state_dict)

        return self.nn_model


class Controller_Trainer:
    def __init__(self, dyn, controller, domain, x_dim, u_dim, u_lb, u_ub, x_target, save_dir):
        self.dynamics = dyn
        self.controller = controller
        self.x_dim, self.u_dim = x_dim, u_dim

        self.domain = domain
        self.save_dir = save_dir

        self.horizon = 5

        # set of sampled initial conditions
        self.sample_set = None

        self.u_lb, self.u_ub = u_lb, u_ub
        self.x_target = x_target

    def generate_training_samples(self, num_samples=10000):
        data = uniform_random_sample_from_Polyhedron(self.domain, num_samples)
        self.sample_set = data
        sample_path = os.path.join(self.save_dir, 'init_state_samples.p')
        torch.save(self.sample_set, sample_path)
        return data

    def load_sample_set(self, path):
        data = torch.load(path)
        self.sample_set = data

    def closed_loop_dynamics_forward(self, x0):
        # TODO: check if this works well; otherwise we can use additional ReLU layers
        u = self.controller(x0).clamp(min=self.u_lb, max=self.u_ub)
        aug_x = torch.cat((x0, u), dim=-1)
        return self.dynamics(aug_x)

    def cost_fcn(self, x0_samples, horizon):
        x_target = self.x_target
        x_traj = x0_samples
        x = x0_samples
        for i in range(horizon):
            xn = self.closed_loop_dynamics_forward(x)
            x_traj = torch.cat((x_traj, xn), dim=-1)

        target = x_target.repeat(x0_samples.size(0), horizon+1)
        loss = torch.nn.MSELoss()
        return loss(x_traj, target)

    def train_controller(self, num_epochs=500, l1_lambda=None, horizon = 5, save_path=None):

        samples = self.sample_set
        x_samples = torch.from_numpy(samples.astype('float32')).to(self.device),
        y_samples = torch.ones((x_samples.size(0),), dtype=x_samples.dtype).to(self.device)

        dataset = torch.utils.data.TensorDataset(x_samples, y_samples)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

        controller = self.controller
        controller.train()
        opt = torch.optim.Adam(controller.parameters())

        for epoch in range(num_epochs):
            running_loss = 0.0
            for x_batch, y_batch in dataloader:
                opt.zero_grad()

                loss = self.loss(x_batch, horizon)
                if l1_lambda is not None:
                    l1_norm = sum(p.abs().sum() for p in controller.parameters())
                    loss += l1_lambda * l1_norm

                loss.backward()
                opt.step()

                running_loss += loss.item()
            print(f"epoch {epoch} training loss " + f"{running_loss / len(dataloader)},")

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'controller_model.p')

        torch.save(controller.state_dict(), save_path)


def train_nn_approximator(dataset, nn_model, loss_fcn, batch_size=50, num_epochs=500, lr=0.001, l1_lambda=None, save_path=None):
    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_set_size, test_set_size])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    nn_model.train()
    opt = torch.optim.Adam(nn_model.parameters(), lr=lr)

    model_state_dict = None
    best_test_loss = np.inf
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            opt.zero_grad()

            loss = loss_fcn(nn_model(x_batch), y_batch)
            if l1_lambda is not None:
                l1_norm = sum(p.abs().sum() for p in nn_model.parameters())
                loss += l1_lambda * l1_norm

            loss.backward()
            opt.step()

            running_loss += loss.item()

        test_input, test_target = test_set[:]
        test_loss = loss_fcn(nn_model(test_input), test_target)

        if test_loss < best_test_loss:
            model_state_dict = copy.deepcopy(nn_model.state_dict())
            best_test_loss = test_loss

        print(f"epoch {epoch} training loss {running_loss / len(train_loader)}, test loss {test_loss}")

    if save_path is not None:
        for k, v in model_state_dict.items():
            model_state_dict[k] = v.detach().cpu()
        torch.save(model_state_dict, save_path)

    return model_state_dict

