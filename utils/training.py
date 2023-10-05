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
from .sampling import pickle_file

import torch.nn as nn
#
# '''
# simulate closed-loop system with NN components
# NN structure and data are from pytorch
# '''
#
# def simulate_NN_system(nn_model, init_states, step = 0):
#     # init_states is a list of initial conditions in numpy
#
#     N = init_states.shape[0]
#     nx = init_states.shape[1]
#
#     traj_list = []
#     for i in range(N):
#         x = torch.from_numpy(init_states[i])
#         x = x.type(nn_model[0].weight.dtype)
#
#         traj = x.unsqueeze(0)
#         for j in range(step+1):
#             x_next = nn_model(x)
#             traj = torch.cat((traj, x_next.unsqueeze(0)))
#             x = x_next
#         traj_list.append(traj)
#     return traj_list
#
#
#
# def plot_multiple_traj_tensor_to_numpy(traj_list, **kwargs):
#     num_traj = len(traj_list)
#     for i in range(num_traj):
#         plot_state_space_trajectory(traj_list[i].detach().numpy(), **kwargs)
#
#
# def bounds_list_to_polyhedron_list(bounds_list):
#     N = len(bounds_list)
#     poly_list = []
#
#     if 'A' in bounds_list[0].keys():
#         for i in range(N):
#             X = Polyhedron(bounds_list[i]['A'], bounds_list[i]['b'])
#             poly_list.append(X)
#     elif 'lb' in bounds_list[0].keys():
#         for i in range(N):
#             X = Polyhedron.from_bounds(bounds_list[i]['lb'], bounds_list[i]['ub'])
#             poly_list.append(X)
#
#     return poly_list
#
# def plot_poly_list(poly_list, **kwargs):
#     N = len(poly_list)
#     for i in range(N):
#         poly_list[i].plot(**kwargs)


# train NN basis function for reach-avoid analysis



# custom data set
class SystemDataSet(Dataset):
    def __init__(self, x_samples, y_samples):
        x_samples = torch.from_numpy(x_samples)
        y_samples = torch.from_numpy(y_samples)
        nx = x_samples.size(-1)
        ny = y_samples.size(-1)
        self.nx = nx
        self.ny = ny

        # we sample trajectories
        x_samples, y_samples = x_samples.type(torch.float32), y_samples.type(torch.float32)
        x_samples = x_samples.unsqueeze(1)
        y_samples = y_samples.unsqueeze(1)

        self.x_samples = x_samples
        self.y_samples = y_samples

    def __len__(self):
        return len(self.x_samples)

    def __getitem__(self, index):
        target = self.y_samples[index]
        data_val = self.x_samples[index]
        return data_val, target


def criterion(pred_traj, label_traj):
    batch_size = pred_traj.size(0)
    step = pred_traj.size(1)
    label_step = label_traj.size(1)
    if step > label_step:
        warnings.warn('prediction step mismatch')

    slice_step = min(step, label_step)

    label_traj_slice = label_traj[:, :slice_step, :]
    pred_traj_slice = pred_traj[:, :slice_step, :]

    # label_traj_slice_norm = torch.unsqueeze(torch.linalg.norm(label_traj_slice, 2, dim = 2), 2)
    # label_traj_slice = label_traj_slice/label_traj_slice_norm
    # pred_traj_slice = pred_traj_slice/label_traj_slice_norm
    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(batch_size*step)

    # err = 0.0
    # for i in range(batch_size):
    #     err += torch.linalg.norm(label_traj_slice[i].reshape(-1) - pred_traj_slice[i].reshape(-1), np.inf)/(torch.linalg.norm(label_traj_slice[i].reshape(-1), np.inf) + 1e-4)
    #
    # err = err/batch_size
    #
    #
    err = torch.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)/(batch_size*slice_step)

    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(pred_traj_slice.reshape(-1).size(0))

    return err

def torch_train_nn(nn_model, dataloader, l1 = None, epochs = 30, step = 5, lr = 1e-4, decay_rate = 1.0, clr = None):

    if clr is None:
        optimizer = optim.Adam(nn_model.parameters(), lr= lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate, last_epoch=-1)
        lr_scheduler = lambda t: lr
        cycle = 1
        update_rate = 1
    else:
        lr_base = clr['lr_base']
        lr_max = clr['lr_max']
        step_size = clr['step_size']
        cycle = clr['cycle']
        update_rate = clr['update_rate']
        optimizer = optim.Adam(nn_model.parameters(), lr= lr_max)
        lr_scheduler = lambda t: np.interp([t], [0, step_size, cycle], [lr_base, lr_max, lr_base])[0]

    lr_test = {}
    cycle_loss = 0.0
    cycle_count = 0

    nn_model.train()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        lr = lr_scheduler((epoch//update_rate)%cycle) # change learning rate every two epochs
        optimizer.param_groups[0].update(lr=lr)

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            batch_size = inputs.size(0)
            x = inputs
            y = nn_model(x)
            traj = y
            for _ in range(step):
                x = y
                y = nn_model(x)
                traj = torch.cat((traj, y), 1)

            loss_1 = criterion(traj, labels)

            # add l1 regularization
            if l1 is not None:
                l1_regularization = 0.0
                for param in nn_model.parameters():
                    '''attention: what's the correct l1 regularization'''
                    l1_regularization += torch.linalg.norm(param.view(-1), 1)
                loss = loss_1 + l1*l1_regularization
            else:
                loss = loss_1

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

            cycle_loss += loss_1.item()

            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        if (epoch + 1) % update_rate == 0:
            lr_test[cycle_count] = cycle_loss / update_rate / len(dataloader)
            print('\n [%d, %.4f] cycle loss: %.6f' % (cycle_count, lr, lr_test[cycle_count]))

            cycle_count += 1
            cycle_loss = 0.0

        # scheduler.step()
    pickle_file(lr_test, 'lr_test_temp')

    print('finished training')
    save_torch_nn_model(nn_model, 'torch_nn_model_dict_temp')
    return nn_model


def train_nn_and_save(dataloader, nn_structure, num_epochs= 30, l1 = None, pred_step = 5, lr = 1e-4, decay_rate = 1.0, clr = None, path = 'torch_nn_model_temp'):
    nn_model = torch_train_nn(nn_structure, dataloader, l1 = l1, epochs = num_epochs, step = pred_step, lr = lr, decay_rate = decay_rate, clr = clr)
    save_torch_nn_model(nn_model, path)
    return nn_model


def load_torch_nn_model(nn_model, model_param_name):
    nn_model.load_state_dict(torch.load(model_param_name))
    return nn_model

def save_torch_nn_model(nn_model, path):
    torch.save(nn_model.state_dict(), path)


def get_nn_info(net):
    # get information from an MLP
    input_size = net[0].in_features
    dims = [input_size]
    for layer in net:
        if isinstance(layer, nn.Linear):
            dims.append(layer.out_features)

    # extract neural network structure
    act_layer_count = 0
    for layer in list(net):
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.Sigmoid):
            act_layer_count += 1

    # extract the weights of the NN 
    weights_list = []

    negative_slope=0.0
    for layer in net:
        if isinstance(layer, nn.Linear):
            if layer.bias is not None:
                weights_list.append([layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()])
            else:
                weights_list.append([layer.weight.detach().cpu().numpy(), None])
        elif isinstance(layer, nn.LeakyReLU):
            negative_slope=layer.negative_slope
    return dims, act_layer_count, weights_list, negative_slope


############## codes from neural-network-lyapunov #################

def train_approximator(dataset,
                       model,
                       output_fun,
                       batch_size,
                       num_epochs,
                       lr,
                       additional_variable=None,
                       output_fun_args=dict(),
                       verbose=True,
                       l1_lambda=None):
    """
    @param additional_variable A list of torch tensors (with
    requires_grad=True), such that we will optimize the model together with
    additional_variable.
    @param output_fun_args A dictionnary of additional arguments to pass to
    output_fun
    """
    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_set_size, test_set_size])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    variables = model.parameters() if additional_variable is None else\
        list(model.parameters()) + additional_variable
    optimizer = torch.optim.Adam(variables, lr=lr)
    # TODO: need to pass customized training loss
    loss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.
        for i, data in enumerate(train_loader, 0):
            input_samples, target = data
            optimizer.zero_grad()

            # SC: the output function transforms the NN outputs
            output_samples = output_fun(model, input_samples,
                                        **output_fun_args)
            batch_loss = loss(output_samples, target)

            if l1_lambda is not None:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                batch_loss += l1_lambda * l1_norm

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
        test_input_samples, test_target = test_set[:]
        test_output_samples = output_fun(model, test_input_samples,
                                         **output_fun_args)
        test_loss = loss(test_output_samples, test_target)

        if verbose:
            print(f"epoch {epoch} training loss " +
                  f"{running_loss/len(train_loader)}," +
                  f" test loss {test_loss}")


def uniform_sample_in_box(lo: torch.Tensor, hi: torch.Tensor,
                          num_samples) -> torch.Tensor:
    """
    Take uniform samples in the box lo <= x <= hi.
    @return samples A num_samples x x_dim tensor.
    """
    x_dim = lo.numel()
    assert (hi.shape == (x_dim, ))
    samples = torch.rand(num_samples, x_dim, dtype=torch.float32)
    samples = samples @ torch.diag(hi - lo)
    samples += torch.reshape(lo, (1, x_dim))
    return samples


def uniform_sample_on_box_boundary(lo: torch.Tensor, hi: torch.Tensor,
                                   num_samples) -> torch.Tensor:
    """
    Uniformly samples on the boundary of the box lo <= x <= hi
    """
    samples_in_box = uniform_sample_in_box(lo, hi, num_samples)
    x_dim = lo.numel()
    boundary_face_rand = torch.rand((num_samples, )) * 2 - 1
    for i in range(num_samples):
        boundary_face_index = int(
            (torch.abs(boundary_face_rand[i]) * x_dim).item())
        samples_in_box[i, boundary_face_index] = lo[
            boundary_face_index] if boundary_face_rand[i] < 0 else hi[
                boundary_face_index]
    return samples_in_box


def save_second_order_forward_model(forward_relu, q_equilibrium, u_equilibrium,
                                    dt, file_path):
    linear_layer_width, negative_slope, bias = extract_relu_structure(
        forward_relu)
    torch.save(
        {
            "linear_layer_width": linear_layer_width,
            "state_dict": forward_relu.state_dict(),
            "negative_slope": negative_slope,
            "bias": bias,
            "q_equilibrium": q_equilibrium,
            "u_equilibrium": u_equilibrium,
            "dt": dt
        }, file_path)

def extract_relu_structure(relu_network):
    """
    Get the linear_layer_width, negative_slope and bias flag.
    """
    linear_layer_width = []
    negative_slope = None
    bias = None
    for layer in relu_network:
        if isinstance(layer, torch.nn.Linear):
            if len(linear_layer_width) == 0:
                # first layer
                linear_layer_width.extend(
                    [layer.in_features, layer.out_features])
            else:
                linear_layer_width.append(layer.out_features)
            if layer.bias is not None:
                assert (bias is None or bias)
                bias = True
            else:
                assert (bias is None or not bias)
                bias = False
        elif isinstance(layer, torch.nn.ReLU):
            if negative_slope is None:
                negative_slope = 0.
            else:
                assert (negative_slope == 0.)
        elif isinstance(layer, torch.nn.LeakyReLU):
            if negative_slope is None:
                negative_slope = layer.negative_slope
            else:
                assert (negative_slope == layer.negative_slope)
        else:
            raise Exception("extract_relu_structure(): unknown layer.")
    return tuple(linear_layer_width), negative_slope, bias


def setup_relu(relu_layer_width: tuple,
               params=None,
               negative_slope: float = 0.01,
               bias: bool = True,
               dtype=torch.float32):
    """
    Setup a relu network.
    @param negative_slope The negative slope of the leaky relu units.
    @param bias whether the linear layer has bias or not.
    """
    assert (isinstance(relu_layer_width, tuple))
    if params is not None:
        assert (isinstance(params, torch.Tensor))

    def set_param(linear, param_count):
        linear.weight.data = params[param_count:param_count +
                                    linear.in_features *
                                    linear.out_features].clone().reshape(
                                        (linear.out_features,
                                         linear.in_features))
        param_count += linear.in_features * linear.out_features
        if bias:
            linear.bias.data = params[param_count:param_count +
                                      linear.out_features].clone()
            param_count += linear.out_features
        return param_count

    linear_layers = [None] * (len(relu_layer_width) - 1)
    param_count = 0
    for i in range(len(linear_layers)):
        next_layer_width = relu_layer_width[i + 1]
        linear_layers[i] = torch.nn.Linear(relu_layer_width[i],
                                           next_layer_width,
                                           bias=bias).type(dtype)
        if params is None:
            pass
        else:
            param_count = set_param(linear_layers[i], param_count)
    layers = [None] * (len(linear_layers) * 2 - 1)
    for i in range(len(linear_layers) - 1):
        layers[2 * i] = linear_layers[i]
        layers[2 * i + 1] = torch.nn.LeakyReLU(negative_slope)
    layers[-1] = linear_layers[-1]
    relu = torch.nn.Sequential(*layers)
    return relu




