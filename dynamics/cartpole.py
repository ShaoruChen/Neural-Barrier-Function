
import numpy as np
import torch
from torch.autograd.functional import jacobian
from pympc.geometry.polyhedron import Polyhedron
import torch.nn as nn
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CartPole:
    def __init__(self, mp, mc, length, dt = 0.1):
        self.mp = mp
        self.mc = mc
        self.g = 9.81
        self.len = length

        nx, nu = 4, 1
        self.nx, self.nu = nx, nu

        # state = [p, pdot, theta, thetadot]
        self.state = np.zeros(nx)
        self.u = np.zeros(1)
        self.dt = dt

        self.Ad = None
        self.Bd = None

    def vector_field(self, state_u):

        p, pdot, theta, thetadot, u  = state_u[0], state_u[1], state_u[2], state_u[3], state_u[4]

        vec = np.zeros(self.nx)
        vec[0] = pdot
        vec[2] = thetadot

        denom = self.len*(4/3 - self.mp*np.cos(theta)**2/(self.mc + self.mp))
        thetaddot = (self.g*np.sin(theta) + np.cos(theta)*(-u-self.mp*self.len*thetadot**2*np.sin(theta))/(self.mp + self.mc))/denom

        vec[1] = (u + self.mp*self.len*(thetadot**2*np.sin(theta) - thetaddot*np.cos(theta)))/(self.mp + self.mc)
        vec[3] = thetaddot

        return vec

    def Jacobian_torch(self, state, u):
        p, pdot, theta, thetadot = state[0], state[1], state[2], state[3]

        vec = torch.zeros(self.nx)
        vec[0] = pdot
        vec[2] = thetadot

        denom = self.len * (4 / 3 - self.mp * torch.cos(theta) ** 2 / (self.mc + self.mp))
        thetaddot = (self.g * torch.sin(theta) + torch.cos(theta) * (-u - self.mp * self.len * thetadot ** 2 * torch.sin(theta)) / (
                    self.mp + self.mc)) / denom

        vec[1] = (u + self.mp * self.len * (thetadot ** 2 * torch.sin(theta) - thetaddot * torch.cos(theta))) / (
                    self.mp + self.mc)
        vec[3] = thetaddot
        return vec

    def discrete_time_linear_dyn(self):
        state = torch.tensor([0.0, 0.0, 0.0, 0.0])
        u = torch.tensor([0.0])
        Ac, Bc = jacobian(self.Jacobian_torch, (state, u))

        dt = self.dt
        Ac, Bc = Ac.numpy(), Bc.numpy()
        Id = np.eye(self.nx)
        Ad = Id + Ac*dt
        Bd = Bc*dt

        self.Ad, self.Bd = Ad, Bd
        return Ad, Bd

    def discrete_dyn(self, state_u):
        # use Euler discretization with dt
        x = state_u[:self.nx]
        u = state_u[self.nx:]
        x_next = x + self.vector_field(state_u)*self.dt
        return x_next

    def discrete_residual_dyn(self, state_u):
        x_next = self.discrete_dyn(state_u)
        x = state_u[:self.nx]
        u = state_u[self.nx:]
        if self.Ad is None or self.Bd is None:
            self.discrete_time_linear_dyn()

        Ad, Bd = self.Ad, self.Bd
        res = x_next - x@Ad.T - u@Bd.T
        return res

class CartPoleNNSystem(nn.Module):
    def __init__(self, A, B, res_dyn, controller):
        super(CartPoleNNSystem, self).__init__()
        self.A, self.B = A, B
        self.res_dyn = res_dyn
        self.controller = controller
        self.nx = 4

    def forward(self, x):
        if x.dim() ==1:
            x = x.unsqueeze(0)

        u = self.controller(x)
        res = self.res_dyn(torch.cat((x, u), dim = -1))
        x_next = x@self.A.T + u@self.B.T + res
        x_next = x_next.squeeze(0)
        return x_next

def cartpole_res_dyn():
    # neural network dynamics with randomly generated weights
    model = nn.Sequential(
        nn.Linear(5, 50),
        nn.ReLU(),
        nn.Linear(50, 4)
    )
    return model

def nn_controller():
    model = nn.Sequential(
        nn.Linear(4, 100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    return model

mp, mc, length = 0.1, 0.25, 0.2
dt = 0.05

if __name__ == "__main__":
    nx = 4
    cartpole = CartPole(mp, mc, length, dt = dt)
    cartpole.discrete_time_linear_dyn()
    Ad, Bd = cartpole.Ad, cartpole.Bd

    is_train = True
    nn_system_file_name = 'model_data/cartpole_res_0.pt'
    if is_train:
        '''train a neural network to approximate the given dynamics'''
        # domain of the system
        x_u_min = np.array([-5.0, -5.0, -np.pi, -2*np.pi, -4.0])
        x_u_max = np.array([5.0, 5.0, np.pi, 2*np.pi, 4.0])
        domain = Polyhedron.from_bounds(x_u_min, x_u_max)
        nn_model = cartpole_res_dyn()
        sampling_param = 10000
        res_nn = nn_dyn_approx(cartpole.res_dyn_labels, nn_model, domain, sampling_param,
                                 batch_size = 30, num_epochs = 500, random_sampling = True, save_model_path = nn_system_file_name)

    res_model = cartpole_res_dyn()
    res_model.load_state_dict(torch.load(nn_system_file_name))

    # save nn model in mat
    # nn_model_weights = [res_model[0].weight.detach().numpy(), res_model[0].bias.detach().numpy(),
    #                     res_model[2].weight.detach().numpy(), res_model[2].bias.detach().numpy() ]
    #
    # dat = {}
    # dat['nn_model_weights'] = nn_model_weights
    # cartpole.discrete_time_linear_dyn()
    # Ad, Bd = cartpole.Ad, cartpole.Bd
    # dat['Ad'] = Ad
    # dat['Bd'] = Bd
    # dat['dt'] = cartpole.dt
    # savemat('cartpole_res_dyn.mat', dat)

    # train a nn controller from imitation learning

    controller_train = True
    controller_path = 'model_data/cartpole_nn_controller_0.pt'

    if controller_train:
        controller = nn_controller()
        X_train = loadmat('model_data/X_train.mat')
        y_train = loadmat('model_data/y_train.mat')

        X_train = X_train['X_train_nnmpc_temp']
        y_train = y_train['y_train_nnmpc_temp']

        X_train = X_train.astype(np.double)
        y_train = y_train.astype(np.double)

        batch_size = 30
        num_epochs = 1000

        train_data_set = SystemDataSet(X_train, y_train)
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

        controller_nn = train_nn_and_save(train_loader, controller, num_epochs=num_epochs, l1=None,
                                     pred_step=0, lr=1e-4, decay_rate=1.0, clr=None, path=controller_path)

    # connect the nn
    controller = nn_controller()
    controller.load_state_dict(torch.load(controller_path))

    nx = 4
    nu = 1

    # cartpole closed loop system
    # range of cartpole states are bounded by [10; pi; 10; 10]
    A, B = torch.from_numpy(Ad), torch.from_numpy(Bd)
    A, B = A.to(torch.float32), B.to(torch.float32)

    # x_0_lb = torch.tensor([[2.0, 0.2, -0.15, -0.2]])
    # x_0_ub = torch.tensor([[2.2, 0.3, -0.13, -0.1]])
    x_0_lb = torch.tensor([[0.0,  -0.2, np.pi/12, -0.15]])
    x_0_ub = torch.tensor([[0.3,  -0.1, np.pi/12 + 0.1, -0.05]])


    plot_dim = [2, 3]
    plt.figure()

    # sample trajectories from the nnds
    horizon = 10
    cartpole_nn = CartPoleNNSystem(A, B, res_model, controller)
    domain = Polyhedron.from_bounds(x_0_lb.squeeze(0).detach().numpy(), x_0_ub.squeeze(0).detach().numpy())
    init_states = ut.uniform_random_sample_from_Polyhedron(domain, 50)
    traj_list = ut.simulate_NN_system(cartpole_nn.forward, init_states, step=horizon - 1)

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim, color='gray', linewidth=0.5, alpha = 0.5)
    domain.plot(residual_dimensions=plot_dim, fill=False, ec='k', linestyle='--', linewidth=2)



