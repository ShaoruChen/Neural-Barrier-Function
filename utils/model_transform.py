import torch
import torch.nn as nn
import numpy as np

def agent_model_to_nn(env_model, state_size, action_size):
    scaler = env_model.scaler
    # TODO: add the transformation as a linear layer
    mu, std = scaler.mu, scaler.std
    assert mu is not None
    assert std is not None

    ensemble_model = env_model.ensemble_model
    device = next(ensemble_model.nn1.parameters()).device

    input_size = state_size + action_size
    hidden_size = ensemble_model.hidden_size
    ensemble_model_output_dim = ensemble_model.output_dim

    output_size = state_size

    num_hidden_layers = 4

    nn_model = nn.Sequential(nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, state_size)
                             ).to(device)

    ensemble_model_layers = [ensemble_model.nn1, ensemble_model.nn2, ensemble_model.nn3, ensemble_model.nn4]
    for i in range(4):
        # TODO: need to check the dimension of the matrices here
        weight = ensemble_model_layers[i].weight.data[0].T
        bias = ensemble_model_layers[i].bias.data[0]
        nn_model[2*i].weight.data = weight
        nn_model[2*i].bias.data = bias

    nn_model[-1].weight.data = ensemble_model.nn5.weight.data[0][:,:state_size].T
    nn_model[-1].bias.data = ensemble_model.nn5.bias.data[0][:state_size]

    # merge the scaling layer into the first linear layer
    scaling_layer = scaler_to_linear_layer(mu, std)
    scaling_layer = scaling_layer.to(device)
    
    new_layer = merge_two_linear_layers(scaling_layer, nn_model[0])
    nn_model[0].weight.data = new_layer.weight.data
    nn_model[0].bias.data = new_layer.bias.data

    return nn_model

def agent_policy_to_nn(agent):
    policy = agent.policy
    num_inputs = policy.linear1.in_features
    hidden_dim = policy.linear1.out_features
    output_dim = policy.mean.out_features

    nn_policy = nn.Sequential(nn.Linear(num_inputs, hidden_dim),
                              nn.ReLU(),
                              nn.Linear(hidden_dim, hidden_dim),
                              nn.ReLU(),
                              nn.Linear(hidden_dim, output_dim)
                              )

    nn_policy[0].weight.data, nn_policy[0].bias.data = policy.linear1.weight.data, policy.linear1.bias.data
    nn_policy[2].weight.data, nn_policy[2].bias.data = policy.linear2.weight.data, policy.linear2.bias.data
    nn_policy[-1].weight.data, nn_policy[-1].bias.data = policy.mean.weight.data, policy.mean.bias.data

    return nn_policy

def scaler_to_linear_layer(mu, std):
    # mu, std: np.array of size 1 x n
    dim = mu.shape[1]
    bias = - mu / std
    weight = np.diag(1.0/std[0])

    layer = nn.Linear(dim, dim)
    layer.weight.data = torch.from_numpy(weight)
    layer.bias.data = torch.from_numpy(bias[0])
    return layer

def merge_two_linear_layers(linear_1, linear_2):
    # merge two linear layers into one linear layer
    # W_2@(W_1@x + b_1) + b_2 = W_2@W_1 x + W_2@b_1 +  b_2

    input_dim = linear_1.in_features
    output_dim = linear_2.out_features
    device = next(linear_1.parameters()).device
    new_layer = nn.Linear(input_dim, output_dim).to(device)

    new_layer.weight.data = linear_2.weight.data@linear_1.weight.data
    new_layer.bias.data = linear_2.weight.data@linear_1.bias.data + linear_2.bias.data

    return new_layer




