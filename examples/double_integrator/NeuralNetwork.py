
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, path, A=None, B=None, c=None):
        super().__init__()
        stateDictionary = torch.load(path, map_location=torch.device("cpu"))
        layers = []
        for keyEntry in stateDictionary:
            if "weight" in keyEntry:
                layers.append(nn.Linear(stateDictionary[keyEntry].shape[1], stateDictionary[keyEntry].shape[0]))
                layers.append(nn.ReLU())
        layers.pop()
        self.Linear = nn.Sequential(
            *layers
        )
        self.rotation = nn.Identity()
        self.load_state_dict(stateDictionary)
        
        self.A = A
        self.B = B
        self.c = c
        if self.A is None:
            dimInp = self.Linear[0].weight.shape[1]
            self.A = torch.zeros((dimInp, dimInp)).float()
            self.B = torch.eye((dimInp)).float()
            self.c = torch.zeros(dimInp).float()
        self.repetition = 1

    def load(self, path):
        stateDict = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(stateDict)

    def setRepetition(self, repetition):
        self.repetition = repetition

    def forward(self, x):
        x = self.rotation(x)
        for i in range(self.repetition):
            x = x @ self.A.T + self.Linear(x) @ self.B.T + self.c
        return x
