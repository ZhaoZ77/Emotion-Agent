"""
Created on Tue Dec 02 15:48:32 2025

@author: Zhihao Zhou

"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, in_dim=310, hid_dim=256, action_dim=2,num_layers=1):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(in_dim,hid_dim,num_layers=num_layers,bidirectional=True,batch_first=True)
        self.fc1 = nn.Linear(hid_dim*2, hid_dim)
        self.fc2 = nn.Linear(hid_dim,action_dim)


    def forward(self, state):
        h, _ = self.lstm(state)
        output = F.relu(h)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        prob = torch.softmax(output, dim=-1)
        # prob = torch.sigmoid(output)
        return prob


class Critic(nn.Module):
    def __init__(self, in_dim=1024, hid_dim=256, action_dim=2, num_layers=1):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(in_dim,hid_dim,num_layers=num_layers,bidirectional=True,batch_first=True)
        self.fc1 = nn.Linear(hid_dim*2, hid_dim)
        self.fc2 = nn.Linear(hid_dim,1)

    def forward(self, state):
        h, _ = self.lstm(state)
        output = F.relu(h)
        output = F.relu(self.fc1(output))
        value = F.relu(self.fc2(output))
        return value





