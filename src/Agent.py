"""
MIT License

Copyright (c) 2025 ZhaoZ77

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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





