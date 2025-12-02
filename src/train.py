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


import torch
import random
import numpy as np
import rl_utils
from PPO import PPO
from EEG_Environment import Environment
from cluster import clustering
import matplotlib.pyplot as plt



device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


actor_lr = 1e-4
critic_lr = 1e-3

# 原总episode数为6300，这里设置为30轮，每轮210个episode（30*210=6300，保持总数量不变）
train_epoch = 30        # 训练轮数
episode_num = 210       # 每轮训练的episode数量

hidden_dim = 256
gamma = 0.98
lmbda = 0.95            # GAE
epochs = 5
eps = 0.2               # PPO截断参数
num_of_class = 3        # Kmeans聚类数量


state_dim = 310
action_dim = 2   # 动作空间：0和1


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


feature_path = "Your EEG Dataset"


for test_id in range(15):
    env = Environment(feature_path, test_id)

    cluster_centers, cluster_sigma = clustering(env.datasets, env.train_keys, num_of_class)

    cluster_centers = torch.tensor(cluster_centers).to(device)
    cluster_sigma = torch.tensor(cluster_sigma).to(device)

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent_KeyFrame(
        env, agent, train_epoch, episode_num, cluster_centers, cluster_sigma, test_id
    )

