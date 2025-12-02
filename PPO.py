import torch
import torch.nn.functional as F
from Agent import Actor,Critic
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from torch.optim import lr_scheduler



class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.scheduler_Actor = lr_scheduler.StepLR(self.actor_optimizer, step_size=2, gamma=0.5)
        self.scheduler_Critic = lr_scheduler.StepLR(self.critic_optimizer, step_size=2, gamma=0.5)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state, train_flag):
        if train_flag:
            state = torch.unsqueeze(state, dim=0)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        else:
            state = torch.unsqueeze(state, dim=0)
            probs = self.actor(state)

            action = torch.argmax(probs)
        return action.item(), probs


    def update(self, transition_dict):

        # States
        states = transition_dict['states']
        states = torch.stack(states)
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        # next states
        next_states = transition_dict['next_states']
        next_states = torch.stack(next_states)
        next_states = torch.tensor(next_states, dtype=torch.float)


        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)


        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()


        for _ in range(self.epochs):
            probs = self.actor(states).gather(1, actions)

            # prob loss  动作概率正则
            cost = 1.0 * (probs.mean() - 0.5) ** 2

            log_probs = torch.log(probs)
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage # 截断

            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            actor_loss = actor_loss + cost

            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()