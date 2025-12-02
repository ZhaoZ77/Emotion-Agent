from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from scipy.io import savemat
from utils import weights_init, save_checkpoint
import os.path as osp


use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if self.size() < batch_size:
            transitions = random.sample(self.buffer, self.size())

        else:
            transitions = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = zip(*transitions)
        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.tensor(reward)
        reward = torch.stack(reward)
        next_state = torch.stack(next_state)
        done = torch.stack(done)

        return state, action, reward, next_state, done

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent_KeyFrame(env, agent, train_epoch, episode_num, cluster_centers, cluster_sigma, test_id):
    """
    训练函数：按训练轮次（train_epoch）和每轮episode数量（episode_num）组织循环
    train_epoch: 总训练轮数
    episode_num: 每轮训练的episode数量
    总训练episode数 = train_epoch * episode_num
    """
    return_list = []

    for epoch in range(train_epoch):
        agent.scheduler_Actor.step()
        agent.scheduler_Critic.step()
        
        # 进度条：每轮处理episode_num个episode
        with tqdm(total=episode_num, desc=f'Epoch {epoch + 1}/{train_epoch}') as pbar:
            for i_episode in range(episode_num):
                env.train_flag = True
                episode_return = 0
                transition_dict = {
                    'states': [], 'actions': [], 'next_states': [], 
                    'rewards': [], 'dones': [], 'probs': []
                }
                state, _ = env.reset()
                done = False
                while not done:
                    action, probs = agent.take_action(state, env.train_flag)
                    next_state, reward, done= env.step(state, action, cluster_centers, cluster_sigma)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)

                # 每10个episode显示一次平均回报
                if (i_episode + 1) % 10 == 0:
                    return_listCpu = torch.stack(return_list)
                    last_10_mean = torch.mean(return_listCpu[-10:]).item()
                    # 显示当前总episode数：epoch*episode_num + i_episode + 1
                    pbar.set_postfix({
                        'total_episode': f'{epoch * episode_num + i_episode + 1}', 
                        'last_10_return': f'{last_10_mean:.3f}'
                    })
                pbar.update(1)

    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)