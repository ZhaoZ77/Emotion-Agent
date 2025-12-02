# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:36:47 2020

"""
import numpy as np
import pandas as pd
import scipy.io
import random
import matplotlib.pyplot as plt
import os
import torch
import argparse
import h5py



use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Environment:
    def __init__(self,featuresPath, test_id:int):
        self.train_flag = True  # 1代表训练 0代表测试

        self.featuresPath = featuresPath   #feature file path
        self.test_id = test_id

        self.features_test = []
        self.features = []
        self.KeyFrame_pick_idxs = []

        self.KeyFrame_probs = []
        self.KeyFrame_test = []

        self.features_test = []
        self.labels_test = []

        self.video_length = 0
        # trial
        self.train_index = 0

        # time
        self.index = 0

        self.test_index = 0
        self.datasets = h5py.File(self.featuresPath, 'r')

        self.all_keys = self.datasets.keys()
        self.all_keys = sorted(self.all_keys, key=lambda x: int(''.join(filter(str.isdigit, x))))

        self.test_keys = self.all_keys[15*self.test_id : 15*(self.test_id + 1)]
        del self.all_keys[15 * self.test_id : 15 * (self.test_id + 1)]
        self.train_keys = self.all_keys

        self.done = False

        self.emotion_labels = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
   
    def readfile(self):

        if self.train_index == 210:
            self.train_index = 0

        index = self.train_index

        fea = self.datasets[self.train_keys[index]]['features'][...]

        fea = torch.tensor(fea).to(device)

        self.train_index = self.train_index + 1


        return fea
    
    def readfile_test(self):        

        if self.test_index == 15:
            self.test_index = 0

        index = self.test_index
        fea = self.datasets[self.test_keys[index]]['features'][...]
        fea = torch.tensor(fea).to(device)

        self.test_index = self.test_index + 1
        return fea
    
    def reset_test(self):
        self.features_test.clear()
        self.KeyFrame_pick_idxs.clear()
        fea = self.readfile_test()
        fea = torch.tensor(fea)

        feature_size = fea.shape
        self.video_length = feature_size[0]

        self.features_test.append(fea)
        self.index= 0
        return fea[0],self.video_length
    
    def reset(self):
        self.features.clear()
        self.KeyFrame_pick_idxs.clear()
        fea = self.readfile()

        fea = torch.tensor(fea)
        feature_size = fea.shape
        self.video_length = feature_size[0]

        self.features.append(fea)

        self.index=0

        return fea[0],self.video_length


    def step_test(self,probs):
        next_feature = self.features_test[0][self.index]
        self.done = False
        self.KeyFrame_probs.append(probs)

        if self.index == self.video_length-1:
            self.done = True
            
            KeyFrame_probs = torch.stack(self.KeyFrame_probs)
            order,_ = torch.sort(KeyFrame_probs)[::-1]
            key_frameLength = int(len(order) * 0.5)
            KeyFrame_idx = order[: key_frameLength]

            KeyFrame_idx = KeyFrame_idx.cpu()
            KeyFrame_idx = np.array(KeyFrame_idx)

            data = self.features_test
            key_frame = data[0][KeyFrame_idx]
            labels = len(KeyFrame_idx) * [self.emotion_labels[self.test_index-1]]

            self.KeyFrame_test.extend(key_frame)
            self.labels_test.extend(labels)

            self.KeyFrame_pick_idxs.clear()
            self.features_test.clear()
            self.KeyFrame_probs.clear()

        else:
            self.index = self.index+1

        return next_feature,self.done

    def step(self,seq,action, cluster_centers, cluster_sigma):
        next_feature = self.features[0][self.index]

        if action == 1:
            self.KeyFrame_pick_idxs.append(self.index)

        self.done = False

        if self.index == self.video_length-1:
            self.done = True
            reward = self.get_reward(seq, action, cluster_centers, cluster_sigma)
            self.features.clear()

        else:
            self.index = self.index+1

            reward = self.get_reward(seq,action, cluster_centers, cluster_sigma)
        return next_feature, reward, self.done
    
    
    def get_reward(self,seq,actions, cluster_centers, cluster_sigma):
        """
        Compute diversity reward and representativeness reward
        Args:
            seq: sequence of features, shape (1, seq_len, dim)
            actions: binary action sequence, shape (1, seq_len, 1)
            ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
            temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
            use_gpu (bool): whether to use GPU
        """
        _seq = seq.detach()
        _seq = _seq.unsqueeze(0)
        reward = torch.tensor([0.],dtype=torch.float64).to(device)
        _actions = actions.detach


        if _actions == 1:
            cluster_centers = cluster_centers
            cluster_sigma = cluster_sigma
            distances = torch.cdist(_seq.double(), cluster_centers.double())

            # # Distance
            dist, min_index = torch.min(distances, axis=1)
            min_index = min_index.item()

            reward_dis = 1 / (dist / ( cluster_sigma[min_index] ) + 1)

            inter = distances[0][min_index] / cluster_sigma[min_index]
            intra = 0.
            for i in range(distances.shape[1]):
                intra = intra + distances[0][i] * cluster_sigma[i]

            intra = intra - distances[0][min_index] * cluster_sigma[min_index]
            intra = intra / (cluster_sigma.sum() - cluster_sigma[min_index])
            reward_inter_intra = intra / inter


        if actions == 0 :
            reward_inter_intra = 0.
            reward_dis = 0.


        reward = reward_dis * 0.5 + reward_inter_intra * 0.33 * 0.1

        return reward