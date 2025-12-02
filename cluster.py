import torch
import random
from utils import normalize
import numpy as np
from sklearn.cluster import KMeans


def clustering(datasets, train_keys, num_of_class):
    # 跨被试被试的数据估计全局情绪原型
    print(train_keys)
    x_train = []
    
    for key in train_keys:
        seq  = datasets[key]['features'][...]
        seq = np.array(seq)
        x_train.extend(seq)

    x_train = np.array(x_train)
    kmeans = KMeans(n_clusters=num_of_class)
    kmeans.fit(x_train)
    labels = kmeans.predict(x_train)
    KMeans_centers = kmeans.cluster_centers_
    print(np.unique(labels))

    # 计算每个簇的簇内方差（衡量簇内样本的离散程度）
    cluster_variance = []  # 存储每个簇的簇内方差（替换原 cluster_compactness）
    for cluster_label in np.unique(labels):
        cluster_points = x_train[labels == cluster_label]  # 当前簇的所有样本 [N, D]（N为簇内样本数，D为特征维度）
        centroid = KMeans_centers[cluster_label]  # 当前簇的中心 [D,]
        centroid = np.expand_dims(centroid, axis=0)  # 扩展维度为 [1, D]，适配广播计算
        
        # 步骤1：计算簇内每个样本到中心的欧氏距离
        distances = np.linalg.norm(cluster_points - centroid, axis=1)  # 形状 [N,]，每个元素是单个样本到中心的欧氏距离
        # 步骤2：计算距离的平方（簇内方差定义核心：平方误差的平均值）
        distances_sq = np.square(distances)
        # 步骤3：计算平方距离的平均值 → 簇内方差
        cluster_var = np.mean(distances_sq)
        cluster_variance.append(cluster_var)

    # 转为 numpy 数组并扩展为列向量（形状：[num_of_class, 1]），保持与原代码输出格式一致
    cluster_weights = np.array(cluster_variance)
    cluster_weights = np.expand_dims(cluster_weights, axis=1)

    return KMeans_centers, cluster_weights
