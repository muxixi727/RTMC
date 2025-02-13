#!/user/bin/python3
# -*- coding:utf-8 -*-
import math
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from sklearn.preprocessing import OneHotEncoder
# import gdown

# Calculate dissonance of a vector of alpha #
def getDisn(alpha):
    evi = alpha - 1
    s = torch.sum(alpha, axis=1, keepdims=True)
    blf = evi / s
    idx = np.arange(alpha.shape[1])
    diss = 0.0
    Bal = lambda bi, bj: 1 - torch.abs(bi - bj) / (bi + bj + 1e-8)
    for i in idx:
        score_j_bal = [blf[:, j] * Bal(blf[:, j], blf[:, i]) for j in idx[idx != i]]
        score_j = [blf[:, j] for j in idx[idx != i]]
        diss += blf[:, i] * sum(score_j_bal) / (sum(score_j) + 1e-8)
    return diss

def kl_divergence(alpha, num_classes):

    beta = torch.ones([1, num_classes], dtype=torch.float32).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def normalize(x):
    """
    :param x: input data
    :return: normalization
    """
    scaler = MinMaxScaler((0, 1))
    norm_x = scaler.fit_transform(x, 0)

    return norm_x


def data_loader_without_leak(data_name, multi_view, ood):
    data_path = "/home/zxj/data/"+data_name+'/'+data_name+'.mat'
    print(data_path)
    data = scipy.io.loadmat(data_path)
    x = dict()
    un = dict()
    if data_name != "PIE":
        if multi_view:
            n_views = len(data) - 4
        else:
            n_views = 1
        for i in range(n_views):
            if multi_view:
                x[i] = normalize(data['x' + str(i + 1)].astype(np.float32))
            else:
                x[i] = normalize(data['x' + str(i + 2)].astype(np.float32))
            x[i] = torch.FloatTensor(x[i])
        y = data['y']
    else:
        if multi_view:
            n_views = data['X'].shape[1]
        else:
            n_views = 1
        for i in range(n_views):
            ##########
            if multi_view:
                x[i] = normalize(data['X'][0][i].T.astype(np.float32))
            else:
                x[i] = normalize(data['X'][0][i + 1].T.astype(np.float32))
            x[i] = torch.FloatTensor(x[i])
        y = data['gt']
    y = OneHotEncoder(sparse_output=False).fit_transform(y)
    label = [np.argmax(i) for i in y]

    label = torch.FloatTensor(label)
    y = torch.FloatTensor(y)
    return x, y, label

def data_loader(data_name):
    """
    train_test_split
    """
    # "/home/zxj/data/" + args.data_name + '/' + args.data_name + '.mat'
    data_path = "/home/zxj/data/"+data_name+'/'+data_name+'.mat'
    print(data_path)
    data = scipy.io.loadmat(data_path)
    n_views = len(data) - 4
    x = dict()
    x_train = dict()
    x_test = dict()
    for i in range(n_views):
        x[i] = normalize(data['x' + str(i + 1)].astype(np.float32))
        x[i] = torch.FloatTensor(x[i])
    y = data['y']
    y = OneHotEncoder(sparse_output=False).fit_transform(y)
    label = [np.argmax(i) for i in y]

    label = torch.FloatTensor(label)
    y = torch.FloatTensor(y)
    # Choose different n_views in terms of different data
    if n_views == 2:  # CUB, Food101, HMDB, Caltech101, pie
        x1_train, x1_test, x2_train, x2_test, label_train, label_test, y_train, y_test, idx_train, idx_test \
            = train_test_split(x[0], x[1], label, y, list(range(len(y))), test_size=0.2)
        x_train[0] = x1_train
        x_train[1] = x2_train
        x_test[0] = x1_test
        x_test[1] = x2_test


    elif n_views == 6:  # Handwritten
        x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, x4_train, x4_test, x5_train, x5_test, x6_train, x6_test, \
        label_train, label_test, y_train, y_test = train_test_split(x[0], x[1], x[2], x[3], x[4], x[5], label, y, test_size=0.2)
        for i in range(n_views):
            x_train[i] = eval('x'+str(i+1)+'_train')
            x_test[i] = eval('x'+str(i+1)+'_test')

    elif n_views == 3:  # Scene15
        x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, label_train, label_test, y_train, y_test \
            = train_test_split(x[0], x[1], x[2], label, y, test_size=0.2)
        for i in range(n_views):
            x_train[i] = eval('x'+str(i+1)+'_train')
            x_test[i] = eval('x'+str(i+1)+'_test')

    data_loaders = {
        'train': x_train,
        'val': x_test,
    }
    label_loaders = {
        'train': label_train,
        'val': label_test,
    }
    y_loaders = {
        'train': y_train,
        'val': y_test,
    }

    return data_loaders, label_loaders, y_loaders, idx_train, idx_test
