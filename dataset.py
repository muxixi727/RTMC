#!/user/bin/python3
# -*- coding:utf-8 -*-
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
# from utils import data_loader
from sklearn.preprocessing import OneHotEncoder


class MultiViewDataWithoutLeak(Dataset):
    def __init__(self, dataname, multi_view, std, train=True):
        super(MultiViewDataWithoutLeak, self).__init__()
        self.multi_view = multi_view
        self.std = std
        self.train = train

        if dataname is not None:
            x, y, label = data_loader_without_leak(dataname, self.multi_view, std)
            self.x = x
            self.y = y
            self.label = label
        else:
            self.x = dict()
            self.y = None
            self.label = None

    def __getitem__(self, index):
        data = dict()
        for n_v in range(len(self.x)):
            data[n_v] = self.x[n_v][index]
            # if self.std > 0:
            #     if not self.train:
            #         noise = np.random.normal(0, 1000000, data[n_v].shape).astype(np.float32)
            #         data[n_v] += noise

        y = self.y[index]
        label = self.label[index]
        return data, y, label

    def __len__(self):
        return len(self.x[0])

    def postprocessing(self, addNoise=False, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addConflict:
            self.addConflict(ratio_conflict)
        pass
        if addNoise:
            self.addNoise(ratio_noise)

    def addNoise(self, ratio):
        num_classes = len(np.unique(self.label))
        num_views = len(self.x)
        for i in range(len(self.label)):
            if num_views == 6:
                vs = np.random.choice(num_views, size=3, replace=False)
                for v in vs:
                    self.x[v][i] = self.x[v][i] * (1 - ratio) + ratio * np.random.normal(0, 10000)
            else:
                v = np.random.randint(num_views)
                self.x[v][i] = self.x[v][i] * (1 - ratio) + + ratio * np.random.normal(0, 10000)

    def addConflict(self, ratio):
        num_classes = len(np.unique(self.label))
        num_views = len(self.x)
        for c in range(num_classes):
            samples = np.where(self.label == c)[0].tolist()
            other_class_indices = (self.label != c).nonzero(as_tuple=True)[0]
            if len(other_class_indices) > 0:
                for sample in samples:
                    other_sample_idx = other_class_indices[torch.randint(0, len(other_class_indices), (1,)).item()]
                    if num_views == 6:
                        vs = np.random.choice(num_views, size=3, replace=False)
                        for v in vs:
                            self.x[v][sample] = self.x[v][sample] * (1 - ratio) + ratio * self.x[v][other_sample_idx]
                    else:
                        v = np.random.randint(num_views)
                        self.x[v][sample] = self.x[v][sample] * (1 - ratio) + ratio * self.x[v][other_sample_idx]

    def Split_Training_Test_Dataset(self, train_sam_ratio=0.75, std=0.0):
        sam_num = self.__len__()
        train_sam_num = torch.tensor(torch.ceil(torch.tensor(sam_num * train_sam_ratio)).item(), dtype=torch.int64)
        rand_sam_ind = torch.randperm(sam_num)
        train_sam_ind = rand_sam_ind[range(train_sam_num)]

        train_data_set = MultiViewDataWithoutLeak(None, self.multi_view, std)
        n_views = len(self.x)
        for view in range(n_views):
            train_data_set.x[view] = self.x[view][train_sam_ind]
        train_data_set.y = self.y[train_sam_ind]
        train_data_set.label = self.label[train_sam_ind]

        test_sam_ind = rand_sam_ind[range(train_sam_num, sam_num)]
        test_data_set = MultiViewDataWithoutLeak(None, self.multi_view, std, train=False)
        for view in range(n_views):
            test_data_set.x[view] = self.x[view][test_sam_ind]
        test_data_set.y = self.y[test_sam_ind]
        test_data_set.label = self.label[test_sam_ind]
        return train_data_set, test_data_set


class MultiViewData(Dataset):

    def __init__(self, dataname, train=True):
        super(MultiViewData, self).__init__()

        self.data_loaders, self.label_loaders, self.y_loaders, idx_train, idx_val = data_loader(dataname)
        if train:
            self.x = self.data_loaders['train']
            self.label = self.label_loaders['train']
            self.y = self.y_loaders['train']
            self.idx = idx_train
        else:
            self.x = self.data_loaders['val']
            self.label = self.label_loaders['val']
            self.y = self.y_loaders['val']
            self.idx = idx_val

    def __getitem__(self, index):
        data = dict()
        for n_v in range(len(self.x)):
            data[n_v] = (self.x[n_v][index])
        y = self.y[index]
        label = self.label[index]
        return data, y, label

    def __len__(self):
        return len(self.x[0])

