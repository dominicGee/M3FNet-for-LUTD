import os
import random

import torch
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

class SequenceDataset3Branch():
    def  __init__(self, sheet1_path, sheet2_path, sheet3_path, label_path, transform=None):
        # 读取数据
        self.seq1 = pd.read_csv(sheet1_path, header=None).values.astype('float32')
        self.seq2 = pd.read_csv(sheet2_path, header=None).values.astype('float32')
        self.seq3 = pd.read_csv(sheet3_path, header=None).values.astype('float32')
        self.labels = pd.read_excel(label_path)['label3'].values  # 读取label3列

        # 提取标签为0、1、2、3的样本
        mask = (self.labels == 0) | (self.labels == 1) | (self.labels == 2) | (self.labels == 3)
        self.seq1 = self.seq1[mask]
        self.seq2 = self.seq2[mask]
        self.seq3 = self.seq3[mask]
        self.labels = self.labels[mask]

        # 应用变换
        self.transform = transform

        # 打印每种标签的样本数量
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f'标签 {label} 的样本数: {count}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample1 = self.seq1[idx].astype('float32')
        sample2 = self.seq2[idx].astype('float32')
        sample3 = self.seq3[idx].astype('float32')
        label = self.labels[idx]

        if self.transform:
            sample1 = torch.tensor(sample1)
            sample2 = torch.tensor(sample2)
            sample3 = torch.tensor(sample3)
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
            sample3 = self.transform(sample3)

        return sample1, sample2, sample3, label

class SequenceDataset3Branchbothlabel():
    def  __init__(self, sheet1_path, sheet2_path, sheet3_path, label_path1, label_path2, transform=None):
        # 读取数据
        self.seq1 = pd.read_csv(sheet1_path, header=None).values.astype('float32')
        self.seq2 = pd.read_csv(sheet2_path, header=None).values.astype('float32')
        self.seq3 = pd.read_csv(sheet3_path, header=None).values.astype('float32')
        self.labels1 = pd.read_excel(label_path1)['label1'].values
        self.labels2 = pd.read_excel(label_path2)['label2'].values
        # 提取标签为0、1的样本
        mask = (self.labels1 == 0) | (self.labels1 == 1)
        self.seq1 = self.seq1[mask]
        self.seq2 = self.seq2[mask]
        self.seq3 = self.seq3[mask]
        self.labels1 = self.labels1[mask]
        self.labels2 = self.labels2[mask]

        # 应用变换
        self.transform = transform

        # 打印每种标签的样本数量
        unique1, counts1 = np.unique(self.labels1, return_counts=True)
        for label1, counts1 in zip(unique1, counts1):
            print(f'标签1 {label1} 的样本数: {counts1}')
        unique2, counts2 = np.unique(self.labels2, return_counts=True)
        for label2, counts2 in zip(unique2, counts2):
            print(f'标签2 {label2} 的样本数: {counts2}')

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):
        sample1 = self.seq1[idx].astype('float32')
        sample2 = self.seq2[idx].astype('float32')
        sample3 = self.seq3[idx].astype('float32')
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
            sample3 = self.transform(sample3)

        sample1 = torch.tensor(sample1)
        sample2 = torch.tensor(sample2)
        sample3 = torch.tensor(sample3)

        return sample1, sample2, sample3, label1, label2
