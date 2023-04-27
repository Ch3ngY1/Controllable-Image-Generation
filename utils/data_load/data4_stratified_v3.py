import random

import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from xpinyin import Pinyin
from PIL import Image
import copy
import pandas as pd
import csv

mapping = {'urban': 0, 'people': 1, 'nature': 2, 'animals': 3}

class MyDataset(data_utils.Dataset):

    def __init__(self, img_root, data_root, dataset, transform=None, fold=3):
        self.data_list = []
        self.items = []
        self.transform = transform

        root = '/data2/chengyi/dataset/ord_reg/aesthetics/stratified/'

        '''
        制作分层采样数据集：
        '''
        # subcls = [[[],[],[],[],[]] for _ in range(4)]
        # self.items = []
        # with open(root + 'all.csv', 'r') as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     for row in reader:
        #         _, id, sub, imgpath, _, label = row
        #         sub = mapping[sub]
        #         item = [id, sub, imgpath, label]
        #         subcls[sub][int(label)].append(item)
        #         # self.items.append(row[1:])
        #
        # train = []
        # valid = []
        # for sub_i in range(4):
        #     for label_j in range(5):
        #         current = subcls[sub_i][label_j]
        #         random.shuffle(current)
        #         interval = len(current) // 4
        #         valid.extend(current[:interval])
        #         train.extend(current[interval:])
        #         pass
        #
        #
        # column = ['id', 'sub_cls', 'img', 'label']
        # test = pd.DataFrame(columns=column, data=valid)
        # test.to_csv(root + 'valid.csv', encoding='gbk')
        #
        # test = pd.DataFrame(columns=column, data=train)
        # test.to_csv(root + 'train.csv', encoding='gbk')

        '''
        加载数据集
        '''
        # dataset = 'validation' if dataset == 'valid' else dataset
        f = open(root + dataset + '_{}.csv'.format(fold), "r")
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            self.items.append(row[1:])


        # print(subcls)
        print(len(self.items))
        self.label_list = [int(x[-1]) for x in self.items]
        # {'urban': 0, 'people': 1, 'nature': 2, 'animals': 3}: [818, 713, 1053, 836]

    def choose_ref(self, label):
        if label == 0:
            out = 1
        elif label == 4:
            out = 3
        else:
            out = label + 1 if random.random() > 0.5 else label - 1
        return out


    def __getitem__(self, idx):
        # 'id', 'sub_cls', 'img', 'label'
        item = copy.deepcopy(self.items[idx])
        img_path = item[2]
        label = int(item[-1])

        ref_label = self.choose_ref(label)
        while True:
            a = random.randint(0, len(self.items) // 2)
            b = random.randint(len(self.items) // 2, len(self.items) - 1)
            try:
                idx2 = self.label_list.index(ref_label, a, b)
                break
            except:
                pass

        item_ref = copy.deepcopy(self.items[idx2])
        img_path_ref = item_ref[2]

        img_ref = Image.open(img_path_ref).convert('RGB')
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img_ref = self.transform(img_ref)
            img = self.transform(img)

        return img, img_ref, label, ref_label, 0., 0.

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    a = MyDataset(None, None, 'valid')