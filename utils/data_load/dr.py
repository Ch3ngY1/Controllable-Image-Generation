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
import random


class MyDataset(data_utils.Dataset):

    def __init__(self, img_root, data_root, dataset, transform=None, fold=3):
        self.data_list = []
        # self.items = []
        self.transform = transform

        if dataset == 'train':
            data_num = [i for i in range(10) if i != fold]
        elif dataset == 'valid':
            data_num = [fold]

        for i in data_num:
            f = open('data_split/dr/fold_{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data_list.append(row[1:])

        self.label_list = [int(x[-1]) for x in self.data_list]


        self.label_num = [0, 0, 0, 0, 0]
        for each in self.label_list:
            self.label_num[each] += 1
        print(self.label_num)


        print(len(self.data_list))

    def choose_ref(self, label):
        if label == 0:
            out = 1
        elif label == 4:
            out = 3
        else:
            left_num = self.label_num[label-1]
            right_num = self.label_num[label+1]
            out = label + 1 if random.random() < (left_num/(left_num+right_num)) else label - 1
        return out

    def __getitem__(self, idx):
        item = copy.deepcopy(self.data_list[idx])

        label = int(item[1])
        img_path = '/<your path>/DR_dataset/train/' + item[0] + '.jpg'

        ref_label = int(self.choose_ref(label))
        while True:
            a = random.randint(0, len(self.data_list) // 2)
            b = random.randint(len(self.data_list) // 2, len(self.data_list) - 1)
            try:
                idx2 = self.label_list.index(ref_label, a, b)
                break
            except:
                pass

        img_path_ref = '/<your path>/DR_dataset/train/' + copy.deepcopy(self.data_list[idx2])[
            0] + '.jpg'
        img_ref = Image.open(img_path_ref).convert('RGB')
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            img_ref = self.transform(img_ref)


        return img, img_ref, label, ref_label

    def __len__(self):
        return len(self.data_list)
