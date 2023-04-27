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

        root = 'data_split/aesthetics/stratified/'

        f = open(root + dataset + '_{}.csv'.format(fold), "r")
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            self.items.append(row[1:])

        print(len(self.items))
        self.label_list = [int(x[-1]) for x in self.items]

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

        return img, img_ref, label, ref_label

    def __len__(self):
        return len(self.items)
