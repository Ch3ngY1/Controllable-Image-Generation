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


class MyDataset(data_utils.Dataset):

    def __init__(self, img_root, data_root, dataset, transform=None, fold=4):
        self.data_list = []
        self.transform = transform
        '''
        同时倒入frontal和普通的，
        frontal就是对应的aligned
        普通的对应的是faces
        '''
        root = '/data2/chengyi/dataset/ord_reg/AdienceBenchmarkGenderAndAgeClassification/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold'

        train = 'age_train.txt'
        test = 'age_test.txt'

        if dataset == 'train':
            file_name = [train]
        else:
            file_name = [test]

        for each in file_name:
            f_path = root + '/test_fold_is_' + str(fold) + '/' + each
            with open(f_path, 'r') as f:
                l = f.readlines()
                self.data_list.extend(l)

        self.label_list = [int(x[-2]) for x in self.data_list]
        print(len(self.data_list))
        # label:
        # [1601, 1172, 1469, 969, 2770, 1271, 413, 401, 0]
        # 15.9---11.64---14.59---9.62---27.51---12.62---4.1---3.98

    def choose_ref(self, label):
        if label == 0:
            out = 1
        elif label == 7:
            out = 6
        else:
            out = label + 1 if random.random() > 0.5 else label - 1
        return out

    def __getitem__(self, idx):
        item = copy.deepcopy(self.data_list[idx])
        img_path = item[:-3]
        label = int(item[-2])
        # img_path = item[2]
        ref_label = self.choose_ref(label)
        while True:
            a = random.randint(0, len(self.data_list) // 2)
            b = random.randint(len(self.data_list) // 2, len(self.data_list) - 1)
            try:
                idx2 = self.label_list.index(ref_label, a, b)
                break
            except:
                pass

        img_path_ref = copy.deepcopy(self.data_list[idx2])[:-3]
        img_path_ref = '/data2/wangjinhong/data/ord_reg/data/aligned/' + img_path_ref
        img_ref = Image.open(img_path_ref).convert('RGB')
        # label = item[-1]
        img_path = '/data2/wangjinhong/data/ord_reg/data/aligned/' + img_path
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            img_ref = self.transform(img_ref)

        # multi_hot_target = torch.zeros(1).long()
        # multi_hot_target[list(range(label))] = 1
        # # # 这是为了rank的模式：111111000000这种
        # multi_hot_target_ref = torch.zeros(1).long()
        # multi_hot_target_ref[list(range(label))] = 1
        multi_hot_target = multi_hot_target_ref = 0.
        # label = torch.tensor(float(label), dtype=torch.float32)

        return img, img_ref, label, ref_label, multi_hot_target, multi_hot_target_ref

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    train_dataloader = MyDataset(img_root=None, data_root='/data2/wangjinhong/data/ord_reg/data/', dataset='valid',
                                 transform=None, fold=0)
    train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=4, shuffle=True, num_workers=4)
    train_iter = iter(train_dataloader)
    a, b, c, d = train_iter.next()
