import os, sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import models
from datetime import datetime
from tensorboardX import SummaryWriter

from config.cfg import arg2str
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from evaluater import metric


class DefaultTrainer(object):

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.warmup_steps = args.warmup_steps
        self.eval_only = args.eval_only
        self.model = getattr(models, args.model_name.lower())(args)
        # state_dict = '/data2/chengyi/ord_reg/result/save_model/checkpoint_catar/test2/ResTrans_best_acc_0.0200656708329916.pth'
        # self.model.load_state_dict(torch.load(state_dict)['net_state_dict'])


        # state_dict = '/data2/chengyi/ord_reg/result/save_model/checkpoint_catar/my_method_l2_restrans2_binary/ResTrans_best_acc_0.6545056700706482.pth'
        # state_dict = torch.load(state_dict)
        # self.model.load_state_dict(state_dict['net_state_dict'])

        self.model.cuda()

        self.loss = nn.CrossEntropyLoss()
        self.max_acc = 0
        self.min_loss = 1000
        self.min_mae = 1000
        self.seperate_out = False

        self.start = 0

        self.optim = getattr(torch.optim, args.optim) \
            (filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=args.weight_decay)

        if args.resume:
            if os.path.isfile(self.args.resume):
                iter, index = self.load_model(args.resume)
                self.start_iter = iter


    def train_iter(self, step, dataloader):

        img, label = dataloader.next()
        img = img.float().cuda()
        label = label.cuda()
        # l0, l2, l4 = l0.cuda(), l2.cuda(), l4.cuda()


        self.model.train()
        if self.eval_only:
            self.model.eval()


        pred, loss = self.model(img, label)
        # loss = self.loss(pred[2], l0) + self.loss(pred[1], l2) + self.loss(pred[1], l4)

        # loss = self.loss()

        '''generate logger'''
        if self.start == 0:
            self.init_writer()
            self.start = 1


        print( 'Training - Step: {} - Loss: {:.4f}' \
               .format(step, loss.item()))

        loss.backward()
        self.optim.step()
        self.model.zero_grad()



        if step % self.args.display_freq == 0:

            if self.seperate_out:
                acc4, acc2, acc1, acc, mae = metric.seperate_acc(pred, label)
                print(
                    'Training - Step: {} - Acc4: {:.4f} - Acc2: {:.4f} - Acc1: {:.4f} - Acc1: {:.4f} - lr:{:.4f}' \
                    .format(step, acc4, acc2, acc1, acc, self.lr_current))
                scalars = [loss.item(), acc4, acc2, acc1, acc, mae, self.lr_current]
                names = ['loss', 'acc4', 'acc2', 'acc1', 'acc', 'MAE', 'lr']
            else:
                acc = metric.accuracy(pred, label)
                mae = 0
                print(
                    'Training - Step: {} - Acc: {:.4f} - MAE {:.4f} - lr:{:.4f}' \
                    .format(step, acc, mae, self.lr_current))
                scalars = [loss.item(), acc, mae, self.lr_current]
                names = ['loss', 'acc', 'MAE', 'lr']


            write_scalars(self.writer, scalars, names, step, 'train')



    def train(self, train_dataloader, valid_dataloader=None):

        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader)

        for step in range(self.start_iter, self.max_iter):

            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step,
                                                                             train_epoch_size))
                train_iter = iter(train_dataloader)

            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and (
                    step % self.args.val_freq == 0 or step == self.args.max_iter - 1) and (step != 0):
                val_iter = iter(valid_dataloader)
                val_loss, val_acc, val_mae = self.validation(step, val_iter, val_epoch_size)
                if val_acc > self.max_acc:
                    self.delete_model(best='best_acc', index=self.max_acc)
                    self.max_acc = val_acc
                    self.save_model(step, best='best_acc', index=self.max_acc, gpus=1)

                if val_loss.item() < self.min_loss:
                    self.delete_model(best='min_loss', index=self.min_loss)
                    self.min_loss = val_loss.item()
                    self.save_model(step, best='min_loss', index=self.min_loss, gpus=1)

                # if val_mae.item() < self.min_mae:
                #     self.delete_model(best='min_mae', index=self.min_mae)
                #     self.min_mae = val_mae.item()
                #     self.save_model(step, best='min_mae', index=self.min_mae, gpus=1)
        print('best_acc = {}'.format(self.max_acc))
        return self.min_loss, self.max_acc, self.min_mae
        # if step % self.args.save_freq == 0 and step != 0:
        #     self.model.save_model(step, best='step', index=step, gpus=1)


    def validation(self, step, val_iter, val_epoch_size):

        print('============Begin Validation============:step:{}'.format(step))

        self.model.eval()

        total_score = [[],[],[]]
        total_target = [[],[],[]]
        loss = 0.
        with torch.no_grad():
            for i in range(val_epoch_size):

                img, label = next(val_iter)
                img = img.float().cuda()
                target = label.cuda()
                # l0, l1, l2 = l0.cuda(), l1.cuda(), l2.cuda()
                # target = l0 + l1 * 2 + l2 * 4



                # img, target = next(val_iter)
                # img = img.float().cuda()
                # target = target.cuda()


                score, loss_out = self.model(img, target)


                if i == 0:
                    total_score = score
                    total_target = target
                else:
                    # if len(score.shape) == 1:
                    #     score = score.unsqueeze(0)

                    total_score = torch.cat((total_score, score), 0)
                    # total_score[1] = torch.cat((total_score[1], score[1]), 0)
                    # total_score[2] = torch.cat((total_score[2], score[2]), 0)

                    total_target = torch.cat((total_target, target), 0)
                    # total_target[1] = torch.cat((total_target[1], l1), 0)
                    # total_target[2] = torch.cat((total_target[2], l2), 0)


        # loss = self.loss(total_score, total_target)
                loss += loss_out#self.loss(total_score[0], total_target[0]) + self.loss(total_score[1], total_target[1]) + self.loss(total_score[2], total_target[2])
        # acc = metric.accuracy(total_score, total_target)

        # acc, mae = metric.accuracy_new(total_score, total_target)

        mae = 0
        if self.seperate_out:
            acc = (total_score == total_target).float().mean()
            acc1 = (total_score % 2 == total_target % 2).float().mean()
            acc2 = ((total_score % 4) // 2 == (total_target % 4) // 2).float().mean()
            acc4 = (total_score // 4 == total_target // 4).float().mean()

            print(
                'Valid - Step: {} - Acc4: {:.4f} - Acc2: {:.4f} - Acc1: {:.4f} - Loss: {:.4f} - Acc: {:.4f} - lr:{:.4f}' \
                    .format(step, acc4, acc2, acc1, loss/len(total_score), acc, self.lr_current))
            scalars = [loss.item(), acc4, acc2, acc1, mae, self.lr_current]
            names = ['loss', 'acc4', 'acc2', 'acc1', 'MAE', 'lr']
        else:
            acc = metric.accuracy(total_score, total_target)
            print(
                'Valid - Step: {} \n Loss: {:.4f} \n Acc: {:.4f} \n MAE: {:.4f}' \
                    .format(step, loss.item(), acc, mae))
            scalars = [loss.item(), acc, mae]
            names = ['loss', 'acc', 'MAE']

        write_scalars(self.writer, scalars, names, step, 'val')

        # acc = acc2

        return loss, acc, mae



################

    def _adjust_learning_rate_iter(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if step <= self.warmup_steps:  # 增大学习率
            self.lr_current = self.args.lr * float(step) / float(self.warmup_steps)

        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iter) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):
        """ Tensorboard writer initialization
            """

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)


        if self.args.exp_name == 'test':
            log_path = os.path.join(self.args.save_log, self.args.exp_name)
        else:
            log_path = os.path.join(self.args.save_log, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + self.args.exp_name)
        log_config_path = os.path.join(log_path, 'configs.log')

        self.writer = SummaryWriter(log_path)
        with open(log_config_path, 'w') as f:
            f.write(arg2str(self.args))


    def load_model(self, model_path):
        if os.path.exists(model_path):
            load_dict = torch.load(model_path)
            net_state_dict = load_dict['net_state_dict']

            try:
                self.model.load_state_dict(net_state_dict)
            except:
                self.model.module.load_state_dict(net_state_dict)
            self.iter = load_dict['iter'] + 1
            index = load_dict['index']

            print('Model Loaded!')
            return self.iter, index
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def delete_model(self, best, index):
        if index == 0 or index == 1000000:
            return
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        if os.path.exists(save_path):
            os.remove(save_path)

    def save_model(self, step, best='best_acc', index=None, gpus=1):

        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        if gpus == 1:
            save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        else:
            save_fname = '%s_%s_%s.pth' % (self.model.module.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.module.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved')

def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)