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
from evaluater.metric_poe import get_metric


def is_fc(para_name):
    split_name = para_name.split('.')
    if split_name[-2] == 'final':
        return True
    else:
        return False


class DefaultTrainer(object):

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.sub_iter = args.max_iter//2 if args.sub_iter == -1 else args.sub_iter
        self.warmup_steps = args.warmup_steps
        self.eval_only = args.eval_only
        self.main_loss_type = args.main_loss_type

        self.poe = args.poe
        self.gan = args.no_GAN
        self.model = getattr(models, args.model_name.lower())(args)

        # root = '/data2/chengyi/Ordinal_GAN/result/save_model/checkpoint_ENDE/Faces/REMAKE/'
        # for each in os.listdir(root +'fold_{}'.format(args.fold)):
        #     if 'acc' in each:
        #         state_dict = root +'fold_{}/'.format(args.fold) + each
        #         break

        # state_dict = torch.load(state_dict)['net_state_dict']
        # self.model.load_state_dict(state_dict)

        self.model.cuda()
        self.max_acc = 0
        self.min_loss = 1000
        self.min_mae = 1000
        self.min_mae2 = 1000
        self.loss_name = args.loss_name
        self.start = 0
        self.wrong = None
        self.log_path = os.path.join(self.args.save_folder, self.args.exp_name, 'result.txt')

        # 这个只是用于vgg2的
        print('LR = 0.0001')

        params_pre = []
        params_gen = []
        for keys, param_value in self.model.net.named_parameters():
            # print(keys)
            if keys[0] == 'd':
                params_pre += [{'params': [param_value], 'lr': self.lr}]
            elif keys[0] == 'u':
                params_gen += [{'params': [param_value], 'lr': self.lr * 50}]
            else:
                raise KeyError

        self.optim_pred = torch.optim.Adam(params_pre, lr=self.lr,
                                           betas=(0.9, 0.999), eps=1e-08)
        # self.optim_pred = torch.optim.SGD(params_pre, lr=self.lr * 10)
        # self.optim_gen = torch.optim.SGD(params_gen, lr=self.lr * 500)
        #
        self.optim_gen = torch.optim.Adam(params_gen, lr=self.lr * 50,
                                          betas=(0.9, 0.999), eps=1e-08)

    def train_iter(self, step, dataloader):

        img, img2, label, label2, mh, mh2 = dataloader.next()
        img, img2 = img.float().cuda(), img2.float().cuda()
        label, label2 = label.cuda(), label2.cuda()

        if self.main_loss_type != 'rank':
            mh = mh2 = None
        self.model.train()
        if self.eval_only:
            self.model.eval()

        if self.start == 0:
            self.init_writer()
            self.start = 1

        if step > self.sub_iter or self.args.cls_only:
            # only cls
            logit, loss = self.model(img, img2, label, label2)

            print('Only Pred Training - Step: {} - Loss: {:.4f}' \
                  .format(step, loss.item()))

            loss.backward()
            self.optim_pred.step()
            loss_print = loss
        # elif step < 0:
        #     # only gan
        #     logit, loss = self.model(img, img2, label, label2, GAN=True)
        #     print('Only Generator Training - Step: {} - Loss: {:.4f}' \
        #           .format(step, loss[0].item()))
        #     # loss_pred, loss_gen
        #     # loss[0].backward(retain_graph=True)
        #     loss[1].backward()
        #     # self.optim_pred.step()
        #     self.optim_gen.step()
        #     loss_print = loss[1]
        else:
            # both
            logit, loss = self.model(img, img2, label, label2, GAN=True)
            print('Both Training - Step: {} - Loss: {:.4f}' \
                  .format(step, loss[0].item()))

            loss[0].backward(retain_graph=True)
            loss[1].backward()
            self.optim_pred.step()
            self.optim_gen.step()
            loss_print = loss[0]

        self.model.zero_grad()

        if step % self.args.display_freq == 0:
            if 'poe' in self.args.model_name:
                acc, mae = metric.cal_mae_acc_cls(logit, label)
            else:
                acc = metric.accuracy(logit, label)
                mae = metric.MAE(logit, label)
            print(
                'Training - Step: {} - Acc: {:.4f} - MAE {:.4f} - lr:{:.4f}' \
                    .format(step, acc, mae, self.lr_current))
            scalars = [loss_print.item(), acc, mae, self.lr_current]
            names = ['loss_pred', 'acc', 'MAE', 'lr']
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
                val_loss, val_acc, val_mae, val_mae2 = self.validation(step, val_iter, val_epoch_size)
                if val_acc > self.max_acc:
                    self.delete_model(best='best_acc', index=self.max_acc)
                    self.max_acc = val_acc
                    self.save_model(step, best='best_acc', index=self.max_acc, gpus=1)
                    self.log = open(self.log_path, mode='a')
                    self.log.write('step = {}, best_ACC, [acc, mae] = {}\n'.format(step, [val_acc, val_mae]))
                    self.log.close()

                if val_loss.item() < self.min_loss:
                    self.delete_model(best='min_loss', index=self.min_loss)
                    self.min_loss = val_loss.item()
                    self.save_model(step, best='min_loss', index=self.min_loss, gpus=1)

                if val_mae.item() < self.min_mae:
                    self.delete_model(best='min_mae', index=self.min_mae)
                    self.min_mae = val_mae.item()
                    self.save_model(step, best='min_mae', index=self.min_mae, gpus=1)
                    self.log = open(self.log_path, mode='a')
                    self.log.write('step = {}, best_MAE, [acc, mae] = {}\n'.format(step, [val_acc, val_mae]))
                    self.log.close()


        return self.min_loss, self.max_acc, self.min_mae
        # if step % self.args.save_freq == 0 and step != 0:
        #     self.model.save_model(step, best='step', index=step, gpus=1)

    def validation(self, step, val_iter, val_epoch_size):

        print('============Begin Validation============:step:{}'.format(step))

        self.model.eval()
        loss = 0.
        total_score = []
        total_target = []
        with torch.no_grad():
            for i in range(val_epoch_size):

                img, img2, target, target2, mh, mh2 = next(val_iter)
                img, img2 = img.float().cuda(), img2.float().cuda()
                target, target2 = target.cuda(), target2.cuda()
                if self.main_loss_type != 'rank':
                    mh = mh2 = None

                logit, loss_tmp = self.model(img, img2, target, target2)

                loss += loss_tmp

                if i == 0:
                    total_logit = logit
                    total_target = target
                else:
                    if 'poe' in self.args.model_name:
                        total_logit = torch.cat([total_logit, logit], dim=1)
                    else:
                        total_logit = torch.cat([total_logit, logit], dim=0)
                    total_target = torch.cat([total_target, target], 0)

        if 'poe' in self.args.model_name:
            acc, mae = metric.cal_mae_acc_cls(total_logit, total_target)
        else:
            acc = metric.accuracy(total_logit, total_target)
            mae = metric.MAE(total_logit, total_target)
            mae2 = metric.Arg_MAE(total_logit, total_target)

        '''
        记录做错的img
        '''
        # self.wrong_perspective_target = total_target.cpu().numpy()
        # _, pred = total_score.max(1)
        # wrong = (pred != total_target).float()
        # if self.wrong:
        #     self.wrong += wrong
        # else:
        #     self.wrong = wrong

        print(
            'Valid - Step: {} \n Loss: {:.4f} \n Acc: {:.4f} \n MAE: {:.4f} \n MAE2: {:.4f}' \
                .format(step, loss.item(), acc, mae, mae2))

        scalars = [loss.item(), acc, mae, mae2]
        names = ['loss', 'acc', 'MAE', 'MAE2']
        write_scalars(self.writer, scalars, names, step, 'val')

        return loss, acc, mae, mae2

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

        for param_group in self.optim_pred.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):
        """ Tensorboard writer initialization
            """

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)

        if self.args.exp_name == 'test':
            log_path = os.path.join(self.args.save_log, self.args.exp_name)
        else:
            log_path = os.path.join(self.args.save_log,
                                    datetime.now().strftime(
                                        '%b%d_%H-%M-%S') + '_' + self.args.exp_name)
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
            if isinstance(index, list):
                save_fname = '%s_%s_%s_%s.pth' % (self.model.model_name(), best, index[0], index[1])
            else:
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
