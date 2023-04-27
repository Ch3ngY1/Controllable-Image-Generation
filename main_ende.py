import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
from training.trainer_ende import DefaultTrainer
from training.trainer_ende_poe import DefaultTrainer as DefaultTrainer_POE
import os
from utils import data_load
import numpy as np
from torchvision.transforms import InterpolationMode

def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.manual_seed(runseed)
    np.random.seed(runseed)


    dataset = getattr(data_load, args.data_name.lower())

    train_data = dataset(
        img_root=args.img_root,
        data_root=args.data_root,
        dataset='train',
        transform=transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=args.fold
    )

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             drop_last=True)

    val_data = dataset(
        img_root=args.img_root,
        data_root=args.data_root,
        dataset='valid',
        transform=transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=args.fold
    )

    val_load = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.poe:
        trainer = DefaultTrainer_POE(args)
    else:
        trainer = DefaultTrainer(args)
    trainer.train(train_load, val_load)


if __name__ == '__main__':
    cfg = BaseConfig()
    lr = 0.0001
    gpu_id = 0
    dataset = 'aesthetics'

    # ============================= Adience =============================
    if dataset == 'adience':
        weights = '5 5 0.2'
        for max_iter in [16000]:
            steps = max_iter // 4 * 3
            for fold in range(5):
                ratio = 0.9
                model_name = 'endetmix'
                show = True
                ckpt_name = 'Adience/{model_name}/'.format(model_name=model_name)
                fixed = '--gpu_id {gpu_id} ' \
                        '--model_name endevgg ' \
                        '--ratio {ratio} ' \
                        '--data_name faces_final_v3_ratio ' \
                        '--batch_size 18 ' \
                        '--lr {lr} ' \
                        '--num_classes 8 ' \
                        '--max_iter {max_iter} ' \
                        '--stepvalues {steps} ' \
                        '--warmup_steps 1000 ' \
                        '--weights {weights} ' \
                        '--exp_name fold_{fold} ' \
                        '--val_freq 40 ' \
                        '--fold {fold} ' \
                        '--save_folder /data2/chengyi/Ordinal_GAN/result/save_model/checkpoint_{ckpt_name}/ ' \
                        '--save_log /data2/chengyi/Ordinal_GAN/result/save_log/logs_{ckpt_name}/'.format(fold=fold,
                                                                                                         steps=steps,
                                                                                                         ckpt_name=ckpt_name,
                                                                                                         lr=lr,
                                                                                                         weights=weights,
                                                                                                         max_iter=max_iter,
                                                                                                         ratio=ratio,
                                                                                                         gpu_id=gpu_id)\
                .split()
                args = cfg.initialize(fixed, show=show)
                main(args)


    # ============================= dr_v3 =============================
    if dataset == 'dr':
        N = 12000
        steps = 3 * N
        max_iter = 4 * N
        model_name = 'endetmix'
        for fold in range(10):
            show = True if fold == 0 else False
            ckpt_name = 'DR/{model_name}/'.format(model_name=model_name)
            fixed = '--gpu_id 6 ' \
                    '--exp_name fold_{fold} ' \
                    '--model_name {model_name} ' \
                    '--data_name dr_v3 ' \
                    '--batch_size 18 ' \
                    '--lr {lr} ' \
                    '--num_classes 5 ' \
                    '--max_iter {max_iter} ' \
                    '--stepvalues {steps} ' \
                    '--warmup_steps 1000 ' \
                    '--val_freq 40 ' \
                    '--fold {fold} ' \
                    '--save_folder /data2/chengyi/Ordinal_GAN/result/save_model/checkpoint_{ckpt_name}/ ' \
                    '--save_log /data2/chengyi/Ordinal_GAN/result/save_log/logs_{ckpt_name}/'.format(fold=fold,
                                                                                                     steps=steps,
                                                                                                     ckpt_name=ckpt_name,
                                                                                                     lr=lr,
                                                                                                     max_iter=max_iter,
                                                                                                     model_name=model_name)\
            .split()
            args = cfg.initialize(fixed=fixed, show=show)
            main(args)

    # ============================= POE_DR_Further =============================
    if dataset == 'aesthetics':
        for fold in range(5):
                weights = '5 2 0.2'
                data_name = 'aesthetics_ratio'
                model_name = 'endetmix'
                steps = 2000
                sub_iter = 1000
                max_iter = 5000
                ckpt_name = 'Aesthetics/{model_name}/'.format(model_name=model_name)
                exp_name = 'fold_{}'.format(fold)
                fixed = '--gpu_id 0 ' \
                        '--exp_name {exp_name} ' \
                        '--fold {fold} ' \
                        '--model_name {model_name} ' \
                        '--data_name {data_name} ' \
                        '--batch_size 18 ' \
                        '--lr {lr} ' \
                        '--num_classes 5 ' \
                        '--max_iter {max_iter} ' \
                        '--stepvalues {steps} ' \
                        '--sub_iter {sub_iter} ' \
                        '--warmup_steps 1000 ' \
                        '--weights {weights} ' \
                        '--val_freq 40 ' \
                        '--save_folder /data2/chengyi/Ordinal_GAN/result/save_model/checkpoint_{ckpt_name}/ ' \
                        '--save_log /data2/chengyi/Ordinal_GAN/result/save_log/logs_{ckpt_name}/'.format(steps=steps,
                                                                                                         ckpt_name=ckpt_name,
                                                                                                         lr=lr,
                                                                                                         fold=fold,
                                                                                                         weights=weights,
                                                                                                         sub_iter=sub_iter,
                                                                                                         max_iter=max_iter,
                                                                                                         exp_name=exp_name,
                                                                                                         model_name=model_name,
                                                                                                         data_name=data_name) \
                    .split()
                args = cfg.initialize(fixed, show=True)
                main(args)


'''
cd /data2/chengyi/Ordinal_GAN/
source activate torch18
python main_ende.py

'''



