import torch.nn as nn
import torch
import torch.nn.functional as F
from models.soft_target_and_lossfunc import l2_loss
from losses import ssim
from models.POE_all import vgg16_bn

class tmp(nn.Module):
    def __init__(self):
        super(tmp, self).__init__()
        self.backbone = vgg16_bn()

class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]

        backbone = vgg16_bn(pretrained=True, num_output_neurons=args.num_classes)
        backbone = nn.Sequential(*list(backbone.children()))
        self.dclassifier = backbone[1]
        # self.demb = backbone[2]
        # self.dvar = backbone[3]
        # self.ddrop = backbone[4]
        self.dfinal = backbone[5]

        # 下采样
        self.d1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=3, padding=1), backbone[0][1:6])
        self.d2 = backbone[0][6:13]
        self.d3 = backbone[0][13:23]
        self.d4 = backbone[0][23:33]
        self.d5 = backbone[0][33:43]
        self.d6 = backbone[0][43]
        # 上采样
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # 输出
        self.uo = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], 3, 3, 1, 1),
            nn.Sigmoid(),
            # BCELoss
        )


    def forward(self, x, y, cls=False):
        x = torch.cat([x, y], dim=1)

        out_1 = self.d1(x)
        out_2 = self.d2(out_1)
        out_3 = self.d3(out_2)
        out_4 = self.d4(out_3)
        out4 = self.d5(out_4)

        if cls:
            out4 = self.d6(out4)
            x = out4.view(out4.size(0), -1)
            x = self.dclassifier(x)
            x = self.dfinal(x)
            return x


        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.uo(out8)
        return out

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.net = UNet(args)
        self.GAN = args.no_GAN
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.simi_loss = ssim.SSIM(size_average=False)
        self.weight_pred = args.weights
        self.name = 'ENDEvgg'

    def lossGen(self, gen_x, x, y):
        '''
        SSIM越到表示两个图越相似
        目的是让gen_x和x相似但是gen_x和y不相似，因此计算loss simi和loss diff
        由于图像差异性问题，因此做归一化(softmax)
        在图像完全一致时，ssim=1，因此通过1-0来表示tgt，但是在这里0也是有意义的，因此使用BCELoss
        '''
        loss_simi = self.simi_loss(gen_x, x)
        loss_diff = self.simi_loss(gen_x, y)
        loss_cat = torch.cat([loss_simi.unsqueeze(1), loss_diff.unsqueeze(1)], dim=1)
        loss_cat = F.softmax(loss_cat, dim=1)
        tgt = torch.cat([torch.ones_like(loss_simi.unsqueeze(1)), torch.zeros_like(loss_diff.unsqueeze(1))], dim=1)
        lossGen = nn.BCELoss()(loss_cat, tgt)
        return lossGen

    def model_name(self):
        return self.name

    def forward(self, x, y, target_x, target_y, GAN=False):

        x_cls = self.net(x, x, cls=True)
        loss_x_input = self.criterion(x_cls, target_x)
        if self.training and GAN:
            x_gen = self.net(x, y)
            lossGen1 = self.lossGen(x_gen, x, y)

            x_gen_cls = self.net(x_gen, x_gen, cls=True)

            y_cls = self.net(y, y, cls=True)

            # simi, diff = ((x_gen_cls - x_cls) ** 2).mean(dim=1), ((x_gen_cls - y_cls) ** 2).mean(dim=1)
            # simi, diff = F.softmax(torch.stack([simi, diff]), dim=0)
            # loss2x = self.criterion(x_gen_cls, target_x) * diff #* simi
            # loss2y = self.criterion(x_gen_cls, target_y) * simi #* diff
            #
            # loss_xgen_input = loss2x.mean() + loss2y.mean()

            loss_xgen_input = self.criterion(x_gen_cls, target_y).mean()
            # loss_xgen_input = (-loss2x.mean() + loss2y.mean()) * 400

            '''
            diff * self.criterion(x_gen_cls, target_y) - self.criterion(x_gen_cls, target_x) * simi
            
            '''
            # loss_xgen_input = (diff - simi).mean() * 400 # 完全用不了

            loss_pred = loss_x_input.mean() + loss_xgen_input * self.weight_pred[-1]
            lossGen2 = l2_loss(x_gen_cls, y_cls)

            loss_gen = lossGen1 * self.weight_pred[0] + lossGen2 / ((x_gen_cls ** 2).mean() + (y_cls ** 2).mean()) * self.weight_pred[1]

            return x_cls, [loss_pred, loss_gen]

        return x_cls, loss_x_input.mean()


class Args():
    def __init__(self):
        self.no_GAN = False
        self.weights = 1.
        self.num_classes = 5

if __name__ == '__main__':
    args = Args()
    data = torch.rand([2, 3, 224, 224]).cuda()
    label = torch.tensor([1,2]).cuda()
    net = Model(args).cuda()
    net.train()
    pred = net(data, data, label, label, GAN=True)
    print(pred.shape)


'''
通过两个优化器来优化模型
'''