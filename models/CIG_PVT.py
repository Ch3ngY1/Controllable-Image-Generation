import torch.nn as nn
import torch
import torch.nn.functional as F
from models.soft_target_and_lossfunc import l2_loss
from losses import ssim
from models import pvt


def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d

        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=6, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ffn(x)

        return x


class EnDe(nn.Module):
    def __init__(self, args):
        super(EnDe, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        self.d = pvt.feature_pvt_v2_b3()
        ckpt = torch.load('/data2/chengyi/.torch/models/pvt_v2_b3.pth')
        self.d.load_state_dict(ckpt)
        # 输出
        self.u = Transformer(
            512,
            512 * 4,
            depth=1,
            num_heads=8,
            dim_per_head=64,
        )
        self.dhead = nn.Linear(512, args.num_classes)
        self.uhead = nn.Linear(512, 768)
        self.ushape = nn.Linear(512, 512)
        self.ufeature = nn.Linear(512, 512)

        # tau = args.tau

        self.ushape = nn.Linear(512, round(512 * args.tau))
        self.ufeature = nn.Linear(512, round(512 * (1 - args.tau)))

    def loss(self, x, x_shape):
        x_feature = self.ufeature(x)
        res = torch.cat([x_shape, x_feature], dim=2)
        loss1 = ((res - x) ** 2).mean()  # --> 0
        return loss1


    def forward(self, x, y, cls=False):
        b, c, h, w = x.shape
        # x = torch.cat([x, y], dim=1)
        x = self.d(x)
        if cls:
            x = x.mean(dim=0).squeeze(dim=1)
            out = self.dhead(x)
            return out

        y = self.d(y)  # 49 * 2 * 512

        # x = torch.cat([x, y], dim=0)
        shape_x = self.ushape(x)
        feature_y = self.ufeature(y)
        # out4 = shape_x + feature_y
        x = torch.cat([shape_x, feature_y], dim=2)

        loss = self.loss(x, shape_x)

        x = x.repeat(4, 1, 1)
        x = x.permute(1, 0, 2)
        x = self.u(x)
        x = self.uhead(x)

        x = x.view(b, h // 16, w // 16, 16, 16, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return x, loss


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.net = EnDe(args)
        self.GAN = args.no_GAN
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.simi_loss = ssim.SSIM(size_average=False)
        self.name = 'UnetGenerator'
        self.alpha = args.alpha
        self.beta = args.beta
        self.lmbd = args.lmbd
        self.tau = args.tau

    def lossGen(self, gen_x, x, y):
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
            x_gen, loss_orth = self.net(x, y)
            lossGen1 = self.lossGen(x_gen, x, y)

            x_gen_cls = self.net(x_gen, x_gen, cls=True)

            y_cls = self.net(y, y, cls=True)

            loss_xgen_input = self.criterion(x_gen_cls, target_y).mean()

            loss_pred = loss_x_input.mean() + loss_xgen_input * self.lmbd
            lossGen2 = l2_loss(x_gen_cls, y_cls)

            loss_gen = lossGen1 * self.alpha + lossGen2 / ((x_gen_cls ** 2).mean() + (y_cls ** 2).mean()) * \
                       self.beta + loss_orth

            return x_cls, [loss_pred, loss_gen]

        return x_cls, loss_x_input.mean()


class Args():
    def __init__(self):
        self.no_GAN = True
        self.weights = [2, 5, 0.1]
        self.num_classes = 5


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    args = Args()
    data = torch.rand([2, 3, 224, 224]).cuda()
    label = torch.tensor([1, 2]).cuda()
    net = Model(args).cuda()
    pred = net(data, data, label, label)
    pred = net(data, data, label, label, GAN=True)
    # print(pred.shape)