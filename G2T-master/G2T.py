import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.data import Data
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# from torch_geometric.nn import SAGEConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, adj, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.adj = adj
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.I = torch.eye(len(self.adj[0]), len(self.adj[0]), requires_grad=False).to(device)
        self.mask = torch.ceil(self.adj)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        A = self.mask + self.I
        # A = torch.clamp(A, 0.1)  # This is a trick.
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        A_hat = torch.unsqueeze(A_hat, 0)
        A_hat = torch.unsqueeze(A_hat, 0)
        attn = attn + A_hat
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, adj, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, adj, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, dim, adj, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.adj = adj
        self.transformer = Transformer(dim, adj, depth, heads, dim_head, mlp_dim, dropout)
        self.V = len(self.adj[0])
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.V+ 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128)
        )

    def forward(self, x):
        # x += self.pos_embedding[:, :self.V]
        x = self.dropout(x)
        x = self.transformer(x)

        return self.mlp_head(x)


class G2T(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int,
                 Q: torch.Tensor, A: torch.Tensor,model='normal'):  # Q2: torch.Tensor,A2: torch.Tensor,Q3: torch.Tensor,A3: torch.Tensor
        super(G2T, self).__init__()
        self.class_count = class_count  # 类别数
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q
        layers_count = 2
        self.WH = 0
        self.M = 2
        self.conv1_1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(),
                                     )
        self.conv1_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(),
                                     )
        self.conv2_1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(),
                                     )
        self.conv2_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(),
                                     )

        self.CNN_denoise1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise1.add_module('CNN_denoise_Conv' + str(i),
                                             nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                # self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                # self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise1.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_denoise2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(384))
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i),
                                             nn.Conv2d(384, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                # self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                # self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.ViT_Branch = ViT(dim=128, adj=A, depth=2, heads=10, mlp_dim=256, dim_head=64, dropout=0., emb_dropout=0.)
        self.Softmax_linear = nn.Sequential(nn.Linear(256, self.class_count))

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        noise = self.CNN_denoise1(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        x = self.conv1_1(noise)
        x = self.conv1_2(x)
        y = self.conv2_1(noise)
        y = self.conv2_2(y)
        z = torch.cat([x, y], dim=1)
        z = torch.cat([z, noise], dim=1)
        noise = self.CNN_denoise2(z)

        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 直连

        clean_x_flatten = clean_x.reshape([h * w, -1])  # reshape(N, -1)指定N行，列数自动确定一列
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分 encode

        ViT_result = self.ViT_Branch(torch.unsqueeze(superpixels_flatten, 0))
        ViT_result = torch.squeeze(ViT_result, 0)

        ViT_result = torch.matmul(self.Q, ViT_result)
        # Y = 0.95*ViT_result+0.05*clean_x_flatten
        Y = torch.cat([ViT_result, clean_x_flatten], -1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y