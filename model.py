#!/usr/bin/env python

"""
    model.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

# >>

class ResBlock(nn.Module):
    def __init__(self, ni, no, stride, dropout=0):
        super().__init__()
        
        self.conv0    = nn.Conv2d(ni, no, 3, stride, padding=1, bias=False)
        self.bn0      = nn.BatchNorm2d(no)
        self.dropout0 = nn.Dropout2d(dropout) if dropout > 0 else lambda x: x
        
        self.conv1    = nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(no)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else lambda x: x
        
        self.conv2 = nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(no)
        
        if stride == 2 or ni != no:
            self.shortcut = nn.Conv2d(ni, no, 1, stride=1, padding=0)

    def forward(self, x):
        
        y = self.conv0(x)
        y = self.bn0(y)
        y = F.relu(y, inplace=True)
        y = self.dropout0(y)
        
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)
        y = self.dropout1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = y + self.shortcut(x) # Shortcut
        y = F.relu(y, inplace=True)
        
        return y


class ResNet12(nn.Module):
    def __init__(self, in_channels, width=1, dropout=0.5):
        super().__init__()
        self.out_channels = 512
        assert(width == 1) # Comment for different variants of this model
        self.widths = [x * int(width) for x in [64, 128, 256]]
        self.widths.append(self.out_channels * width)
        self.bn_out = nn.BatchNorm1d(self.out_channels)
        
        for i in range(len(self.widths)):
            setattr(self, "group_%d" %i, ResBlock(in_channels, self.widths[i], 1, dropout))
            in_channels = self.widths[i]
    
    def up_to_embedding(self, x):
        for i in range(len(self.widths)):
            x = getattr(self, "group_%d" % i)(x)
            x = F.max_pool2d(x, 3, 2, 1)
        
        return x

    def forward(self, x):
        *args, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.up_to_embedding(x)
        x = x.mean(3).mean(2)
        # return F.relu(self.bn_out(x), True)
        return x


# <<

class Conv4Block(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[64, 64, 64, 64]):
        super().__init__()
        
        self.layers = nn.Sequential(
            Conv4Block(in_channels, hidden_dims[0]),
            Conv4Block(hidden_dims[0], hidden_dims[1]),
            Conv4Block(hidden_dims[1], hidden_dims[2]),
            Conv4Block(hidden_dims[2], hidden_dims[3]), 
        )
        
        self.out_channels = hidden_dims[-1]
    
    def forward(self, x):
        batch_of_tasks = len(x.shape) == 5
        if batch_of_tasks:
            b, s, c, w, h = x.shape
            x = x.view(b * s, c, w, h)
        
        x = self.layers(x)
        x = x.mean(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1)
        
        if batch_of_tasks:
            x = x.view(b, s, -1)
        
        return x


class FastHead(nn.Module):
    def __init__(self, in_channels, out_channels, n_steps, lr=0.1):
        super().__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_steps      = n_steps
        
        self.lr      = nn.Parameter(torch.FloatTensor([lr]))
        self.scale   = nn.Parameter(torch.FloatTensor([1]))
        self.weights = nn.Parameter(torch.ones(in_channels, 1))
    
    def forward(self, x, y):
        batch_of_tasks = len(x.shape) == 3
        if batch_of_tasks:
            b, s, c = x.shape
        
        w = (self.scale * self.weights).clone()
        
        if batch_of_tasks:
            w = w.unsqueeze(0).repeat((b, 1, self.out_channels))
        else:
            w = w.repeat((1, self.out_channels))
        
        for _ in range(self.n_steps):
            out  = x @ w
            loss = F.cross_entropy(out.view(-1, self.out_channels), y.view(-1), reduction='sum')
            grad = torch.autograd.grad(loss, [w], create_graph=True)[0]
            w    = w - self.lr * grad
        
        return w

# # >>
# def global_consistency(weights, alpha, norm_prop):
#     n          = weights.shape[1]
#     identity   = torch.eye(n, dtype=weights.dtype, device=weights.device)
#     isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
    
#     S          = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
    
#     propagator = identity - alpha * S
#     propagator = torch.inverse(propagator[None, ...])[0]
    
#     if norm_prop:
#         propagator = F.normalize(propagator, p=1, dim=-1)
    
#     return propagator

# def get_similarity_matrix(x, rbf_scale):
#     b, c    = x.shape
    
#     sq_dist = ((x.view(b, 1, c) - x.view(1, b, c))**2).sum(-1) / np.sqrt(c)
    
#     mask    = sq_dist != 0
#     sq_dist = sq_dist / sq_dist[mask].std()
#     weights = torch.exp(-sq_dist * rbf_scale)
    
#     mask    = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)
#     weights = weights * (~mask).float()
    
#     return weights

# def embedding_propagation(x, alpha=0.5, rbf_scale=1, norm_prop=False):
#     weights    = get_similarity_matrix(x, rbf_scale=rbf_scale)
#     propagator = global_consistency(weights, alpha=alpha, norm_prop=norm_prop)
#     return propagator @ x
# # <<

class EZML(nn.Module):
    def __init__(self, encoder, n_classes, inner_steps=1):
        super().__init__()
        
        self.encoder    = encoder
        self.model_head = FastHead(encoder.out_channels, n_classes, n_steps=inner_steps)
    
    def forward(self, x_sup, y_sup, x_tar):
        enc_sup = self.encoder(x_sup)
        enc_tar = self.encoder(x_tar)
        
        # # Embedding propagation
        # enc     = embedding_propagation(torch.cat([enc_sup, enc_tar], dim=0))
        # enc_sup = enc[:x_sup.shape[0]]
        # enc_tar = enc[x_sup.shape[0]:]
        
        fast_w = self.model_head(enc_sup, y_sup)
        return enc_tar @ fast_w


