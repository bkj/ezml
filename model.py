#!/usr/bin/env python

"""
    model.py
"""

import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Sequential):
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
            Block(in_channels, hidden_dims[0]),
            Block(hidden_dims[0], hidden_dims[1]),
            Block(hidden_dims[1], hidden_dims[2]),
            Block(hidden_dims[2], hidden_dims[3]), 
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
        self.lr           = lr
        
        self.scale   = nn.Parameter(torch.FloatTensor([1]))
        self.weights = nn.Parameter(torch.ones(in_channels, 1))
    
    def forward(self, x, y):
        batch_of_tasks = len(x.shape) == 3
        if batch_of_tasks:
            b, s, c = x.shape
        
        w = (self.scale * self.weights)
        
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


class EZML(nn.Module):
    def __init__(self, encoder, n_classes, inner_steps=1):
        super().__init__()
        
        self.encoder    = encoder
        self.model_head = FastHead(encoder.out_channels, n_classes, n_steps=inner_steps)
    
    def forward(self, x_sup, y_sup, x_tar):
        enc_sup = self.encoder(x_sup)
        fast_w  = self.model_head(enc_sup, y_sup)
        return self.encoder(x_tar) @ fast_w
