#!/usr/bin/env python

"""
    main.py
"""

import sys
import json
import argparse
from time import time

import torch
from torch import nn
from torch.nn import functional as F

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

from model import EZML, SimpleEncoder

# --
# Helpers

def dict2cuda(x, device='cuda:0'):
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            x[k] = v.to(device)
        elif isinstance(v, list):
            x[k] = [vv.to(device) for vv in v]
    
    return x

def do_eval(model, dataloader, max_batches):
    assert not model.training
    
    total, correct = 0, 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == max_batches:
            break
        
        batch        = dict2cuda(batch)
        x_sup, y_sup = batch['train']
        x_tar, y_tar = batch['test']
        
        for xx_sup, yy_sup, xx_tar, yy_tar in zip(x_sup, y_sup, x_tar, y_tar):
            pred_tar = model(xx_sup, yy_sup, xx_tar)
            
            total   += int(pred_tar.shape[0])
            correct += int((pred_tar.argmax(dim=-1) == yy_tar).sum())
    
    return batch_correct / batch_total

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ways',       type=int, default=20)
    parser.add_argument('--shots',      type=int, default=1)
    
    parser.add_argument('--inner-steps', type=int, default=1)
    
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr',         type=float, default=1e-3)
    
    parser.add_argument('--valid-interval', type=int, default=100)
    parser.add_argument('--valid-batches',  type=int, default=100)
    
    return parser.parse_args()

args = parse_args()

# --
# IO

dataset_kwargs = {"ways" : args.ways, "shots"  : args.shots}
train_dataset  = omniglot("./data", meta_split='train', **dataset_kwargs)
valid_dataset  = omniglot("./data", meta_split='val', **dataset_kwargs)
test_dataset   = omniglot("./data", meta_split='test', **dataset_kwargs)

train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
valid_dataloader = BatchMetaDataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4)
test_dataloader  = BatchMetaDataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

# --
# Define model

model = EZML(
    encoder=SimpleEncoder(), 
    n_classes=args.ways, 
    inner_steps=args.inner_steps
).to('cuda:0')

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# --
# Run

train_hist = []
t_start    = time()
valid_acc  = 0
test_acc   = 0
for batch_idx, batch in enumerate(train_dataloader):
    
    # --
    # Train
    
    opt.zero_grad()
    
    batch        = dict2cuda(batch)
    x_sup, y_sup = batch['train']
    x_tar, y_tar = batch['test']
    
    batch_total, batch_correct = 0, 0
    for xx_sup, yy_sup, xx_tar, yy_tar in zip(x_sup, y_sup, x_tar, y_tar):
        pred_tar = model(xx_sup, yy_sup, xx_tar)
        
        loss = F.cross_entropy(pred_tar, yy_tar)
        loss.backward()
        
        batch_total   += int(pred_tar.shape[0])
        batch_correct += int((pred_tar.argmax(dim=-1) == yy_tar).sum())
    
    opt.step()
    
    batch_acc = batch_correct / batch_total
    train_hist.append({
        "batch_idx" : batch_idx,
        "batch_acc" : batch_acc,
        "valid_acc" : valid_acc,
        "test_acc"  : test_acc,
        "elapsed"   : time() - t_start,
    })
    print(json.dumps(train_hist[-1]))
    sys.stdout.flush()
    
    # --
    # Eval
    
    if (batch_idx > 0) and (batch_idx % args.valid_interval == 0):
        _ = model.eval()
        valid_acc = do_eval(model, valid_dataloader, max_batches=args.valid_batches)
        test_acc  = do_eval(model, test_dataloader, max_batches=args.valid_batches)
        _ = model.train()