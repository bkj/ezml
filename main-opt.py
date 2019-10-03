#!/usr/bin/env python

"""
    main-opt.py
    
    
"""

import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from time import time

import torch
from torch import nn
from torch.nn import functional as F

from torchmeta.datasets import helpers as torchmeta_datasets_helpers
from torchmeta.utils.data import BatchMetaDataLoader

from model import EZML, SimpleEncoder

torch.cudnn.deterministic = True

# --
# Helpers

def set_seeds(seed):
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed + 1)
    _ = np.random.seed(seed + 2)


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
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx == max_batches:
            break
            
        batch        = dict2cuda(batch)
        x_sup, y_sup = batch['train']
        x_tar, y_tar = batch['test']
        
        logit_tar = model(x_sup, y_sup, x_tar)
        logit_tar = logit_tar.view(-1, args.ways)
        pred_tar  = logit_tar.argmax(dim=-1)
        
        y_tar     = y_tar.view(-1)
        
        total   += int(pred_tar.shape[0])
        correct += int((pred_tar.argmax(dim=-1) == y_tar).sum())
    
    return total / correct

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str, default='omniglot')
    parser.add_argument('--ways',       type=int, default=20)
    parser.add_argument('--shots',      type=int, default=1)
    
    parser.add_argument('--inner-steps', type=int, default=1)
    
    parser.add_argument('--batch-size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-3)
    
    parser.add_argument('--valid-interval', type=int, default=512)
    parser.add_argument('--valid-batches',  type=int, default=100)
    parser.add_argument('--valid-shots',    type=int, default=1)
    parser.add_argument('--max-iters',      type=int, default=2 ** 14)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

args = parse_args()
set_seeds(args.seed)

print(json.dumps(vars(args)))

# --
# IO

dataset_in_channels = {
    "omniglot"     : 1,
    "miniimagenet" : 3,
}
in_channels = dataset_in_channels[args.dataset]

dataset_cls = getattr(torchmeta_datasets_helpers, args.dataset)

dataset_kwargs    = {
    "folder"   : "./data", 
    "ways"     : args.ways, 
    "shots"    : args.shots, 
    "shuffle"  : True, 
    "download" : True,
}

train_dataset  = dataset_cls(meta_split='train', **dataset_kwargs)
valid_dataset  = dataset_cls(meta_split='val',  test_shots=args.valid_shots, **dataset_kwargs)
test_dataset   = dataset_cls(meta_split='test', test_shots=args.valid_shots, **dataset_kwargs)

dataloader_kwargs = {
    "batch_size"  : args.batch_size, 
    "num_workers" : 4, 
    "shuffle"     : True, 
    "pin_memory"  : True,
}
train_dataloader = BatchMetaDataLoader(train_dataset, **dataloader_kwargs)
valid_dataloader = BatchMetaDataLoader(valid_dataset, **dataloader_kwargs)
test_dataloader  = BatchMetaDataLoader(test_dataset, **dataloader_kwargs)

# --
# Define model

model = EZML(
    encoder=SimpleEncoder(in_channels=in_channels),
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
    
    logit_tar = model(x_sup, y_sup, x_tar)
    logit_tar = logit_tar.view(-1, args.ways)
    pred_tar  = logit_tar.argmax(dim=-1)
    
    y_tar     = y_tar.view(-1)
    
    loss = F.cross_entropy(logit_tar, y_tar)
    loss.backward()
    
    batch_total   = int(logit_tar.shape[0])
    batch_correct = int((pred_tar == y_tar).sum())
    
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
    
    if batch_idx == args.max_iters:
        break
