#!/usr/bin/env python

"""
    helpers.py
"""

import torch
import numpy as np

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
