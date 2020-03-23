#!/usr/bin/env python

"""
    plot-results.py
"""

import json
import argparse
import pandas as pd

from rsub import *
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='results/mi-results.jl')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    x = open(args.inpath).read().splitlines()
    config, x = json.loads(x[0]), x[1:]
    
    columns = ('batch_idx', 'elapsed', 'batch_acc', 'valid_acc', 'test_acc')
    df = pd.DataFrame([json.loads(xx) for xx in x], columns=columns)

    df['ways']  = config['ways']
    df['shots'] = config['shots']
    
    _ = plt.plot(df.batch_acc, label='batch_acc', c='black', alpha=0.25)
    _ = plt.plot(df.valid_acc, label='valid_acc', c='red', alpha=0.5)
    _ = plt.plot(df.test_acc,  label='test_acc', c='blue', alpha=0.5)
    _ = plt.legend()
    _ = plt.grid('both')
    _ = plt.title(args.inpath)
    _ = plt.xlabel('batch')
    _ = plt.ylabel('accuracy')
    show_plot()
    
    print(df.tail())