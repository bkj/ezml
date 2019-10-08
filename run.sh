#!/bin/bash

# run.sh

# --
# Run 20way-1shot omniglot

mkdir -p results/omniglot

python main.py --dataset omniglot --ways 20 --shots 1 | tee results/omniglot/20way-1shot.jl

# --
# Run 5way-1shot miniimagenet

mkdir -p results/miniimagenet

python main.py --dataset miniimagenet --ways 5 --shots 1 | tee results/miniimagenet/5way-1shot.jl
