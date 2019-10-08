#!/bin/bash

# run.sh

mkdir -p results/{omniglot,miniimagenet}

# --
# Run 20way-1shot omniglot

python main.py --dataset omniglot --ways 20 --shots 1 | tee results/omniglot/20way-1shot.jl

# --
# Run 5way-1shot miniimagenet

python main.py --dataset miniimagenet --ways 5 --shots 1 | tee results/miniimagenet/5way-1shot.jl
