#!/bin/bash

# run.sh

# --
# Run basic implementation

mkdir -p results

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset omniglot --ways 20 --shots 1 | tee results/og-results.jl

CUDA_VISIBLE_DEVICES=1 python main.py \
    --dataset miniimagenet --ways 5 --shots 1 | tee results/mi-results.jl

# --
# Run more optimized, but less obvious implementation

CUDA_VISIBLE_DEVICES=3 python main-opt.py --ways 5 --shots 1 | tee results/og-opt-results.jl