#!/bin/bash

# run.sh

# --
# Run basic implementation

mkdir -p results

CUDA_VISIBLE_DEVICES=0 python main.py | tee results/og-results.jl

CUDA_VISIBLE_DEVICES=1 python main.py --dataset miniimagenet | tee results/mi-results.jl

# --
# Run more optimized, but less obvious implementation

CUDA_VISIBLE_DEVICES=4 python main-opt.py | tee results/og-opt-results.jl