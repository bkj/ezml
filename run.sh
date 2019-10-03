#!/bin/bash

# run.sh

# --
# Run

mkdir -p results

CUDA_VISIBLE_DEVICES=5 python main.py | tee results/og-results.jl

python main.py --dataset miniimagenet | tee results/mi-results.jl