# ezml

Simple MAML-like metalearning baseline.

TL;DR: MAML adjusts all parameters for each task, EZML only updates the weights in the classifier head.

EZML attains similar or superior performance to MAML on Omniglot in 1s-10s of minutes; MAML takes 1-10s of hours to train.  This is primarily due to the fact that we only have to pass each batch through the featurizer once, no matter how many inner-loop update steps we use.

EZML is very similar in spirit to [ANIL](https://arxiv.org/abs/1909.09157).

## Installation

See `./install.sh`

## Usage

See `./run.sh`

## Notes

- The weights in the classifier head are tied, and no bias is learned -- this makes the network invariant to permutations in the order of classes in a task.

## Optimizations

- The dataloader adds ~ 25% overhead to training -- this could likely be reduced