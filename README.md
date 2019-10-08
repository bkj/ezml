# ezml

Simple MAML-like metalearning baseline.

TL;DR: MAML adjusts all parameters for each task, EZML only updates the weights in the classifier head.

EZML attains similar or superior performance to MAML on Omniglot in 1s-10s of minutes; MAML takes 1-10s of hours to train.  This is primarily due to the fact that we only have to pass each batch through the featurizer once, no matter how many inner-loop update steps we use.

EZML is very similar in spirit to [ANIL](https://arxiv.org/abs/1909.09157).

## Installation

See `./install.sh`

## Usage

See `./run.sh`

## Performance

Omniglot 20-way 1-shot:
```
{"batch_idx": 50000, "batch_acc": 0.975, "valid_acc": 0.959125, "test_acc": 0.953, "elapsed": 4062.5901415348053}
```

Mini-ImageNet 5-way 1-shot:
```
{"batch_idx": 50000, "batch_acc": 0.75, "valid_acc": 0.4785, "test_acc": 0.48175, "elapsed": 3896.5144040584564}
```

## Notes

- The weights in the classifier head are tied, and no bias is learned -- this makes the network invariant to permutations in the order of classes in a task.

## Optimizations

- The dataloader adds ~ 25% overhead to training -- this could likely be reduced