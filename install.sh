#!/bin/bash

# install.sh

conda create -y -n ezml_env python=3.7
conda activate ezml_env

conda install -y -c pytorch pytorch==1.2.0 torchvision
pip install torchmeta==1.1.1

conda install -y pandas
conda install -y matplotlib

pip install git+https://github.com/bkj/rsub.git