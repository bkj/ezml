#!/bin/bash

# install.sh

conda create -y -n ezml_env python=3.7
conda activate ezml_env

conda install -y -c pytorch pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0
pip install torchmeta==1.4.0

conda install -y pandas
conda install -y matplotlib

pip install git+https://github.com/bkj/rsub.git