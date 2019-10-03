#!/bin/bash

# install.sh

conda create -y -n ezml_env python=3.7
conda activate ezml_env

conda install -y -c pytorch pytorch==1.2.0 torchvision

pip install torchmeta==1.1.1

# conda install -y numpy 
# conda install -y scipy
# conda install -y matplotlib
# conda install -y -c conda-forge opencv
# conda install -y -c conda-forge pbzip2 
# conda install -y -c conda-forge pydrive
# conda install -y pillow 
# conda install -y tqdm



# pip install git+https://github.com/bkj/rsub
# pip install matplotlib
