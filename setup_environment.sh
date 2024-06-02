#!/bin/bash

conda env create -f environment.yaml
conda activate venv
pip install torch==1.11.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmsegmentation==0.30.0
mim install mmcv-full==1.6.2