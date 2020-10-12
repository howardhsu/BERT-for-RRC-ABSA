#!/bin/bash

GPU=0

. ~/anaconda3/etc/profile.d/conda.sh
conda activate p3-torch13

export CUDA_VISIBLE_DEVICES=${GPU}

export PYTHONPATH="${PYTHONPATH}:./"

rm -rf ./aruns/*

python src/analyze.py
