#!/bin/bash

# this script fine-tune a task for 10 times.

CONFIG=$1
GPU=$2
RUNS=$3

. ~/anaconda3/etc/profile.d/conda.sh
conda activate p3-torch13

export CUDA_VISIBLE_DEVICES=${GPU}

export PYTHONPATH="${PYTHONPATH}:./"

for RUN in `seq 1 1 $RUNS`
do
    python src/runner.py \
        --config ${CONFIG} \
        --seed ${RUN}
done

python src/eval.py --config ${CONFIG}
