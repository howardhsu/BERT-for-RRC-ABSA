#!/bin/bash

task=$1
bert=$2
domain=$3
run_dir=$4
runs=$5

#. ~/anaconda2/etc/profile.d/conda.sh

#conda activate p3-torch10


if ! [ -z $6 ] ; then
    export CUDA_VISIBLE_DEVICES=$6
    echo "using cuda"$CUDA_VISIBLE_DEVICES
fi


DATA_DIR="../"$task/$domain

for run in `seq 1 1 $runs`
do
    OUTPUT_DIR="../run/"$run_dir/$domain/$run

    mkdir -p $OUTPUT_DIR
    if ! [ -e $OUTPUT_DIR/"valid.json" ] ; then
        python ../src/run_$task.py \
            --bert_model $bert --do_train --do_valid \
            --max_seq_length 100 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 4 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/train_log.txt 2>&1
    fi

    if ! [ -e $OUTPUT_DIR/"predictions.json" ] ; then 
        python ../src/run_$task.py \
            --bert_model $bert --do_eval --max_seq_length 100 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/test_log.txt 2>&1
    fi
    if [ -e $OUTPUT_DIR/"predictions.json" ] && [ -e $OUTPUT_DIR/model.pt ] ; then
        rm $OUTPUT_DIR/model.pt
    fi
done
