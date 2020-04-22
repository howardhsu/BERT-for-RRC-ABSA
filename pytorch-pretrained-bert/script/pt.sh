#!/bin/bash

domain=$1
dp=$2
steps=$3

if ! [ -z $4 ] ; then
    export CUDA_VISIBLE_DEVICES=$4
    echo "using cuda"$CUDA_VISIBLE_DEVICES
fi


#. ~/anaconda2/etc/profile.d/conda.sh

#conda activate p3-torch10

BERT="bert-base"

if ! [ -e ../domain_corpus/${domain}/data.npz ] ; then
    python ../src/gen_pt_review.py \
        --input_file ../domain_corpus/raw/${domain}.txt \
        --output_file ../domain_corpus/${domain}/data.npz \
        --bert-model $BERT \
        --max_seq_length=320 \
        --max_predictions_per_seq=40 \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=$dp > ../domain_corpus/${domain}/data.log 2>&1
fi


if ! [ -e ../squad/data.npz ] ; then
    python ../src/gen_pt_squad.py \
        --input_dir ../squad \
        --output_dir ../squad \
        --bert-model $BERT \
        --max_seq_length=320 \
        --seed=12345 > ../squad/data.log 2>&1
fi

OUT_DIR="../pt_model/${domain}_pt"
mkdir -p $OUT_DIR

python ../src/run_pt.py \
    --bert_model $BERT \
    --review_data_dir ../domain_corpus/${domain} \
    --squad_data_dir ../squad/ \
    --output_dir $OUT_DIR \
    --train_batch_size 16 \
    --do_train \
    --num_train_steps=$steps \
    --gradient_accumulation_steps=2 \
    --fp16 --loss_scale 2 \
    --save_checkpoints_steps 10000 > $OUT_DIR/train.log 2>&1

cp ../pt_model/$BERT/vocab.txt ./$OUT_DIR
cp ../pt_model/$BERT/bert_config.json ./$OUT_DIR
