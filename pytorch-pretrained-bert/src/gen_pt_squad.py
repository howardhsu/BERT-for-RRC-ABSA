# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
import argparse
import random
import json
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer

import squad_data_utils as data_utils
import modelconfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def gen(args):

    tokenizer = BertTokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model] )

    train_examples = data_utils.read_squad_examples(os.path.join(args.input_dir, "train.json"), is_training=True)
    
    train_features = data_utils.convert_examples_to_features(
        train_examples, tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, is_training=True)
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))

    input_ids_np = np.array([f.input_ids for f in train_features], dtype=np.int16)
    segment_ids_np = np.array([f.segment_ids for f in train_features], dtype=np.int16)
    input_mask_np = np.array([f.input_mask for f in train_features], dtype=np.int16)
    start_positions_np = np.array([f.start_position for f in train_features], dtype=np.int16)
    end_positions_np = np.array([f.end_position for f in train_features], dtype=np.int16)

    np.savez_compressed(os.path.join(args.output_dir, "data.npz"), 
                        input_ids=input_ids_np, 
                        segment_ids = segment_ids_np, 
                        input_mask = input_mask_np, 
                        start_positions = start_positions_np, 
                        end_positions = end_positions_np)
    
    #>>>>> validation
    valid_examples=data_utils.read_squad_examples(os.path.join(args.input_dir,"dev.json"), is_training=True)

    valid_features = data_utils.convert_examples_to_features(
        valid_examples, tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, is_training=True)
    
    logger.info("  Num orig examples = %d", len(valid_examples))
    logger.info("  Num split examples = %d", len(valid_features))

    valid_input_ids_np = np.array([f.input_ids for f in valid_features], dtype=np.int16)
    valid_segment_ids_np = np.array([f.segment_ids for f in valid_features], dtype=np.int16)
    valid_input_mask_np = np.array([f.input_mask for f in valid_features], dtype=np.int16)
    valid_start_positions_np = np.array([f.start_position for f in valid_features], dtype=np.int16)
    valid_end_positions_np = np.array([f.end_position for f in valid_features], dtype=np.int16)
    
    np.savez_compressed(os.path.join(args.output_dir, "dev.npz"), 
                        input_ids=valid_input_ids_np, 
                        segment_ids = valid_segment_ids_np, 
                        input_mask = valid_input_mask_np, 
                        start_positions = valid_start_positions_np, 
                        end_positions = valid_end_positions_np)
    #<<<<< end of validation declaration

def main():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert-model", default='bert-base', type=str)

    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=320,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    
    parser.add_argument('--doc_stride',
                        type=int,
                        default=128)
    
    parser.add_argument('--max_query_length',
                        type=int,
                        default=30)
    
    parser.add_argument('--max_answer_length',
                        type=int,
                        default=30)
    
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    gen(args)
    
if __name__=="__main__":
    main()