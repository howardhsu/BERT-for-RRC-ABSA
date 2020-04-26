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
import random
import json

import numpy as np
import torch
import sklearn
import sklearn.metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import (
    BertConfig, BertTokenizer,
)


import logging
logger = logging.getLogger(__name__)

from . import absa_data_util as data_util

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
}


class Trainer(object):
    def _convert_examples_to_features(self, config, examples, tokenizer, max_seq_length, label_list):
        raise NotImplementedError
        
    def _config_task(self, config):
        raise NotImplementedError
    
    def train(self, args):
        processor, model_class = self._config_task(args)
        
        label_list = processor.get_labels()
        num_labels = len(label_list)

        logger.info("***** Processor *****")
        logger.info("  label_list (%d) = %s", len(label_list), str(label_list))

        config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        
        config = config_class.from_pretrained(args.model_name_or_path, num_labels = num_labels, finetuning_task = args.task)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case = args.do_lower_case)
        
        model = model_class.from_pretrained(args.model_name_or_path, config = config)    
        
        model.to(args.device)

        train_examples = processor.get_train_examples()
        
        train_features = self._convert_examples_to_features(args, train_examples, tokenizer, args.max_seq_length, label_list, args.model_type)

        train_dataset = data_util.build_dataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size=args.train_batch_size)

        #>>>>> validation        
        valid_examples = processor.get_dev_examples()
        valid_features = self._convert_examples_to_features(args, valid_examples, tokenizer, args.max_seq_length, label_list, args.model_type)
        
        valid_dataset = data_util.build_dataset(valid_features)
        
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler = valid_sampler, batch_size = args.train_batch_size)    

        valid_losses = []
        #<<<<< end of validation declaration
        
        # Prepare optimizer
        
        t_total = len(train_dataloader) * args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = t_total)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
                
        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)
            
        global_step = 0
        best_valid_loss = float('inf')
        model.zero_grad()
        model.train()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(args.device) for t in batch)

                inputs = {"input_ids": batch[0], 
                          "attention_mask": batch[1],
                          "token_type_ids": batch[2], 
                          "labels": batch[3]}
                if args.model_type == 'distilbert': 
                    inputs.pop("token_type_ids", None)
                outputs = model(**inputs)
                loss = outputs[0]
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                                        
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()
                global_step += 1
                
            model.eval()
            with torch.no_grad():
                losses=[]
                valid_size=0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.to(args.device) for t in batch) # multi-gpu does scattering it-self
                    inputs = {"input_ids": batch[0], 
                              "attention_mask": batch[1],
                              "token_type_ids": batch[2], 
                              "labels": batch[3]}
                    if args.model_type == 'distilbert': 
                        inputs.pop("token_type_ids", None)
                    outputs = model(**inputs)
                    tmp_valid_loss, logits = outputs[:2]
                    losses.append(tmp_valid_loss.detach().mean().item() * batch[0].size(0) )
                    valid_size += batch[0].size(0)
                valid_loss=sum(losses)/valid_size
                logger.info("validation loss: %f", valid_loss)
                valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                torch.save(model, os.path.join(args.output_dir, "model.pt") )
                best_valid_loss=valid_loss

            model.train()

        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)

    def test(self, args, test_file):
        processor = getattr(data_util, args.task.upper() + "Processor")(args)
        label_list = processor.get_labels()

        config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case = args.do_lower_case)
        
        eval_examples = processor.get_test_examples(test_file)
        
        eval_features = self._convert_examples_to_features(args, eval_examples, tokenizer, args.max_seq_length, label_list, args.model_type)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataset = data_util.build_dataset(eval_features)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model = torch.load(os.path.join(args.output_dir, "model.pt") )
        model.to(args.device)
        model.eval()
        
        self._predict(args, model, eval_examples, eval_dataloader, label_list)
        
        
    def _predict(self, args, model, eval_examples, eval_dataloader, labels):
        raise NotImplementedError