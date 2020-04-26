import logging
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import glob
import re
import shutil

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adamax
from apex import amp

from .util import set_seed, load_and_cache_examples

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    a trainer modified from ```https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py```
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _post_step(self, args, outputs):
        pass

    def _forward(self, args, inputs, labels, masker, model, backprop=True):
        outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        if backprop:
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self._post_step(args, outputs)
            return loss
        else:
            return loss
        
    def _train_writer(self, logging_steps):
        self.tb_writer.add_scalar('lr', self.scheduler.get_lr()[0], self.global_step)
        self.tb_writer.add_scalar('loss', (self.tr_loss - self.logging_loss)/logging_steps, self.global_step)
        self.logging_loss = self.tr_loss

    def _to(self, args, tensor):
        if isinstance(tensor, torch.Tensor):
            output = tensor.to(args.device)
        elif isinstance(tensor[0], torch.Tensor):
            output = [_t.to(args.device) for _t in tensor]
        else:
            output = [(_t[0].to(args.device), _t[1].to(args.device)) for _t in tensor]
        return output

    def _post_training(self):
        pass

    def _train_batch(self, args, step, inputs, labels, masker, eval_dataset, eval_masker, model):
        inputs = self._to(args, inputs)        
        labels = self._to(args, labels)
        
        model.train()
        loss = self._forward(args, inputs, labels, masker, model, backprop=True)

        self.tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            self._post_training()
            self.global_step += 1
            
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and self.global_step % args.logging_steps == 0:
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = self.evaluate(args, eval_dataset, eval_masker, model)
                    for key, value in results.items():
                        self.tb_writer.add_scalar('eval_{}'.format(key), value, self.global_step)
                self._train_writer(args.logging_steps)

            if args.local_rank in [-1, 0] and args.save_steps > 0 and self.global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, self.global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                
                self.tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                self._rotate_checkpoints(args, checkpoint_prefix)

    
    def _train_epoch(self, args, epoch_id, epoch_iterator, train_masker, eval_dataset, eval_masker, model):
        for step, batch in enumerate(epoch_iterator):
            # consume this example also consume randomness.
            inputs, labels = train_masker.mask_tokens(batch, args.mlm_probability) if args.mlm else (batch, batch)
            if epoch_id < self.epochs_trained:
                logger.info("Continue training: skip epoch %d", epoch_id)
                break
            if self.steps_trained_in_current_epoch > 0:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    self.steps_trained_in_current_epoch -= 1
                continue
            
            if step % 5000 == 0:
                input_ids = inputs[-1].tolist()
                logger.info("domain %s", input_ids[0])
                logger.info("one exmaple is like %s", " ".join(train_masker.tokenizer.convert_ids_to_tokens(input_ids[1:])))
                logger.info("one label is like %s", str(labels[-1].tolist()))

            self._train_batch(args, step, inputs, labels, train_masker, eval_dataset, eval_masker, model)
            if args.max_steps > 0 and self.global_step > args.max_steps:
                epoch_iterator.close()
                break
    
    def _build_train_data_loader(self, args, train_dataset, model):
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        return train_sampler, train_dataloader

    def _init_logging(self):
        self.tr_loss, self.logging_loss = 0.0, 0.0

    def train(self, args, train_dataset, train_masker, eval_dataset, eval_masker, model):
                
        """ Train the model """
        if args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(args.output_dir)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler, train_dataloader = self._build_train_data_loader(args, train_dataset, model)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        self.t_total = t_total

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
        
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = args.warmup_steps, num_training_steps=t_total)
        
        if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_sampler))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epochs_trained = 0
        self.steps_trained_in_current_epoch = 0
        if args.model_name_or_path and os.path.exists(args.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                self.global_step = int(checkpoint_suffix)
                self.epochs_trained = self.global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                self.steps_trained_in_current_epoch = self.global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in epoch %d", self.steps_trained_in_current_epoch, self.epochs_trained)
            except ValueError:
                logger.info("  Starting fine-tuning.")
                
        self._init_logging()
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
        for epoch_id, _ in enumerate(train_iterator):
            # seek to the current epoch.
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            self._train_epoch(args, epoch_id, epoch_iterator, train_masker, eval_dataset, eval_masker, model)
            
            if args.max_steps > 0 and self.global_step > args.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            self.tb_writer.close()

        return self.global_step, self.tr_loss #/ self.global_step

    def _eval(self, args, eval_dataloader, eval_masker, model):
        eval_loss = 0.0
        nb_eval_steps = 0

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = eval_masker.mask_tokens(batch, 0.15) if args.mlm else (batch, batch)
            inputs = self._to(args, inputs)
            labels = self._to(args, labels)

            with torch.no_grad():
                lm_loss = self._forward(args, inputs, labels, eval_masker, model, backprop=False)
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        
        perplexity = torch.exp(torch.tensor(eval_loss))

        return {
            "perplexity": perplexity, 
            "loss": eval_loss, 
        }


    def evaluate(self, args, eval_dataset, eval_masker, model, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = args.output_dir

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        model.eval()

        result = self._eval(args, eval_dataloader, eval_masker, model)
        
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return result

    def _rotate_checkpoints(self, args, checkpoint_prefix, use_mtime=False):
        if not args.save_total_limit:
            return
        if args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
        if len(glob_checkpoints) <= args.save_total_limit:
            return

        ordering_and_checkpoint_path = []
        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)
