import logging
import os
import pickle
import random
import json

import numpy as np
import torch
import glob
import re
import shutil

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Sampler, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from collections import defaultdict, deque
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp

from .util import set_seed, load_and_cache_examples
from .trainer import Trainer

logger = logging.getLogger(__name__)


class MetaSampler(nn.Module):
    def __init__(self, num_domains):
        super().__init__()
        self.domain_logits = nn.Parameter(torch.Tensor(num_domains).normal_(mean=0.0, std=0.02))
        self.target_loss = None
        self.domain_losses = defaultdict(lambda : deque())
        self.domain_cnts = defaultdict(int)
        
    def forward(self):
        probs = torch.softmax(self.domain_logits, 0)
        self.sampler = Categorical(probs)
        self.log_probs = []
        self.losses = []
        self.target_masks = []
        self.sampled_ex_masks = []


class DomainSampler(Sampler):
    def __init__(self, dataset, domain_sampler):
        self.domain_data_dict = defaultdict(list)
        for ex_idx, ex in enumerate(dataset):
            self.domain_data_dict[ex[0].item()].append(ex_idx)
        logger.info("total number of target domain examples: %s", str(len(self.domain_data_dict[0])))
        for domain_idx in self.domain_data_dict:
            random.shuffle(self.domain_data_dict[domain_idx])
        self.domain_sampler = domain_sampler
        self.domain_offsets = {key: 0 for key in self.domain_data_dict}
        self.epsilon = 1.

    def _epsilon_fn(self, global_step, t_total):
        if (float(global_step) / t_total) > 0.1: # warm up.
            step = (self.epsilon - 0.5) / (t_total * 0.3)
            self.epsilon -= step
            self.epsilon = max(self.epsilon, 0.5)

    def __iter__(self):
        ex_cnt = 0
        while ex_cnt < len(self.domain_data_dict[0]):
            if random.random() < self.epsilon:
                domain_idx = 0
                self.domain_sampler.sampled_ex_masks.append(False)
            else:
                domain_idx = self.domain_sampler.sampler.sample()
                while len(self.domain_data_dict[domain_idx.item()]) == 0:
                    domain_idx = self.domain_sampler.sampler.sample()
                self.domain_sampler.log_probs.append(self.domain_sampler.sampler.log_prob(domain_idx))
                domain_idx = domain_idx.item()
                self.domain_sampler.sampled_ex_masks.append(True)
            yield self.domain_data_dict[domain_idx][self.domain_offsets[domain_idx]]
            ex_cnt += 1
            self.domain_offsets[domain_idx] += 1
            if self.domain_offsets[domain_idx] >= len(self.domain_data_dict[domain_idx]):
                # reset the offset.
                random.shuffle(self.domain_data_dict[domain_idx])
                self.domain_offsets[domain_idx] = 0
            
    def __len__(self):
        return len(self.domain_data_dict[0])


class MetaTrainer(Trainer):
    def _forward(self, args, inputs, labels, masker, model, backprop=True):
        outputs = model(inputs, masked_lm_labels=labels, target_loss=self.meta_model.target_loss) if args.mlm else model(inputs, labels=labels)
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
            self._post_step(outputs)
            return loss
        else:
            return loss
    
    def _build_train_data_loader(self, args, train_dataset, model):
        with open("data/pt/domain_tag/doi_domain.json") as f:
            domain_dict = json.load(f)
        self.id_to_domain = {domain_dict[domain]: domain for domain in domain_dict}
        
        self.meta_model = MetaSampler(len(domain_dict)).to(args.device)
        self.meta_optimizer = AdamW(self.meta_model.parameters())
        self.meta_model()

        self.train_sampler = DomainSampler(train_dataset, self.meta_model)
        train_dataloader = DataLoader(train_dataset, sampler=self.train_sampler, batch_size=args.train_batch_size)
        
        return self.train_sampler, train_dataloader
    
    def _post_step(self, outputs):
        self.meta_model.losses.append(outputs[1])
        self.meta_model.target_masks.append(outputs[2])
        for ex_idx, domain_id in enumerate(outputs[3]):
            domain_id = domain_id.item()
            buffer = self.meta_model.domain_losses[domain_id]
            if len(buffer) >= 128:
                buffer.popleft()
            buffer.append(float(outputs[1][ex_idx]))
            self.meta_model.domain_cnts[domain_id] += 1

    def _post_training(self):
        if self.global_step % 16 == 1:
            losses = torch.cat(self.meta_model.losses, dim=0)
            target_masks = torch.cat(self.meta_model.target_masks, dim=0)

            target_loss = torch.mean(losses[target_masks])
            self.meta_model.target_loss = target_loss

            if len(self.meta_model.log_probs) > 0:
                sampled_ex_masks = torch.tensor(self.meta_model.sampled_ex_masks, dtype=torch.bool, device=0)
                log_probs = torch.stack(self.meta_model.log_probs)

                advantages = target_loss - losses[sampled_ex_masks]
                meta_loss = torch.mean(- log_probs * advantages)

                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                self.meta_loss += meta_loss.item()

                logger.info("num of sampled_ex %s / %s", str(int(sampled_ex_masks.sum())), str(sampled_ex_masks.size(0)))
                self.meta_model() # update the sampler
        
        self.train_sampler._epsilon_fn(self.global_step, self.t_total)

    def _init_logging(self):
        super()._init_logging()
        self.meta_loss, self.logging_meta_loss = 0.0, 0.0

    def _train_writer(self, logging_steps):
        super()._train_writer(logging_steps)
        # this is where to to logging.
        self.tb_writer.add_scalar("meta_loss", (self.meta_loss - self.logging_meta_loss)/logging_steps, self.global_step)
        self.tb_writer.add_scalar("epsilon", self.train_sampler.epsilon, self.global_step)
        self.logging_meta_loss = self.meta_loss

        loss_dict = {}
        loss_size = {}
        
        for domain_id in self.meta_model.domain_losses:
            domain_loss_deque = self.meta_model.domain_losses[domain_id]
            if len(domain_loss_deque) > 0:
                domain_loss = float(np.mean(list(domain_loss_deque)))
            else:
                domain_loss = float("inf")
            loss_dict[domain_id] = domain_loss
            loss_size[domain_id] = len(domain_loss_deque)
        
        loss_rank = [self.id_to_domain[k]+": "+str(loss_size[k])+"|"+str(v) for k_id, (k, v) in enumerate(sorted(loss_dict.items(), key=lambda item: item[1])) if k_id < 30]
        
        logger.info("top-30 losses (domains: losses) \n%s\n", "\n".join(loss_rank))

        cnt_rank = [self.id_to_domain[k]+": "+str(v)+"|"+str(loss_dict[k]) for k_id, (k, v) in enumerate(sorted(self.meta_model.domain_cnts.items(), key=lambda item: item[1], reverse=True)) if k_id < 30]
                
        logger.info("top-30 examples (domains : counts, loss) \n%s\n", "\n".join(cnt_rank))
        self.meta_model.domain_cnts = defaultdict(int) # reset
        
        # with torch.no_grad():
        #     _, domain_ids = torch.topk(self.meta_model.domain_logits.data, 30)
        # domain_rank = [self.id_to_domain[domain_id.item()]+": "+str(loss_dict[domain_id.item()]) for domain_id in domain_ids]
                
        # logger.info("top-30 logits (domains: losses) \n %s", "\n".join(domain_rank))
        