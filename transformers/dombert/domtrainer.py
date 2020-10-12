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


class DomSampler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.domain_embedding = model.domain_cls.predictions.decoder
        self.sim_fn = nn.CosineSimilarity(-1)
        self.domain_cnts = defaultdict(int)

    def forward(self):
        with torch.no_grad():
            # laptop uses 0.1, restaurant uses 0.13
            tp = 0.1 # 0.13
            logits = self.sim_fn(self.domain_embedding.weight.data[0], self.domain_embedding.weight.data) / tp
            # logits = torch.matmul(self.domain_embedding.weight.data[0], self.domain_embedding.weight.data.transpose(1, 0))
            probs = torch.softmax(logits.float(), 0)
            self.sampler = Categorical(probs)


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

    def __iter__(self):
        ex_cnt = 0
        while ex_cnt < len(self.domain_data_dict[0]):
            domain_idx = self.domain_sampler.sampler.sample().item()
            while len(self.domain_data_dict[domain_idx]) == 0:
                domain_idx = self.domain_sampler.sampler.sample().item()
            self.domain_sampler.domain_cnts[domain_idx] += 1
            yield self.domain_data_dict[domain_idx][self.domain_offsets[domain_idx]]
            ex_cnt += 1
            self.domain_offsets[domain_idx] += 1
            if self.domain_offsets[domain_idx] >= len(self.domain_data_dict[domain_idx]):
                random.shuffle(self.domain_data_dict[domain_idx])
                self.domain_offsets[domain_idx] = 0
            
    def __len__(self):
        return len(self.domain_data_dict[0])


class DomTrainer(Trainer):    
    def _build_train_data_loader(self, args, train_dataset, model):
        with open("data/pt/domain_tag/doi_domain.json") as f:
            domain_dict = json.load(f)
        self.id_to_domain = {domain_dict[domain]: domain for domain in domain_dict}
        
        self.meta_model = DomSampler(model)
        self.meta_model()

        self.train_sampler = DomainSampler(train_dataset, self.meta_model)
        train_dataloader = DataLoader(train_dataset, sampler=self.train_sampler, batch_size=args.train_batch_size)
        return self.train_sampler, train_dataloader
    
    def _post_step(self, args, outputs):
        # outputs: 0 is the total loss, 1 is the lm_loss, 2 is the domain_cls loss.
        if args.gradient_accumulation_steps > 1:
            lm_loss = outputs[1] / args.gradient_accumulation_steps
            domain_cls_loss = outputs[2] / args.gradient_accumulation_steps
        self.lm_loss += lm_loss.item()
        self.domain_cls_loss += domain_cls_loss.item()

    def _post_training(self):
        self.meta_model() # update the sampler

    def _init_logging(self):
        super()._init_logging()
        self.lm_loss, self.logging_lm_loss, self.domain_cls_loss, self.logging_domain_cls_loss = 0.0, 0.0, 0.0, 0.0

    def _train_writer(self, logging_steps):
        super()._train_writer(logging_steps)
        self.tb_writer.add_scalar("lm_loss", (self.lm_loss - self.logging_lm_loss)/logging_steps, self.global_step)
        self.tb_writer.add_scalar("domain_cls_loss", (self.domain_cls_loss - self.logging_domain_cls_loss)/logging_steps, self.global_step)
        self.logging_lm_loss = self.lm_loss
        self.logging_domain_cls_loss = self.domain_cls_loss
        
        cnt_rank = [self.id_to_domain[k]+": "+str(v) for k_id, (k, v) in enumerate(sorted(self.meta_model.domain_cnts.items(), key=lambda item: item[1], reverse=True)) if k_id < 30]
        logger.info("total sampled examples: %s", str(sum(list(self.meta_model.domain_cnts.values()))))
        logger.info("top-30 examples (domains : counts) \n%s\n", "\n".join(cnt_rank))
        self.meta_model.domain_cnts = defaultdict(int) # reset

        with torch.no_grad():
            probs, domain_ids = torch.topk(self.meta_model.sampler.probs, 30)
            domain_rank = [self.id_to_domain[domain_id.item()]+": "+str(float(probs[rank_id])) for rank_id, domain_id in enumerate(domain_ids)]        
        logger.info("top-30 probs (domains: probs) \n%s\n", "\n".join(domain_rank))
