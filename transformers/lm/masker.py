import os
import numpy as np
import torch
from tqdm import tqdm
import random

import logging

logger = logging.getLogger(__name__)


class Masker(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def compute_masked_indices(self, inputs, model, mlm_probability):
        raise NotImplementedError
    
    def gen_inputs_labels(self, inputs, masked_indices):
        raise NotImplementedError
        
    def mask_tokens(self, inputs, mlm_probability = 0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        masked_indices = self.compute_masked_indices(inputs, mlm_probability)
        return self.gen_inputs_labels(inputs, masked_indices)

        
class BertMasker(Masker):

    def compute_masked_indices(self, inputs, mlm_probability):
        probability_matrix = torch.full(inputs.shape, mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices

    def gen_inputs_labels(self, inputs, masked_indices):
        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
        inputs = inputs.clone()
        labels = inputs.clone()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class SkipDomBertMasker(BertMasker):

    def mask_tokens(self, inputs, mlm_probability = 0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        return super().mask_tokens(inputs[:,1:])
