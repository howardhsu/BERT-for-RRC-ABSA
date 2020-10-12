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
        
    def mask_tokens(self, inputs, model, mlm_probability = 0.15, update_samples = False):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        masked_indices = self.compute_masked_indices(inputs, model, mlm_probability)
        return self.gen_inputs_labels(inputs, masked_indices)

        
class BertMasker(Masker):

    def compute_masked_indices(self, inputs, model, mlm_probability):
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


class DomBertFixedLengthMasker(BertMasker):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.cached_masks = None
        
    def mask_tokens(self, inputs, model, mlm_probability = 0.15, update_samples = False):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        domain_ids, inputs = inputs[:,:1], inputs[:,1:]
        masked_indices = self.compute_masked_indices(inputs, model, mlm_probability)
        inputs, labels = self.gen_inputs_labels(inputs, masked_indices)
        return torch.cat([domain_ids, inputs], dim=1), labels

    def compute_masked_indices(self, inputs, model, mlm_probability):
        batch_size, seq_len = inputs.size()
        if self.cached_masks is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inputs.tolist()[0], already_has_special_tokens=True)
            self.allowed_idxs = torch.LongTensor([token_idx for token_idx in range(seq_len) if not special_tokens_mask[token_idx]])
            self.mask_cnt = int((seq_len - sum(special_tokens_mask)) * mlm_probability)
            cached_masks = []
            for _ in range(1024):
                masked_index = torch.zeros((seq_len,), dtype=torch.bool)
                masked_index[self.allowed_idxs[torch.randperm(self.allowed_idxs.size(0))][:self.mask_cnt]] = True
                cached_masks.append(masked_index)
            self.cached_masks = torch.stack(cached_masks, dim=0)
        masked_indices = self.cached_masks[torch.randint(self.cached_masks.size(0), (batch_size,))]
        return masked_indices
    
    
class DomEmbBertMasker(BertMasker):
    """
    Handle the first position as domain embedding.
    """

    def mask_tokens(self, inputs, model, mlm_probability = 0.15, update_samples = False):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        domain_ids, input_ids = inputs[:,:1], inputs[:,1:]
        domain_labels = domain_ids.clone()
        # randomly flip 10% to [GENERAL] domain.
        domain_ids[torch.bernoulli(torch.full(domain_ids.shape, 0.1)).bool()] = 0

        masked_indices = self.compute_masked_indices(input_ids, model, mlm_probability)
        input_ids, labels = self.gen_inputs_labels(input_ids, masked_indices)
        
        return torch.cat([domain_ids, input_ids], dim=1), labels


class SkipDomEmbBertMasker(BertMasker):
    """
    Handle the first position as domain embedding.
    """

    def mask_tokens(self, inputs, model, mlm_probability = 0.15, update_samples = False):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        domain_ids, input_ids = inputs[:,:1], inputs[:,1:]
        return super().mask_tokens(input_ids, model, mlm_probability, update_samples)
