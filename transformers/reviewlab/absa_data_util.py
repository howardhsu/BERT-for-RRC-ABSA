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


import json
import os
from collections import defaultdict
import random
import torch
from torch.utils.data import TensorDataset


import logging
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, config):
        self.data_dir = config.data_dir
        
    def get_train_examples(self, fn="train.json"):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError

    def get_dev_examples(self, fn="dev.json"):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError
        
    def get_test_examples(self, fn="test.json"):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)
        

class ABSAProcessor(DataProcessor):
    """Processor for the ABSA."""
    def _load_data(self, fn, set_type):
        json_data = self._read_json(os.path.join(self.data_dir, fn))
        if "data" in json_data:
            json_data = json_data["data"]
        return self._create_examples(json_data, set_type)
        
    def get_train_examples(self, fn="train.json"):
        return self._load_data(fn, "train")

    def get_dev_examples(self, fn="dev.json"):
        return self._load_data(fn, "dev")
    
    def get_test_examples(self, fn="test.json"):
        return self._load_data(fn, "test")


class AEProcessor(ABSAProcessor):
    """Processor for Aspect Extraction."""

    def get_labels(self):
        """TODO: load meta data from the dataset."""
        return ["O", "B", "I"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[ids]['sentence']
            text_b = None
            label = lines[ids]['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label) )
        return examples        
        

class ASCProcessor(ABSAProcessor):
    """Processor for Aspect Sentiment Classification."""

    def get_labels(self):
        """TODO: load meta data from the dataset."""
        return ["positive", "negative", "neutral"]
    
    def _create_examples(self, lines, set_type):        
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[ids]['term']
            text_b = lines[ids]['sentence']
            label = lines[ids]['polarity']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples     


class ACCProcessor(ABSAProcessor):
    """Processor for the  Aspect Sentiment Classification."""

    def get_labels(self):
        """See base class."""
        data = self._read_json(os.path.join(self.data_dir, "train.json"))
        return data["meta"]["label_list"]
    
    def _create_examples(self, lines, set_type):        
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[ids]['term']
            text_b = lines[ids]['sentence']
            label = lines[ids]['category']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 


class E2EProcessor(ABSAProcessor):
    """End-to-end Processor of ABSA."""

    def get_labels(self):
        data = self._read_json(os.path.join(self.data_dir, "train.json"))
        return data["meta"]["label_list"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[ids]['tokens']
            text_b = None
            label = lines[ids]['labels']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ASCConverter(object):
    """Aspect Sentiment Classification Converter: convert Examples to Tensors.
    """

    @classmethod
    def convert_examples_to_features(cls, config, examples, tokenizer, max_length, label_list):
        """
        Loads a data file into a list of ``InputFeatures``
        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.
        """

        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d" % (ex_index))

            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens = True,
                max_length = max_length,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)

            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            
            # assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            label = label_map[example.label]

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  label=label))
        return features



def build_dataset(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    label = torch.tensor([f.label for f in features], dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, token_type_ids, label)


class TokenCLSConverter(object):
    """Token Classification Converter (e.g., Aspect Extraction or E2E ABSA)."""
    
    @classmethod
    def convert_examples_to_features(cls, config, 
                                     examples,
                                     tokenizer,
                                     max_seq_length,
                                     label_list,
                                     cls_token_at_end=False,
                                     cls_token="[CLS]",
                                     cls_token_segment_id = 0,
                                     sep_token="[SEP]",
                                     sep_token_extra=False,
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     pad_token_label_id=-100,
                                     sequence_a_segment_id=0,
                                     mask_padding_with_zero=True,
                                     max_tag_size=16
                                     ):
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map.update({"-100": pad_token_label_id})
        
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            tokens = []
            label_ids = []

            for word, label in zip(example.text_a, example.label):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
                label_ids = label_ids[:(max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]

            segment_ids = [sequence_a_segment_id] * len(tokens)            

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += ([pad_token] * padding_length)
                input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids += ([pad_token_segment_id] * padding_length)
                label_ids += ([pad_token_label_id] * padding_length)
                    
            # assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert min(label_ids) >= -100 and max(label_ids) < len(label_list)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            features.append(
                InputFeatures(input_ids = input_ids,
                              attention_mask = input_mask,
                              token_type_ids = segment_ids,
                              label = label_ids))

        return features
