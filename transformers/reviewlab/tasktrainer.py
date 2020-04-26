
import numpy as np
import os
import json
import torch

from transformers import (
    BertForSequenceClassification,
    BertForTokenClassification,
)

# remove this in future if transformers implemented a TokenClassification class.
from .modeling import *

from . import absa_data_util as data_util
from .trainer import Trainer

import logging
logger = logging.getLogger(__name__)


SEQ_CLS_MODEL_CLASSES = {
    'bert': BertForSequenceClassification,
}


TOKEN_CLS_MODEL_CLASSES = {
    'bert': BertForTokenClassification,
}


class TokenCLSTrainer(Trainer):
    """
    all sequence labeling tasks (AE, E2E) share this class.
    """
    
    def _convert_examples_to_features(self, config, examples, tokenizer, max_seq_length, label_list, model_type):
        return data_util.TokenCLSConverter.convert_examples_to_features(
            config, examples, tokenizer, max_seq_length, label_list, 
            cls_token=tokenizer.cls_token,
            sep_token = tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        )
    
    def _predict(self, args, model, eval_examples, eval_dataloader, labels):
        
        preds = None
        out_label_ids = None
        
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], 
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2], 
                    "labels": batch[3]}
            if args.model_type == 'distilbert': 
                inputs.pop("token_type_ids", None)

            outputs = model(**inputs)
            _, logits = outputs[:2]

        
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
            
            if i < 5:
                logger.info("*** Evaluation Example ***")
                logger.info("guid: %s" % (eval_examples[i].guid))
                logger.info("out_label_list: %s " % str(out_label_list[i]) )
                logger.info("preds_list: %s " % str(preds_list[i]) )
        
        # prepare the output.        
        # sort by original order for evaluation because the BERT preprocessor is dict-based.
        # TODO: merge the different keys of sorting of different token cls tasks.

        output = {"sentence": [], 
                "out_label_list": [],
                "preds_list": [],
                "labels": labels
                }
        
        for i, ex in enumerate(eval_examples):
            output["sentence"].append(ex.text_a)
            output["out_label_list"].append(out_label_list[i])
            output["preds_list"].append(preds_list[i])

        output_eval_json = os.path.join(args.output_dir, "predictions.json")
        
        with open(output_eval_json, "w") as fw:            
            json.dump(output, fw)

    
class AETrainer(TokenCLSTrainer):
    def _config_task(self, config):
        processor = getattr(data_util, "AEProcessor")(config)
        model_class = TOKEN_CLS_MODEL_CLASSES[config.model_type]
        return processor, model_class


    def _predict(self, args, model, eval_examples, eval_dataloader, labels):

        preds = None
        out_label_ids = None
        
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], 
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2], 
                    "labels": batch[3]}
            if args.model_type == 'distilbert': 
                inputs.pop("token_type_ids", None)

            outputs = model(**inputs)
            _, logits = outputs[:2]

        
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
            
            if i < 5:
                logger.info("*** Evaluation Example ***")
                logger.info("guid: %s" % (eval_examples[i].guid))
                logger.info("out_label_list: %s " % str(out_label_list[i]) )
                logger.info("preds_list: %s " % str(preds_list[i]) )
        
        # prepare the output.        
        #sort by original order for evaluation because the BERT preprocessor is dict-based.
        recs = {}
        for i, ex in enumerate(eval_examples):
            recs[int(ex.guid.split("-")[1]) ] = {
                "sentence": ex.text_a,
                "out_label_list": out_label_list[i],
                "preds_list": preds_list[i],
            }

        output = {"sentence": [recs[i]["sentence"] for i in range(len(recs))], 
                "out_label_list": [recs[qx]["out_label_list"] for qx in range(len(recs))], 
                "preds_list": [recs[qx]["preds_list"] for qx in range(len(recs))],
                "labels": labels
                }

        output_eval_json = os.path.join(args.output_dir, "predictions.json")
        
        with open(output_eval_json, "w") as fw:            
            json.dump(output, fw)
    

class E2ETrainer(TokenCLSTrainer):
    def _config_task(self, config):
        processor = getattr(data_util, "E2EProcessor")(config)
        model_class = TOKEN_CLS_MODEL_CLASSES[config.model_type]
        return processor, model_class


class ASCTrainer(Trainer):
    """An ASC trainer."""

    def _convert_examples_to_features(self, config, examples, tokenizer, max_seq_length, label_list, model_type):
        return data_util.ASCConverter.convert_examples_to_features(config, examples, tokenizer, max_seq_length, label_list)
    
    
    def _config_task(self, config):
        processor = getattr(data_util, "ASCProcessor")(config)
        model_class = SEQ_CLS_MODEL_CLASSES[config.model_type]
        return processor, model_class
    
    
    def _predict(self, args, model, eval_examples, eval_dataloader, labels):
        
        full_logits=[]
        full_label_ids=[]
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], 
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2], }
            if args.model_type == 'distilbert': 
                inputs.pop("token_type_ids", None)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()

            full_logits.extend(logits.tolist() )
            full_label_ids.extend(label_ids.tolist() )

        output_eval_json = os.path.join(args.output_dir, "predictions.json") 
        with open(output_eval_json, "w") as fw:
            json.dump({"logits": full_logits, 
                       "label_ids": full_label_ids, 
                       "ids": [ex.guid.split("-")[1] for ex in eval_examples] 
                      }, fw)
