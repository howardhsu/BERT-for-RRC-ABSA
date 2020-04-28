
import json
import os
import sklearn.metrics
import numpy as np
import glob
from collections import Counter, namedtuple
import time
import json
import math
import random
import argparse

import xml.etree.ElementTree as ET
from subprocess import check_output

from reviewlab.seq_utils import compute_metrics_absa


import logging
logger = logging.getLogger(__name__)


class Metric(object):
    @classmethod
    def evaluate(cls, config, results):
        raise NotImplementedError

    @property
    def metric_name(cls, index):
        raise NotImplementedError
        
class AEMetric(object):
    commands = {
        "rest": "./java -cp reviewlab/eval/A.jar absa16.Do Eval -prd data/ft/ae/16/rest/rest_pred.xml -gld data/ft/ae/16/rest/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1",
        "laptop": "./java -cp reviewlab/eval/eval.jar Main.Aspects data/ft/ae/14/laptop/laptop_pred.xml data/ft/ae/14/laptop/Laptops_Test_Gold.xml"
    }
    
    templates = {
        "rest": "data/ft/ae/16/rest/EN_REST_SB1_TEST.xml.A",
        "laptop": "data/ft/ae/14/laptop/Laptops_Test_Data_PhaseA.xml",
    }
    
    @classmethod
    def metric_name(cls, index):
        if index == 0:
            return "f1"
        else:
            raise Exception("unknown index")

    @classmethod
    def evaluate(cls, config, pred_json):    
        y_pred = [[pred_json["labels"].index(pred) for pred in preds] for preds in pred_json["preds_list"]]
        
        command = AEMetric.commands[config.domain].split()
        if config.domain == "rest":
            AEMetric._label_rest_xml(AEMetric.templates[config.domain], command[6], pred_json["sentence"], y_pred)
            result = check_output(command).split()
            logger.info("**** java output ****")
            logger.info("%s", result)
            return [float(result[9][10:])]
        elif config.domain == "laptop":
            AEMetric._label_laptop_xml(AEMetric.templates[config.domain], command[4], pred_json["sentence"], y_pred)
            result = check_output(command).split()
            logger.info("**** java output ****")
            logger.info("%s", result)
            return [float(result[15])]
        else:
            raise ValueError("unknown domain %s.", config.domain)

    @classmethod
    def _label_rest_xml(cls, fn, output_fn, corpus, label):
        dom=ET.parse(fn)
        root=dom.getroot()
        pred_y=[]
        for zx, sent in enumerate(root.iter("sentence") ) :
            tokens=corpus[zx]
            lb=label[zx]
            opins=ET.Element("Opinions")
            token_idx, pt, tag_on=0, 0, False
            start, end=-1, -1
            for ix, c in enumerate(sent.find('text').text):
                if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                    pt=0
                    token_idx+=1
                
                if (token_idx>=len(tokens) or token_idx >= len(lb)):
                    if tag_on:
                        end=ix
                        tag_on=False 
                        opin=ET.Element("Opinion")
                        opin.attrib['target']=sent.find('text').text[start:end]
                        opin.attrib['from']=str(start)
                        opin.attrib['to']=str(end)
                        opins.append(opin)
                    else:
                        break
                elif token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                    if tag_on:
                        end=ix
                        tag_on=False
                        opin=ET.Element("Opinion")
                        opin.attrib['target']=sent.find('text').text[start:end]
                        opin.attrib['from']=str(start)
                        opin.attrib['to']=str(end)
                        opins.append(opin)
                    start=ix
                    tag_on=True
                elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                    start=ix
                    tag_on=True
                elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                    end=ix
                    tag_on=False 
                    opin=ET.Element("Opinion")
                    opin.attrib['target']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)


                if c==' ':
                    pass
                elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                    pt+=2
                else:
                    pt+=1
            if tag_on:
                tag_on=False
                end=len(sent.find('text').text)
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            sent.append(opins)
        dom.write(output_fn)

    @classmethod
    def _label_laptop_xml(cls, fn, output_fn, corpus, label):
        dom=ET.parse(fn)
        root=dom.getroot()
        pred_y=[]
        for zx, sent in enumerate(root.iter("sentence") ) :
            tokens=corpus[zx]
            lb=label[zx]
            opins=ET.Element("aspectTerms")
            token_idx, pt, tag_on=0, 0, False
            start, end=-1, -1
            for ix, c in enumerate(sent.find('text').text):
                if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                    pt=0
                    token_idx+=1

                if (token_idx>=len(tokens) or token_idx >= len(lb)):
                    if tag_on:
                        end=ix
                        tag_on=False 
                        opin=ET.Element("aspectTerm")
                        opin.attrib['term']=sent.find('text').text[start:end]
                        opin.attrib['from']=str(start)
                        opin.attrib['to']=str(end)
                        opins.append(opin)
                    else:
                        break
                elif token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                    if tag_on:
                        end=ix
                        tag_on=False
                        opin=ET.Element("aspectTerm")
                        opin.attrib['term']=sent.find('text').text[start:end]
                        opin.attrib['from']=str(start)
                        opin.attrib['to']=str(end)
                        opins.append(opin)
                    start=ix
                    tag_on=True
                elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                    start=ix
                    tag_on=True
                elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                    end=ix
                    tag_on=False 
                    opin=ET.Element("aspectTerm")
                    opin.attrib['term']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                    
                    
                if c==' ' or ord(c)==160:
                    pass
                elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                    pt+=2
                else:
                    pt+=1
            if tag_on:
                tag_on=False
                end=len(sent.find('text').text)
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            sent.append(opins )
        dom.write(output_fn)


class ASCMetric(Metric):
    id2metric = {0: "acc", 1: "mf1", 2: "pos_f1", 3: "neg_f1", 4: "neu_f1"}
    @classmethod
    def evaluate(cls, config, results):
        y_true=results['label_ids']
        y_pred=[np.argmax(logit) for logit in results['logits'] ]

        ####### accuracy ########
        acc = 100 * sklearn.metrics.accuracy_score(y_true, y_pred)

        ####### macro f1 ########
        p_macro, r_macro, f_macro, _=sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
        f_macro = 100 * 2 * p_macro*r_macro/(p_macro+r_macro)

        ####### f1 per class ########
        _, _, f_label, _=sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
        return [acc, f_macro] + f_label.tolist()

    @classmethod
    def metric_name(cls, index):
        return ASCMetric.id2metric[index]


class E2EMetric(Metric):
    id2metric = {0: "precision", 1: "recall", 2: "micro-f1", 3: "macro-f1"}
    @classmethod
    def evaluate(cls, config, pred_json):

        OT = {'O': 0, 'EQ': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
        # the evaluation needs a strange extra indexing.
        y_pred = [[[OT[pred] for pred in preds]] for preds in pred_json["preds_list"]]
        y_true = [[[OT[t] for t in trues]] for trues in pred_json["out_label_list"]] # TODO: change the name.

        all_evaluate_label_ids = [0] * len(y_pred)
        results = compute_metrics_absa(y_pred, y_true, all_evaluate_label_ids, tagging_schema = "OT")
        return [results[E2EMetric.id2metric[ix]] for ix in range(len(E2EMetric.id2metric))]

    @classmethod
    def metric_name(cls, index):
        return E2EMetric.id2metric[index]
