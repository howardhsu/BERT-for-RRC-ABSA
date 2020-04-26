
import os
import argparse
import json
import random
import numpy as np
import torch
import logging
import gzip
from collections import defaultdict


"""
v2: add rating, remove dev (all dev is splited from testing).
v1: split some testing data into dev2.
v0: initial version.
"""

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

def write_text(fn, data, skip_tag=True):
    with open(fn, "w") as fw:
        for (asin, tag, rating, text) in data:
            if not skip_tag:
                fw.write("\n"+asin+" "+tag+" "+rating+"\n")
            fw.write(text+"\n")

def amazon(keep_probs, with_test, train_data, test_data):
    laptop_tag = None
    asin2cat = {}
    with gzip.open("data/pt/kcore_5_review/metadata.json.gz") as f:
        for l in f:
            rec = eval(l)
            # skip all laptop reviews for zero-shot setting.
            if "categories" in rec:
                # we assume only use the first category.
                cat = "/".join(rec["categories"][0])
                if "Laptops" in cat:
                    laptop_tag = cat
                if len(cat) > 0: 
                    asin2cat[rec["asin"]] = cat

    print(laptop_tag)

    with gzip.open("data/pt/kcore_5_review/kcore_5.json.gz") as f:
        for l in f:
            rec = json.loads(l)
            asin = rec["asin"]
            if asin in asin2cat:
                text = rec['reviewText'].replace('\n',' ').strip()
                if len(text) > 0:
                    rnd = random.random() 
                    if rnd < keep_probs:
                        if rnd < 0.0005:
                            # keep training data still exactly the same.
                            test_data.append((asin, asin2cat[asin], str(rec["overall"]), text))
                        else:
                            train_data.append((asin, asin2cat[asin], str(rec["overall"]), text))
                    else:
                        test_data.append((asin, asin2cat[asin], str(rec["overall"]), text))
                        

def yelp(keep_probs, with_test, train_data, test_data):

    asin2cat = {}

    with open("data/pt/yelp/business.json") as f:
        for l in f:
            rec = json.loads(l)
            if "categories" in rec and rec["categories"]:
                if "Restaurants" in rec["categories"]:
                    cat = "Restaurants" 
                else:
                    cat = rec["categories"].split(",")[0]
                if len(cat) > 0:
                    asin2cat[rec["business_id"]] = cat
    
    print("Restaurants")

    with open("data/pt/yelp/review.json") as f:
        for l in f:
            rec = json.loads(l)
            asin = rec["business_id"]
            if asin in asin2cat:
                text = rec['text'].replace('\n',' ').strip()    
                if len(text) > 0:
                    rnd = random.random() 
                    if rnd < keep_probs:
                        if rnd < 0.0005:
                            test_data.append((asin, asin2cat[asin], str(rec["stars"]), text))
                        else:
                            train_data.append((asin, asin2cat[asin], str(rec["stars"]), text))
                    else:
                        test_data.append((asin, asin2cat[asin], str(rec["stars"]), text))
    

def main(keep_probs = 0.8, with_test = True):
    """
    a corpus organized by domain.
    """
    
    random.seed(0)

    train_data = []
    test_data = []

    yelp(keep_probs, with_test, train_data, test_data)
    amazon(keep_probs, with_test, train_data, test_data)
    
    random.shuffle(train_data)
    if with_test:
        random.shuffle(test_data)
        
    split_test_as_dev2 = int(len(test_data) * 0.05)
    write_text("data/pt/domain_v2_train.txt", train_data, skip_tag=False)
    write_text("data/pt/domain_v2_dev.txt", test_data[:split_test_as_dev2], skip_tag=False)
    if with_test:
        write_text("data/pt/domain_v2_test.txt", test_data[split_test_as_dev2:], skip_tag=False)


if __name__=="__main__":
    main()
