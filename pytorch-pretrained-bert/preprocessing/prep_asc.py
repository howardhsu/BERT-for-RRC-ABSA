import nltk
import numpy as np
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
import random
random.seed(1337)
np.random.seed(1337)

"""TODO: this file is not well-tested but just copied from another repository.
"""


valid_split=150
polar_idx={'positive': 0, 'negative': 1, 'neutral': 2}
idx_polar={0: 'positive', 1: 'negative', 2: 'neutral'}

def parse_SemEval14(fn):
    root=ET.parse(fn).getroot()
    corpus=[]
    opin_cnt=[0]*len(polar_idx)
    for sent in root.iter("sentence"):
        opins=set()
        for opin in sent.iter('aspectTerm'):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib['term']!="NULL":
                if opin.attrib['polarity'] in polar_idx:
                    opins.add((opin.attrib['term'], int(opin.attrib['from']), int(opin.attrib['to']), opin.attrib['polarity'] ) )
        for ix, opin in enumerate(opins):
            opin_cnt[polar_idx[opin[3] ] ]+=1
            corpus.append({"id": sent.attrib['id']+"_"+str(ix), "sentence": sent.find('text').text, "term": opin[0], "polarity": opin[-1]})
    print opin_cnt
    print len(corpus)
    return corpus

train_corpus=parse_SemEval14('../SemEval/SemEval14/Laptop_Train_v2.xml')
with open("../asc/laptop/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split] }, fw, sort_keys=True, indent=4)
with open("../asc/laptop/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:] }, fw, sort_keys=True, indent=4)
test_corpus=parse_SemEval14('../SemEval/SemEval14/Laptops_Test_Gold.xml')
with open("../asc/laptop/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)
    
train_corpus=parse_SemEval14('../SemEval/SemEval14/Restaurants_Train_v2.xml')
with open("../asc/rest/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split] }, fw, sort_keys=True, indent=4)
with open("../asc/rest/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:] }, fw, sort_keys=True, indent=4)
test_corpus=parse_SemEval14('../SemEval/SemEval14/Restaurants_Test_Gold.xml')
with open("../rest/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)