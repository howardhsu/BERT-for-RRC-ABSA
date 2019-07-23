"""
Note: this is not intended to be a reproducible code for the NAACL paper but a reference implementation on converting xml files to tokenized-BIO format.
"""

import nltk
import numpy as np
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
import random
random.seed(1337)
np.random.seed(1337)

def parse_SemEval14(fn):
    root=ET.parse(fn).getroot()
    corpus={}
    label=[]
    opin_cnt=[0]*len(polar_idx)
    for sent in root.iter("sentence"):
        text=[]
        opins=set()
        for opin in sent.iter('aspectTerm'):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib['term']!="NULL":
                opins.add((opin.attrib['term'], int(opin.attrib['from']), int(opin.attrib['to']), polar_idx[opin.attrib['polarity'] ] ) )
                    
        for ix, c in enumerate(sent.find('text').text ):
            for opin in opins:
                if (c=='/' or c=='*' or c=='-' or c=='=') and len(text)>0 and text[-1]!=' ':
                    text.append(' ')
                if ix==int(opin[1] ) and len(text)>0 and text[-1]!=' ':
                    text.append(' ')
                elif ix==int(opin[2] ) and len(text)>0 and text[-1]!=' ' and c!=' ':
                    text.append(' ')
            text.append(c)
            if (c=='/' or c=='*' or c=='-' or c=='=') and text[-1]!=' ':
                text.append(' ')
            
        text="".join(text)
        tokens=nltk.word_tokenize(text)
        lb=[0]*len(tokens)
        for opin in opins:
            opin_cnt[opin[3]]+=1
            token_idx, pt, tag_on=0, 0, False
            for ix, c in enumerate(sent.find('text').text):
                if pt>=len(tokens[token_idx] ):
                    pt=0
                    token_idx+=1
                    if token_idx>=len(tokens):
                        break
                #print sent.find('text').text
                #print tokens[token_idx][pt],"<->",c   
                if ix==opin[1]: #from
                    assert pt==0 and c!=' '
                    lb[token_idx]=1
                    tag_on=True
                elif ix==opin[2]: #to
                    assert pt==0
                    tag_on=False   
                elif tag_on and pt==0 and c!=' ':
                    lb[token_idx]=2
                if c==' ' or ord(c)==160:
                    pass
                elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                    pt+=2
                else:
                    pt+=1
        corpus[int(sent.attrib['id'])]={"tokens": tokens, "labels": lb}
    print opin_cnt
    print len(corpus)
    return corpus



def parse_SemEval16(fn):
    sent_tokens, sent_labels = [], []
    root=ET.parse(fn).getroot()
    for sent in review.iter("sentence"):
        text=[]
        opins=set()
        for opin in sent.iter('Opinion'):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib['target']!="NULL":
                opins.add((opin.attrib['target'], int(opin.attrib['from'] ), int(opin.attrib['to'] ), polar_idx[opin.attrib['polarity'] ] ) )
        for ix, c in enumerate(sent.find('text').text ):
            for opin in opins:
                if (c=='/' or c=='*' or c=='-' or c=='=') and len(text)>0 and text[-1]!=' ':
                    text.append(' ')
                if ix==int(opin[1] ) and len(text)>0 and text[-1]!=' ':
                    text.append(' ')
                elif ix==int(opin[2] ) and len(text)>0 and text[-1]!=' ' and c!=' ':
                    text.append(' ')
            text.append(c)
            if (c=='/' or c=='*' or c=='-' or c=='=') and text[-1]!=' ':
                text.append(' ')

        text="".join(text)
        tokens=nltk.word_tokenize(text)
        lb=[0]*len(tokens)
        for opin in opins:
            token_idx, pt, tag_on=0, 0, False
            for ix, c in enumerate(sent.find('text').text):
                if pt>=len(tokens[token_idx] ):
                    pt=0
                    token_idx+=1
                    if token_idx>=len(tokens):
                        break
                #print sent.find('text').text
                #print tokens[token_idx][pt],"<->",c   
                if ix==opin[1]:
                    assert pt==0 and c!=' '
                    lb[token_idx]=1
                    tag_on=True
                elif ix==opin[2]:
                    assert pt==0
                    tag_on=False   
                elif tag_on and pt==0 and c!=' ':
                    lb[token_idx]=2
                if c==' ':
                    pass
                elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                    pt+=2
                else:
                    pt+=1
        sent_tokens.append(tokens)
        sent_labels.append(lb)
    return sent_tokens, sent_labels
