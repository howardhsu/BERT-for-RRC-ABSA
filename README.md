# BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis
code for our NAACL 2019 paper "[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/N19-1242.pdf)", COLING 2020 paper "[Understanding Pre-trained BERT for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2011.00169)" and (draft code of) Findings of EMNLP 2020 "[DomBERT: Domain-oriented Language Model for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2004.13816)".

We found that BERT domain post-training (e.g, 1 day of training) is an economic way to boost the performance of BERT, because it is much harder (e.g., 10 days of training) to learn a general knowledge shared across domains and, meanwhile, loosing the long-tailed domain-specific knowledge.

## News
[Code base](analab.md) for "Understanding Pre-trained BERT for Aspect-based Sentiment Analysis" is released.  
[Code base](transformers.md) on huggingface `transformers` is under `transformers`, with more cross-domain models.  
Preprocessing ABSA xmls organized into a separate [rep](https://github.com/howardhsu/ABSA_preprocessing).  
Want to have post-trained models for other domains in reviews ? checkout a [cross-domain review BERT](transformers/amazon_yelp.md) or download from [HERE](https://drive.google.com/file/d/1YbiI9W3acj4d9JbCbu_SmRjz_tNyShYV/view?usp=sharing).   
A conversational dataset of RRC can be found [here](https://github.com/howardhsu/RCRC).  
If you only care about ASC, a more formal code base can be found in a [similar rep](https://github.com/howardhsu/ASC_failure) focusing on ASC.
**feedbacks are welcomed for missing instructions **

## Problem to Solve
We focus on 3 review-based tasks: review reading comprehension (RRC), aspect extraction (AE) and aspect sentiment classification (ASC).

RRC: given a question ("how is the retina display ?") and a review ("The retina display is great.") find an answer span ("great") from that review;

AE: given a review sentence ("The retina display is great."), find aspects("retina display");

ASC: given an aspect ("retina display") and a review sentence ("The retina display is great."), detect the polarity of that aspect (positive).

[E2E-ABSA](https://github.com/lixin4ever/E2E-TBSA): the combination of the above two tasks as a sequence labeling task.

And how a pre-trained BERT model on reviews be prepared for those tasks.   

## Code Base
For post-training of NAACL 2019 paper, the code base is splited into two versions: `transformers/` ([instructions](transformers.md)) and `pytorch-pretrained-bert/` ([instructions](pytorch-pretrained-bert.md)). 

For analysis of pre-trained BERT model for ABSA (COLING 2020), see this [instructions](analab.md).

Please check corresponding instructions for details.

## Citation
If you find this work useful, please cite as following.
```
@inproceedings{xu_bert2019,
    title = "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis",
    author = "Xu, Hu and Liu, Bing and Shu, Lei and Yu, Philip S.",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = "jun",
    year = "2019",
}
```

```
@inproceedings{xu_understanding2020,
    title = "Understanding Pre-trained BERT for Aspect-based Sentiment Analysis",
    author = "Xu, Hu and Shu, Lei and Yu, Philip S. and Liu, Bing",
    booktitle = "The 28th International Conference on Computational Linguistics",
    month = "Dec",
    year = "2020",
}
```
