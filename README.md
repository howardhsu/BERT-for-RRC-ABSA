# BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis
code for our NAACL 2019 paper "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis".

## Problem to Solve
We focus on 3 review-based tasks: RRC, AE and ASC.

## Environment

### fine-tuning
The code is tested on Python 3.6.8, PyTorch 1.0.1 and [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4. 

### post-training
The post-training code additionally use [apex](https://github.com/NVIDIA/apex) 0.1 to speed up training on FP16, which is compiled with PyTorch 1.0.1(py3.6_cuda10.0.130_cudnn7.4.2_2) and CUDA 10.0.130 on RTX 2080 Ti. It is possible to avoid use GPUs that do not support apex (e.g., 1080 Ti), but need to adjust the max sequence length and number of gradient accumulation but (although the result can be better). 

Fine-tuning code is tested without using apex 0.1 to ensure stability.

### evaluation
Our evaluation wrapper code is written in ipython notebook. 
But you are free to call the evaluation code of each task separately.
AE additionally needs Java JRE/JDK to be installed.

This repository is still under development.

## TODO:
- [ ] data for RRC.
- [ ] preprocessed data
- [ ] pretained model
- [ ] post-training code
- [ ] fine-tuning code
- [ ] example running script

## Citation
If you find this work useful, please cite as following.
```
@inproceedings{xu_bert2019,
    title = "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis",
    author = "Xu, Hu  and
      Liu, Bing and
      Shu, Lei  and
      Yu, Philip S.",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = jun,
    year = "2019",
}
'''
