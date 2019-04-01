# BERT for Review Reading Comprehension and Aspect-based Sentiment Analysis and Aspect-based Sentiment Analysis
code for our NAACL 2019 paper "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis".

## Problem to Solve
We focus on 3 review-based tasks: RRC, AE and ASC.

## Environment
The code is tested on Python 3.6.8, PyTorch 1.0.1 and [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4. 

The post-training code additionally use [apex](https://github.com/NVIDIA/apex) 0.1 to speed up training on FP16, which is compiled with PyTorch 1.0.1(py3.6_cuda10.0.130_cudnn7.4.2_2) and CUDA 10.0.130 on RTX 2080 Ti. It is possible to avoid use apex on no-Turing GPU, e.g., 1080 Ti, but need to adjust the max sequence length and steps of gradient accumulation.

Fine-tuning code is tested without using apex 0.1 to ensure stability.

This repository is still under development.

## TODO:
- [ ] preprocessed data
- [ ] pretained model
- [ ] post-training code
- [ ] fine-tuning code
- [ ] example running script
