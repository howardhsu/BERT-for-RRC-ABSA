# BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis
code for our NAACL 2019 paper "[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis]()".

## Problem to Solve
We focus on 3 review-based tasks: review reading comprehension (RRC), aspect extraction (AE) and aspect sentiment classification (ASC).
RRC: given a question ("how is the retina display ?") and a review ("The retina display is great.") find an answer span ("great") from that review;
AE: given a review sentence ("The retina display is great."), find aspects("retina display");
ASC: given an aspect ("retina display") and a review sentence ("The retina display is great."), detect the polarity of that aspect (positive).

## Environment

### fine-tuning
The code is tested on Python 3.6.8, PyTorch 1.0.1 and [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4. 
We suggest make an anaconda environment for all packages and uncomment environment setup in ```script/run_rrc.sh script/run_absa.sh script/pt.sh```.

### post-training
The post-training code additionally use [apex](https://github.com/NVIDIA/apex) 0.1 to speed up training on FP16, which is compiled with PyTorch 1.0.1(py3.6_cuda10.0.130_cudnn7.4.2_2) and CUDA 10.0.130 on RTX 2080 Ti. It is possible to avoid use GPUs that do not support apex (e.g., 1080 Ti), but need to adjust the max sequence length and number of gradient accumulation but (although the result can be better). 

Fine-tuning code is tested without using apex 0.1 to ensure stability.

### evaluation
Our evaluation wrapper code is written in ipython notebook ```eval/eval.ipynb```. 
But you are free to call the evaluation code of each task separately.
AE ```eval/evaluate_ae.py``` additionally needs Java JRE/JDK to be installed.

### fine-tuning setup

step1: make 2 folders for post-training and fine-tuning.
```
mkdir -p pt_model ; mkdir -p run
```
step2: place post-trained BERT into pt_model. The post-trained Laptop weights can be download [here]() and restaurant [here](). You are free to download other BERT weights into this folder(e.g., [bert-base]() ). Make sure to add an entry into ```src/modelconfig.py```.

step3: make 3 folders for 3 tasks: 
```
mkdir -p rrc ; mkdir -p ae ; mkdir -p asc
```
place fine-tuning data to each respective folder. [rrc](), [ae](), [asc]().

step4: fire a fine-tuning from a BERT weight, e.g.
```
cd script
bash run_rrc.sh rrc laptop_pt laptop pt_rrc 10 0
```
Here rrc is the task to run, laptop_pt is the post-trained weights for laptop, laptop is the domain, pt_rrc is the folder in ```run``` and 10 means run 10 times, 0 means gpu-0.

similarly,
```
bash run_rrc.sh rrc rest_pt rest pt_rrc 10 0
bash run_absa.sh ae laptop_pt laptop pt_ae 10 0
bash run_absa.sh ae rest_pt rest pt_ae 10 0
bash run_absa.sh asc laptop_pt laptop pt_asc 10 0
bash run_absa.sh asc rest_pt rest pt_asc 10 0
```

### post-training setup

This repository is still under development.

## TODO:
- [ ] data for RRC.
- [ ] preprocessed data
- [ ] pretained model


## Citation
If you find this work useful, please cite as following.
```
@inproceedings{xu_bert2019,
    title = "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis",
    author = "Xu, Hu  and Liu, Bing and Shu, Lei and Yu, Philip S.",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = jun,
    year = "2019",
}
```
