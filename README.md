# BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis
code for our NAACL 2019 paper "[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](http://arxiv.org/abs/1904.02232)".

***feedbacks are welcomed for missing instructions ***

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

## Fine-tuning setup

step1: make 2 folders for post-training and fine-tuning.
```
mkdir -p pt_model ; mkdir -p run
```
step2: place post-trained BERTs into ```pt_model/```. Our post-trained Laptop weights can be download [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing) and restaurant [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing). You are free to download other BERT weights into this folder(e.g., bert-base). Make sure to add an entry into ```src/modelconfig.py```.

step3: make 3 folders for 3 tasks: 

place fine-tuning data to each respective folder: ```rrc/, ae/, asc/```. A pre-processed data in json format can be found [here](https://drive.google.com/file/d/1NGH5bqzEx6aDlYJ7O3hepZF4i_p4iMR8/view?usp=sharing).

step4: fire a fine-tuning from a BERT weight, e.g.
```
cd script
bash run_rrc.sh rrc laptop_pt laptop pt_rrc 10 0
```
Here rrc is the task to run, laptop_pt is the post-trained weights for laptop, laptop is the domain, pt_rrc is the fine-tuned folder in ```run/```, 10 means run 10 times and 0 means use gpu-0.

similarly,
```
bash run_rrc.sh rrc rest_pt rest pt_rrc 10 0
bash run_absa.sh ae laptop_pt laptop pt_ae 10 0
bash run_absa.sh ae rest_pt rest pt_ae 10 0
bash run_absa.sh asc laptop_pt laptop pt_asc 10 0
bash run_absa.sh asc rest_pt rest pt_asc 10 0
```
step5: evaluation

RRC: download SQuAD 1.1 evaluation script ([e.g.](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py) ) to ```eval/```.

AE: place official evaluation .jar files as ```eval/A.jar``` and ```eval/eval.jar```.
place testing xml files as (the step 4 of [this](https://github.com/howardhsu/DE-CNN) has a similar setup)
```
ae/official_data/Laptops_Test_Gold.xml
ae/official_data/Laptops_Test_Data_PhaseA.xml
ae/official_data/EN_REST_SB1_TEST.xml.gold
ae/official_data/EN_REST_SB1_TEST.xml.A
```
ASC: built-in as part of ```eval/eval.ipynb```

open ```result.ipynb``` and run as you wish

## Post-training setup

Assume you are on the root folder of this repository.

step1: domain post-training data:

Download pre-processed data from [here](https://drive.google.com/file/d/1s_-OFh7qBWmzA9ths3mf7IYN1P2WrXMo/view?usp=sharing) and place it under the root folder.

Alternatively you can create your own data. For example for laptop, ```mkdir -p domain_corpus ; mkdir -p domain_corpus/laptop ; mkdir -p domain_corpus/raw```. Place newline separated reviews laptop.txt as ```domain_corpus/raw/laptop.txt```
The script in step3 will detect if your ```.npz``` is unavailable under domain_corpus/laptop then it will start to preprocess ```domain_corpus/raw/laptop.txt``` first.

step2: MRC post-training data:

Place and rename [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) (for consistency with other review-based tasks) as ```squad/train.json```.

step3: preprocessing and post-training

```
bash pt.sh laptop 5 70000 0
bash pt.sh rest 1 140000 0
```


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
