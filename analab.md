# Understanding Pre-trained BERT for Aspect-based Sentiment Analysis

code base for the paper: "Understanding Pre-trained BERT for Aspect-based Sentiment Analysis".
We perform an analysis of pre-trained BERT model on reviews for aspect-based sentiment analysis (ABSA). 


## Environment
The code is developed on Ubuntu 18.04 with Python 3.6.9(Anaconda), PyTorch 1.3 and Transformers 2.4.1.


## Usage

```cd transformers```  

edit `script/analyze.sh` for your conda environment. Then,   

`bash script/analyze.sh`

## TODO: 
Salient examples and heads.    
We found the following attention heads (`layer-head`) can be interesting in model `activebus/BERT-XD_Review`:  
`3-8`, `6-9`.

## Citation
If you find this work useful, please cite as following.
```
@inproceedings{xu_understanding2020,
    title = "Understanding Pre-trained BERT for Aspect-based Sentiment Analysis",
    author = "Xu, Hu and Shu, Lei and Yu, Philip S. and Liu, Bing",
    booktitle = "The 28th International Conference on Computational Linguistics",
    month = "Dec",
    year = "2020",
}
```
