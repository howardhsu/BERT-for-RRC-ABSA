# ReviewBERT on Transformers

code base of review language modeling for aspect-based sentiment analysis.

## Major Changes (from NAACL version)
Adapt the NAACL models to [transformers](https://huggingface.co/transformers/) library, which is updated from `pytorch-pretrained-bert`.  
Add more tasks such as [E2E-ABSA](https://github.com/lixin4ever/E2E-TBSA) (more new tasks in ABSA are expected in future).  
A more efficient implementation of `BERTForMaskedLM` as `BertForMaskedLMSelect`, with early apply of labels (EAL) to speed up the training by avoiding wasting time and memory on unmasked tokens.
More fine-tuned (or post-trained) cross-domain language models (XDLM), such as `BERT_Review`, `BERT-XD_Review` (one model to solve all domains but you may lose domain-specific knowledge).  
For aspect extraciton, subwords except the first one (tokens begin with `##` in BERT) is an invalid in labeling space (`-100` in pytorch), whereas NAACL paper treats them as `I`. We observe slightly drop of performance :-(, but seems NER people do sequence labeling [this way](https://github.com/huggingface/transformers/tree/master/examples/ner).  
And, of course, a more OO-style experimental environments.  

## Environment
The code is tested on Ubuntu 18.04 with Python 3.6.9(Anaconda), PyTorch 1.3 (apex 0.1) and Transformers 2.4.1.


## Usage

```cd transformers```  

For aspect extraction, you may need to create a java link, e.g., `ln -s /usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java java`

Example bash files are here, you may need to (1) setup your conda environment; (2) specify your number of GPUSs; (3) change batch size and accumulation steps to fit your GPU memory.  

For end task fine-tuning:  
```bash script/run_ft.sh```
```python src/report.py```

For LM post-training:
```bash script/run_pt.sh```

## Available Models

Available models can be found at [model_cards.md](model_cards.md).

### Instructions
Loading the post-trained weights are as simple as, e.g., 

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("activebus/BERT_Review")
model = AutoModel.from_pretrained("activebus/BERT_Review")

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
