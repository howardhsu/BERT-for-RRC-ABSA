# ReviewBERT on Transformers
code base of review language modeling for aspect-based sentiment analysis.

## Major Changes (from NAACL version)
Adapt the NAACL models to [transformers](https://huggingface.co/transformers/) library, which is updated from `pytorch-pretrained-bert`.  
Add more tasks such as [E2E-ABSA](https://github.com/lixin4ever/E2E-TBSA) (more new tasks in ABSA are expected in future).  
A more efficient implementation of `BERTForMaskedLM`, with early apply of labels (EAL) to speed up the training by avoiding wasting time and memory on unmasked tokens.
More fine-tuned (or post-trained) cross-domain language models (XDLM), such as `BERT-Review`, `BERT-XD` (one model to solve all domains but you may lose domain-specific knowledge).  
For aspect extraciton, subwords except the first one (tokens begin with `##` in BERT) is an invalid in labeling space (`-100` in pytorch), whereas NAACL paper treats them as `I`. We observe slightly drop of performance :-(, but seems NER people do sequence labeling [this way](https://github.com/huggingface/transformers/tree/master/examples/ner).  
And, of course, a more OO-style experimental environments.  

## Usage

```cd transformers```
For end task fine-tuning:  
```bash script/run_ft.sh```
```python src/report.py```
For LM post-training:
```bash script/run_pt.sh```

## Migrating models (from NAACL version) to Transformers

The hugging face `pytorch-pretrained-bert` has gone through significant changes after we finish the research on our [NAACL paper](https://www.aclweb.org/anthology/N19-1242.pdf)".
Now it's called [transformers](https://huggingface.co/transformers/), because it supports more cross-library operations such as Tensorflow.
The API calls of migration can be found at [here](https://huggingface.co/transformers/migration.html).

Our released model is trained from the format defined in `pytorch-pretrained-bert`, **NOT** `transformers`.
Migrating the trained model to `transformers` is not that hard, here is how.

### Instructions
We upload a template of `bert-base-uncased` under `pt/bert-case-uncased-template`, which contains the major changes of `transformers`.
After you download our post-trained model, just copy these files into the same model directory as `pytorch_model.bin`.

Loading the post-trained model the same as other models in `transformers`:

```python
import torch
from transformers import BertModel

model = BertModel.from_pretrained("pt/laptop_pt")

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