# Migrating to Transformers
The hugging face `pytorch-pretrained-bert` has gone through significant changes after we finish the research on our [NAACL paper](https://www.aclweb.org/anthology/N19-1242.pdf)".
Now it's called [transformers](https://huggingface.co/transformers/), because it supports more cross-library operations such as Tensorflow.
The API calls of migration can be found at [here](https://huggingface.co/transformers/migration.html).

Our released model is trained from the format defined in `pytorch-pretrained-bert`, **NOT** `transformers`.
Migrating the trained model to `transformers` is not that hard, here is how.

## Instructions
We upload a template of `bert-base-uncased` under `pt/bert-case-uncased-template`, which contains the major changes of `transformers`.
After you download our post-trained model, just copy these files into the same model directory as `pytorch_model.bin`.

Loading the post-trained model the same as other models in `transformers`:

```python
import torch
from transformers import BertModel

model = BertModel.from_pretrained("pt/laptop_pt")

```
