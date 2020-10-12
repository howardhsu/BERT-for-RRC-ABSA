import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


class ModelUtil(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_config = AutoConfig.from_pretrained(self.model_name)
        model_config.output_attentions = True
        model_config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(self.model_name, config=model_config).to(0)

    def convert_to_inputs(self, example):
        tokens, labels, word_idxs = [], [], []
        for word_idx, (word, label) in enumerate(zip(example.text_a, example.label)):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            labels.extend([label]*len(word_tokens))
            word_idxs.extend([word_idx]*len(word_tokens))

        # pad with special token.
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        labels = ["O"] + labels + ["O"]
        word_idxs = [-100] + word_idxs + [-100]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {
            "tokens": tokens,
            "labels": labels,
            "word_idxs": word_idxs,
            "words": example.text_a,
            "word_labels": example.label,
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=0),
        }

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])

        return {
            "hidden_states": outputs[0].cpu(),
            "CLS": outputs[1].cpu(),
            "layered_hidden_states": tuple([output.cpu() for output in outputs[2]]),
            "layered_attentions": tuple([output.cpu() for output in outputs[3]])
        }
