import torch
import os
import logging
from collections import defaultdict
from reviewlab import absa_data_util

from .modelutil import ModelUtil
from .plot import LineAttnPlot, AspectLineAttnPlot, TSNEPlot, TypedTSNEPlot

logger = logging.getLogger(__name__)


class Job(object):
    def __init__(self, job_config, data_config):
        self.job_config = job_config
        self.data_config = data_config
        self.model = ModelUtil(job_config.model_name)

    def _load_examples(self, data_config):
        processor = getattr(absa_data_util, data_config.task.upper()+"Processor")(data_config)
        return processor.get_dev_examples()

    def run(self):
        raise NotImplementedError


class AttentionJob(Job):
    def run(self):
        examples = self._load_examples(self.data_config)
        for example_idx, example in enumerate(examples):
            if example.guid not in ["dev-87", "dev-35", "dev-52"]:
                continue

            inputs = self.model.convert_to_inputs(example)
            outputs = self.model.forward(inputs)
            for layer in self.job_config.layer:
                headed_attentions = outputs["layered_attentions"][layer][0]
                for head_idx, attentions in enumerate(headed_attentions):
                    labels = [label[2:] if label!="O" else "O" for label in inputs["labels"]]
                    path = os.path.join(
                        self.job_config.out_dir,
                        self.job_config.job_name,
                        self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name,
                        f"layer{layer}"
                        )
                    os.makedirs(path, exist_ok=True)
                    fn = os.path.join(path, f"{example.guid}-{head_idx}.png")
                    plot = AspectLineAttnPlot(fn)
                    plot(inputs["tokens"], labels, attentions.numpy(), {"POS": "g", "NEG": "r", "NEU": "b", "O": "indigo"})


class LatentAspectJob(Job):
    def _build_embs(self, examples):
        embeddings, phrases, labels = [], [], []
        for example_idx, example in enumerate(examples):
            inputs = self.model.convert_to_inputs(example)
            outputs = self.model.forward(inputs)

            for token_idx, token in enumerate(inputs["tokens"]):
                label = inputs["labels"][token_idx]
                if label != "O":
                    if len(phrases) == 0 or type(phrases[-1]) == str:
                        phrases.append([])
                        embeddings.append([])

                    if token.startswith("##"):
                        phrases[-1][-1] = phrases[-1][-1]+token[2:]
                    else:
                        phrases[-1].append(token)
                    embeddings[-1].append(outputs["hidden_states"][:,token_idx])
                elif len(phrases) > 0 and type(phrases[-1]) == list:
                    phrases[-1] = " ".join(phrases[-1])
                    labels.append(prev_label)
                    embeddings[-1] = torch.mean(torch.cat(embeddings[-1], dim=0), dim=0, keepdim=True)
                prev_label = label

            if len(phrases) > 0 and type(phrases[-1]) == list:
                phrases[-1] = " ".join(phrases[-1])
                labels.append(prev_label)
                embeddings[-1] = torch.mean(torch.cat(embeddings[-1], dim=0), dim=0, keepdim=True)

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings, phrases, labels

    def run(self):
        examples = self._load_examples(self.data_config)
        embeddings, phrases, labels = self._build_embs(examples)
        labels = [label[2:] for label in labels]
        path = os.path.join(
            self.job_config.out_dir,
            self.job_config.job_name,
            self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name
        )
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"LatentAspect.png")
        plot = TypedTSNEPlot(fn)
        plot(embeddings, phrases, labels, ["POS", "NEG", "NEU"], {"POS": "g", "NEG": "r", "NEU": "b"})


class LatentANAJob(Job):
    def _build_embs(self, examples):
        embeddings, phrases, labels = [], [], []
        for example_idx, example in enumerate(examples):
            #if example_idx >= 10:
            #    break

            inputs = self.model.convert_to_inputs(example)
            if not any([label!="O" for label in inputs["labels"]]):
                continue
            outputs = self.model.forward(inputs)

            for token_idx, token in enumerate(inputs["tokens"]):
                label = inputs["labels"][token_idx]
                embeddings.append(outputs["hidden_states"][:,token_idx])
                phrases.append(token)
                label = "aspect" if label != "O" else "non-aspect"
                labels.append(label)

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings, phrases, labels

    def run(self):
        examples = self._load_examples(self.data_config)
        embeddings, phrases, labels = self._build_embs(examples)
        path = os.path.join(
            self.job_config.out_dir,
            self.job_config.job_name,
            self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name
        )
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"LatentANA.png")
        plot = TypedTSNEPlot(fn)
        plot(embeddings, phrases, labels, ["aspect", "non-aspect"], {"aspect": "c", "non-aspect": "m"}, False)


class MultiDomainAspectJob(LatentAspectJob):

    def run(self):
        t_embeddings, t_phrases, t_labels = [], [], []
        for data_config in self.data_config:
            examples = self._load_examples(data_config)
            embeddings, phrases, labels = self._build_embs(examples)
            phrases = [data_config.domain[0].upper()+":"+phrase for phrase in phrases]
            labels = [label[2:] for label in labels]
            select = -1
            t_embeddings.append(embeddings[:select])
            t_phrases.extend(phrases[:select])
            t_labels.extend(labels[:select])
        t_embeddings = torch.cat(t_embeddings, dim=0)

        path = os.path.join(
            self.job_config.out_dir,
            self.job_config.job_name,
            self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name
        )
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"LatentAspect.png")
        plot = TypedTSNEPlot(fn)
        plot(
            t_embeddings, t_phrases, t_labels,
            ["POS", "NEG", "NEU"],
            {"POS": "g", "NEG": "r", "NEU": "b"},
            False
        )


class DomainAspectJob(LatentAspectJob):

    def run(self):
        t_embeddings, t_phrases, t_labels = [], [], []
        for data_config in self.data_config:
            examples = self._load_examples(data_config)
            embeddings, phrases, labels = self._build_embs(examples)
            phrases = [data_config.domain[0].upper()+":"+phrase for phrase in phrases]
            labels = [data_config.domain] * len(phrases)
            t_embeddings.append(embeddings)
            t_phrases.extend(phrases)
            t_labels.extend(labels)
        t_embeddings = torch.cat(t_embeddings, dim=0)

        path = os.path.join(
            self.job_config.out_dir,
            self.job_config.job_name,
            self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name
        )
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"dvd.png")
        plot = TypedTSNEPlot(fn)
        plot(
            t_embeddings, t_phrases, t_labels,
            ["laptop", "rest"],
            {"laptop": "g", "rest": "r"},
            False
        )


class LatentOTJob(LatentAspectJob):
    def run(self):
        with open("data/ft/ot/14/laptop/antt.txt") as f:
            lines = f.readlines()

        examples = []
        for line_id  in range(0, len(lines), 3):
            text_a = lines[line_id].strip().split(" ")
            label = lines[line_id+1].strip().split(" ")
            tag, ots = label[0], label[1:]
            label = [tag if token in ots else "O" for token in text_a]
            examples.append(absa_data_util.InputExample(guid=f"{line_id}", text_a=text_a, label=label))

        embeddings, phrases, labels = self._build_embs(examples)

        labels = [label[2:] for label in labels]
        path = os.path.join(
            self.job_config.out_dir,
            self.job_config.job_name,
            self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name
        )
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"LatentOT.png")
        plot = TypedTSNEPlot(fn)
        plot(embeddings[:-30], phrases[:-30], labels[:-30], ["POS", "NEG"], {"POS": "g", "NEG": "r"})


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,f1_score

from .plot import NeuronPlot

class SentiDiscoveryJob(LatentAspectJob):
    def _load_train_examples(self, data_config):
        processor = getattr(absa_data_util, data_config.task.upper()+"Processor")(data_config)
        return processor.get_train_examples()

    def _load_test_examples(self, data_config):
        processor = getattr(absa_data_util, data_config.task.upper()+"Processor")(data_config)
        return processor.get_test_examples()

    def _train(self, X_train, y_train, batch_size = 128):
        # some preparation.

        log_reg = SGDClassifier(loss='log', penalty='l1', alpha=0.001)
        for _ in range(10):
            for i in range(0, len(X_train), batch_size):
                current_X_train = X_train[i:i+batch_size]
                current_y_train = y_train[i:i+batch_size]
                log_reg.partial_fit(current_X_train, current_y_train, classes=[0,1])
        return log_reg

    def _filter_neu(self, embeddings, labels):
        ex_idxs = [ex_idx for ex_idx, label in enumerate(labels) if label != "T-NEU"]
        embeddings = embeddings[ex_idxs]
        idxed_labels = [0 if labels[ex_idx] == "T-POS" else 1 for ex_idx in ex_idxs]
        return embeddings, idxed_labels

    def run(self):
        train_examples = self._load_train_examples(self.data_config)
        embeddings, phrases, labels = self._build_embs(train_examples)

        train_embeddings, train_labels = self._filter_neu(embeddings, labels)

        log_reg = self._train(train_embeddings, train_labels)

        test_examples = self._load_test_examples(self.data_config)
        embeddings, phrases, labels = self._build_embs(test_examples)
        test_embeddings, test_labels = self._filter_neu(embeddings, labels)

        logreg_predictions = log_reg.predict(train_embeddings)
        print(accuracy_score(train_labels, logreg_predictions))
        print(f1_score(train_labels, logreg_predictions))

        logreg_predictions = log_reg.predict(test_embeddings)
        print(accuracy_score(test_labels, logreg_predictions))
        print(f1_score(test_labels, logreg_predictions))

        path = os.path.join(
            self.job_config.out_dir,
            self.job_config.job_name,
            self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name
        )
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"SentiDiscovery.png")

        plot = NeuronPlot(fn)
        plot(log_reg)

import numpy as np

class AspectDiscoveryJob(LatentANAJob):
    def _load_train_examples(self, data_configs):
        examples = []
        for data_config in data_configs:
            processor = getattr(absa_data_util, data_config.task.upper()+"Processor")(data_config)
            examples.extend(processor.get_train_examples())
        return examples

    def _load_test_examples(self, data_configs):
        examples = []
        for data_config in data_configs:
            processor = getattr(absa_data_util, data_config.task.upper()+"Processor")(data_config)
            examples.extend(processor.get_test_examples())
        return examples

    def _train(self, X_train, y_train, batch_size = 128):
        # some preparation.

        log_reg = SGDClassifier(loss='log', penalty='l1', alpha=0.001)
        for _ in range(5):
            for i in range(0, len(X_train), batch_size):
                current_X_train = X_train[i:i+batch_size]
                current_y_train = y_train[i:i+batch_size]
                log_reg.partial_fit(current_X_train, current_y_train, classes=[0,1])
        return log_reg

    def _label_to_idx(self, labels):
        idxed_labels = [1 if label == "aspect" else 0 for label in labels]
        return idxed_labels

    def run(self):
        train_examples = self._load_train_examples(self.data_config)
        train_embeddings, train_phrases, train_labels = self._build_embs(train_examples)
        train_labels = self._label_to_idx(train_labels)
        log_reg = self._train(train_embeddings, train_labels)

        test_examples = self._load_test_examples(self.data_config)
        test_embeddings, test_phrases, test_labels = self._build_embs(test_examples)
        test_labels = self._label_to_idx(test_labels)

        logreg_predictions = log_reg.predict(train_embeddings)
        print(accuracy_score(train_labels, logreg_predictions))
        print(f1_score(train_labels, logreg_predictions))

        logreg_predictions = log_reg.predict(test_embeddings)
        print(accuracy_score(test_labels, logreg_predictions))
        print(f1_score(test_labels, logreg_predictions))

        sorted_coeff_indices = [i for i in sorted(enumerate(log_reg.coef_[0].T), key=lambda x:np.abs(x[1]), reverse=True)]

        print(sorted_coeff_indices[:5])
        sentiment_neuron_index = sorted_coeff_indices[0][0]
        sentiment_indicator_test = test_embeddings[:, sentiment_neuron_index]

        print(sentiment_indicator_test, sentiment_indicator_test.mean(), sentiment_indicator_test.std())
        if sorted_coeff_indices[0][1] > 0:
            logreg_predictions = sentiment_indicator_test > 0.0
        else:
            logreg_predictions = sentiment_indicator_test < 0.0

        print(f1_score(test_labels, logreg_predictions.tolist()))


        path = os.path.join(
            self.job_config.out_dir,
            self.job_config.job_name,
            self.job_config.model_name.split("/")[1] if "/" in self.job_config.model_name else self.job_config.model_name
        )
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f"AspectDiscovery.png")

        plot = NeuronPlot(fn)
        plot(log_reg)
