import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, BertPredictionHeadTransform, BertOnlyMLMHead, BertEncoder, BertPooler, gelu
from transformers.modeling_roberta import RobertaLMHead, RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING 
from transformers.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel
from transformers import RobertaForMaskedLM

logger = logging.getLogger(__name__)


class BertForMaskedLMSelect(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.loss_fct = CrossEntropyLoss()  # -100 index = padding token; initialize once to speed up.

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        
        if masked_lm_labels is not None:
            # only compute select tokens to training to speed up.
            hidden_size = sequence_output.size(-1)
            masked_lm_labels = masked_lm_labels.reshape(-1)
            labels_mask = masked_lm_labels != -100

            selected_masked_lm_labels = masked_lm_labels[labels_mask]
            sequence_output = sequence_output.view(-1, hidden_size)
            selected_sequence_output = sequence_output.masked_select(labels_mask.unsqueeze(1)).view(-1, hidden_size)
            prediction_scores = self.cls(selected_sequence_output)
        else:
            prediction_scores = self.cls(sequence_output)


        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(prediction_scores, selected_masked_lm_labels)
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)
