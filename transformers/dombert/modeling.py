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


class BertForMaskedLMFixedLengthSelect(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')  # -100 index = padding token; initialize once to speed up.
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
        target_loss=None
    ):
        domain_ids, input_ids = input_ids[:,0], input_ids[:,1:]
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
            batch_size, seq_len, hidden_size = sequence_output.size()
            labels_mask = masked_lm_labels != -100
            selected_masked_lm_labels = masked_lm_labels[labels_mask]
            selected_sequence_output = sequence_output.masked_select(labels_mask.unsqueeze(-1)).view(-1, hidden_size)
            prediction_scores = self.cls(selected_sequence_output)
        else:
            prediction_scores = self.cls(sequence_output)


        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(prediction_scores, selected_masked_lm_labels).view(batch_size, -1)
            masked_lm_loss = torch.mean(masked_lm_loss, dim=-1)
            target_mask = domain_ids == 0
            masked_src_loss = masked_lm_loss[~target_mask]
            masked_target_loss = masked_lm_loss[target_mask]

            target_loss = target_loss if target_loss is not None else masked_target_loss.mean()

            allowed_src_ex_mask = masked_src_loss < target_loss
            allowed_masked_src_loss = masked_src_loss[allowed_src_ex_mask]
            if len(allowed_masked_src_loss) > 0:
                loss = torch.cat([masked_target_loss, allowed_masked_src_loss], dim=0).mean()
            else:
                loss = masked_target_loss.mean()
            outputs = (loss,masked_lm_loss.detach(),target_mask,domain_ids) + outputs

        if lm_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)
    
    
# the following heads are for learning domain embeddings on future use.
class DomBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.reducer = nn.Linear(config.hidden_size, config.hidden_size // 12)
        self.decoder = nn.Linear(config.hidden_size // 12, 4680, bias=False)

        # self.bias = nn.Parameter(torch.zeros(4680))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # hidden_states = self.transform(hidden_states)
        hidden_states = self.reducer(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DomBertDomainHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DomBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DomBertForMaskedLMCLSFixedLengthSelect(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.domain_cls = DomBertDomainHead(config)

        self.loss_fct = nn.CrossEntropyLoss()
        self.sim_fn = nn.CosineSimilarity(-1)
        self.eye = torch.eye(4680, device=0)
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
        target_loss=None
    ):
        domain_ids, input_ids = input_ids[:,0], input_ids[:,1:]
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
            batch_size, seq_len, hidden_size = sequence_output.size()
            labels_mask = masked_lm_labels != -100
            selected_masked_lm_labels = masked_lm_labels[labels_mask]
            selected_sequence_output = sequence_output.masked_select(labels_mask.unsqueeze(-1)).view(-1, hidden_size)
            prediction_scores = self.cls(selected_sequence_output)
            domain_prediction_scores = self.domain_cls(outputs[1])
        else:
            prediction_scores = self.cls(sequence_output)
            domain_prediction_scores = self.domain_cls(outputs[1])


        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(prediction_scores, selected_masked_lm_labels)
            domain_cls_loss = self.loss_fct(domain_prediction_scores, domain_ids)
            reg = self.sim_fn(self.domain_cls.predictions.decoder.weight.unsqueeze(1), self.domain_cls.predictions.decoder.weight) - self.eye
            reg = torch.mul(reg, reg).mean()
            loss = 0.9 * masked_lm_loss + 0.1 * domain_cls_loss + reg
            outputs = (loss,masked_lm_loss,domain_cls_loss) + outputs

        if lm_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)
