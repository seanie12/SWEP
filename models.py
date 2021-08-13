from math import sqrt

import torch
import torch.nn as nn
from transformers import BertForQuestionAnswering, ElectraForQuestionAnswering
import torch.nn.functional as F


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        return kl.mean(dim=0)


class VariationalBert(nn.Module):
    def __init__(self, args):
        super(VariationalBert, self).__init__()
        self.model_name = args.bert_model
        self.bert_model = BertForQuestionAnswering.from_pretrained(
            args.bert_model)
        self.noise_net = nn.Sequential(nn.Linear(args.hidden_size,
                                                 args.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(args.hidden_size,
                                                 args.hidden_size * 2))
        config = self.bert_model.config
        self.dropout = config.hidden_dropout_prob  # 0.1

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None):

        if start_positions is not None and end_positions is not None:
            embeddings = self.bert_model.get_input_embeddings()
            encoder = self.bert_model.bert
            with torch.no_grad():
                encoder_inputs = {"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "token_type_ids": token_type_ids}

                outputs = encoder(**encoder_inputs)
                hiddens = outputs[0]

            mask = attention_mask.view(-1)
            indices = (mask == 1)
            mu_logvar = self.noise_net(hiddens)
            mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)
            zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
            noise = zs

            prior_mu = torch.ones_like(mu)
            # If p < 0.5, sqrt makes variance the larger
            prior_var = torch.ones_like(mu) * sqrt(self.dropout / (1-self.dropout))
            prior_logvar = torch.log(prior_var)

            kl_criterion = GaussianKLLoss()
            h = hiddens.size(-1)
            _mu = mu.view(-1, h)[indices]
            _log_var = log_var.view(-1, h)[indices]
            _prior_mu = prior_mu.view(-1, h)[indices]
            _prior_logvar = prior_logvar.view(-1, h)[indices]

            kl = kl_criterion(_mu, _log_var, _prior_mu, _prior_logvar)

            inputs_embeds = embeddings(input_ids)
            inputs = {"inputs_embeds": inputs_embeds * noise,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "start_positions": start_positions,
                      "end_positions": end_positions}

            noise_outputs = self.bert_model(**inputs)
            noise_loss = noise_outputs[0]

            new_inputs = {"inputs_embeds": inputs_embeds,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          "start_positions": start_positions,
                          "end_positions": end_positions}

            outputs = self.bert_model(**new_inputs)
            nll = outputs[0]
            loss = 0.5 * (nll + noise_loss)
            return (loss, kl)

        else:
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "start_positions": start_positions,
                      "end_positions": end_positions}

            outputs = self.bert_model(**inputs)
            return outputs


class VariationalElectra(nn.Module):
    def __init__(self, args):
        super(VariationalElectra, self).__init__()
        self.model_name = args.bert_model
        self.bert_model = ElectraForQuestionAnswering.from_pretrained(
            args.bert_model)

        config = self.bert_model.config
        hidden_size = config.hidden_size
        embedding_size = config.embedding_size
        self.dropout = config.hidden_dropout_prob
        self.embedding_size = embedding_size
        self.noise_net = nn.Sequential(nn.Linear(hidden_size, embedding_size),
                                       nn.ReLU(),
                                       nn.Dropout(args.dropout), # 0.15
                                       nn.Linear(embedding_size, embedding_size * 2))

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None):

        if start_positions is not None and end_positions is not None:
            embeddings = self.bert_model.get_input_embeddings()
            encoder = self.bert_model.electra
            with torch.no_grad():
                encoder_inputs = {"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "token_type_ids": token_type_ids}

                outputs = encoder(**encoder_inputs)
                hiddens = outputs[0]

            mask = attention_mask.view(-1)
            indices = (mask == 1)
            mu_logvar = self.noise_net(hiddens)
            mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)
            zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
            noise = zs

            prior_mu = torch.ones_like(mu)
            # If p < 0.5, sqrt makes variance the larger
            prior_var = torch.ones_like(
                mu) * sqrt(self.dropout / (1-self.dropout))
            prior_logvar = torch.log(prior_var)

            kl_criterion = GaussianKLLoss()
            h = self.embedding_size
            _mu = mu.contiguous().view(-1, h)[indices]
            _log_var = log_var.contiguous().view(-1, h)[indices]
            _prior_mu = prior_mu.contiguous().view(-1, h)[indices]
            _prior_logvar = prior_logvar.contiguous().view(-1, h)[indices]

            kl = kl_criterion(_mu, _log_var, _prior_mu, _prior_logvar)

            inputs_embeds = embeddings(input_ids)
            inputs = {"inputs_embeds": inputs_embeds * noise,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "start_positions": start_positions,
                      "end_positions": end_positions}

            noise_outputs = self.bert_model(**inputs)
            noise_loss = noise_outputs[0]

            new_inputs = {"inputs_embeds": inputs_embeds,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          "start_positions": start_positions,
                          "end_positions": end_positions}

            outputs = self.bert_model(**new_inputs)
            nll = outputs[0]
            loss = 0.5 * (nll + noise_loss)
            return (loss, kl)

        else:
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "start_positions": start_positions,
                      "end_positions": end_positions}

            outputs = self.bert_model(**inputs)
            return outputs
