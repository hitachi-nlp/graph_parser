# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from typing import Optional, List
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, LongformerConfig, LongformerModel

import amparse.common.util as util


class PLM(nn.Module):
    """
    A module for a pre-trained language model
    """
    def __init__(self, model_name_or_path: str, split_document: bool = False,
                 requires_grad: bool = True, attention_window: int = 512):
        super(PLM, self).__init__()

        if 'longformer' in model_name_or_path:
            config = util.from_pretrained(
                LongformerConfig,
                pretrained_model_name_or_path=model_name_or_path,
                output_hidden_states=True,
                output_attentions=True,
                add_pooling_layer=False,
                attention_window=attention_window
            )
            self.trm_model = util.from_pretrained(
                LongformerModel,
                pretrained_model_name_or_path=model_name_or_path,
                config=config,
            )
        else:
            config = util.from_pretrained(
                AutoConfig,
                pretrained_model_name_or_path=model_name_or_path,
                output_hidden_states=True,
                output_attentions=True
            )
            self.trm_model = util.from_pretrained(
                AutoModel,
                pretrained_model_name_or_path=model_name_or_path,
                config=config,
            )

        self.trm_model = self.trm_model.requires_grad_(requires_grad)
        self.requires_grad = requires_grad
        self.trm_hidden_size = self.trm_model.config.hidden_size
        self.split_document = split_document
        self.n_out = self.trm_hidden_size

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'n_layers={self.trm_model.config.num_hidden_layers}'
        s += f', split_document={self.split_document}'
        s += f', hidden_size={self.trm_hidden_size}'
        s += f', n_out={self.n_out}'
        if self.requires_grad:
            s += f', requires_grad={self.requires_grad}'
        s += ')'
        return s

    def _embed_tokens(self, subwords: torch.Tensor,
                      mask: torch.Tensor, global_mask: Optional[torch.Tensor] = None):
        if not self.requires_grad:
            self.trm_model.eval()

        if self.trm_model._get_name() == 'LongformerModel':
            assert subwords[0][0] == 0
            res = self.trm_model(input_ids=subwords,
                                 attention_mask=mask.long(),
                                 global_attention_mask=global_mask)
            x = res.last_hidden_state
        else:
            res = self.trm_model(input_ids=subwords, attention_mask=mask)
            x = res.last_hidden_state

        return x

    def forward(self, subwords: torch.Tensor,
                sentences: List[torch.Tensor], sentence_masks: List[torch.Tensor],
                sentence_global_masks: List[torch.Tensor],
                mask: torch.Tensor, global_mask: Optional[torch.Tensor] = None):

        if self.split_document:
            len_sentences = [s.shape[0] for s in sentences]
            x_sentence = self._embed_tokens(
                subwords=torch.cat(sentences, dim=0),
                mask=torch.cat(sentence_masks, dim=0),
                global_mask=torch.cat(sentence_global_masks, dim=0),
            )
            xs = torch.split(x_sentence, len_sentences, dim=0)  # batch x (seq, dim)
            xs = [_x.flatten(start_dim=0, end_dim=1)[_m.flatten()] for _x, _m in zip(xs, sentence_masks)]
            x = pad_sequence(xs, batch_first=True, padding_value=0.)  # (batch, seq, dim)
        else:
            x = self._embed_tokens(subwords=subwords, mask=mask, global_mask=global_mask)

        return x


class Encoder(nn.Module):
    """
    Encoder module based on a pre-trained language model
    """
    def __init__(self, model_name_or_path: str, split_document: bool = False,
                 dropout: float = .1, attention_window: int = 512):
        super(Encoder, self).__init__()
        parameter_names = []

        # Transformer-based pre-trained language model
        self.pretrained_transformer_encoder = PLM(
            model_name_or_path=model_name_or_path,
            split_document=split_document,
            requires_grad=True,
            attention_window=attention_window
        )

        self.embed_dropout = nn.Dropout(p=dropout)
        parameter_names.append('pretrained_transformer_encoder')
        self._parameter_names = parameter_names
        self._n_embed = self.pretrained_transformer_encoder.n_out

    @property
    def n_embed(self) -> int:
        return self._n_embed

    @property
    def parameter_names(self):
        return self._parameter_names

    def forward(self, token_ids: torch.Tensor, sentences: List[torch.Tensor], sentence_masks: List[torch.Tensor],
                sentence_global_masks: List[torch.Tensor],
                mask: torch.Tensor, global_mask: Optional[torch.Tensor] = None):

        # Embed the input sequence
        x = self.pretrained_transformer_encoder(
            subwords=token_ids,
            sentences=sentences,
            sentence_masks=sentence_masks,
            sentence_global_masks=sentence_global_masks,
            mask=mask,
            global_mask=global_mask
        )  # (batch, seq, dim)

        # Apply dropout
        x = self.embed_dropout(x)

        return {
            'embed': x,  # (batch, seq, dim)
        }

