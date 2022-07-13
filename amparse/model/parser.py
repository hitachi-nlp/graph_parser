# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import (AdamW,
                                       get_linear_schedule_with_warmup)

from amparse.common import params as P
from amparse.common import util as util
from amparse.common.parser_config import AMParserConfig
from amparse.loader.fields import InputTargetField
from amparse.loader.numericalizer import AMNumericalizer
from amparse.loader.vocab import AMVocab
from amparse.model.biaffine import Biaffine
from amparse.model.encoder import Encoder
from amparse.model.mlp import MLP


class AMParser(nn.Module):
    """
    Class for the parser module
    """
    def __init__(self, config: AMParserConfig, numericalizer: AMNumericalizer):
        super(AMParser, self).__init__()
        self.config: AMParserConfig = config
        self.numericalizer = numericalizer

        # Vocabularies
        self.field: InputTargetField = numericalizer.field_by_name('input_target')
        self.vocab = self.field.vocab
        self.fw_vocab: AMVocab = self.field.fw_vocab
        self.fw_2_proposition_vocab: Dict[str, AMVocab] = self.field.fw_2_proposition_vocab
        self.fw_2_edge_vocab: Dict[str, AMVocab] = self.field.fw_2_edge_vocab
        self.bio_vocab: AMVocab = self.field.bio_vocab
        self.pad_id = self.field.vocab.token2id(P.PAD)

        # [Input] Encoder
        self.encoder: Encoder = Encoder(
            model_name_or_path=self.field.model_name_or_path,
            split_document=self.config.split_document,
            dropout=config.embed_dropout,
            attention_window=config.attention_window,
        )
        dim = self.encoder.n_embed
        dim_mlp = config.dim_mlp

        # [Output] Span identification (BIO)
        self.bio_out = nn.ModuleDict({
            fw: nn.Sequential(MLP(n_in=dim, n_hidden=dim_mlp, dropout=config.mlp_dropout),
                              nn.Linear(dim_mlp, self.bio_vocab.max_id + 1))
            for fw in self.field.fw_vocab.tokens
        })

        # [Output] Component classification
        self.proposition_out = nn.ModuleDict({
            fw: nn.Sequential(MLP(n_in=dim, n_hidden=dim_mlp, dropout=config.mlp_dropout),
                              nn.Linear(dim_mlp, self.fw_2_proposition_vocab[fw].max_id + 1))
            for fw in self.field.fw_vocab.tokens
        })

        # [Output] Edge link identification
        self.arc_mlp_h = nn.ModuleDict({
            fw: MLP(n_in=dim, n_hidden=config.dim_biaffine, dropout=config.mlp_dropout)
            for fw in self.field.fw_vocab.tokens
        })
        self.arc_mlp_d = nn.ModuleDict({
            fw: MLP(n_in=dim, n_hidden=config.dim_biaffine, dropout=config.mlp_dropout)
            for fw in self.field.fw_vocab.tokens
        })
        self.arc_biaff = nn.ModuleDict({
            fw: Biaffine(n_in=config.dim_biaffine, n_out=1, bias_x=True, bias_y=True)
            for fw in self.field.fw_vocab.tokens
        })

        # [Output] Edge label classification
        self.rel_mlp_h = nn.ModuleDict({
            fw: MLP(n_in=dim, n_hidden=config.dim_biaffine, dropout=config.mlp_dropout)
            for fw in self.field.fw_vocab.tokens
        })
        self.rel_mlp_d = nn.ModuleDict({
            fw: MLP(n_in=dim, n_hidden=config.dim_biaffine, dropout=config.mlp_dropout)
            for fw in self.field.fw_vocab.tokens
        })
        self.rel_biaff = nn.ModuleDict({
            fw: Biaffine(n_in=config.dim_biaffine, n_out=self.fw_2_edge_vocab[fw].max_id + 1, bias_x=True, bias_y=True)
            for fw in self.field.fw_vocab.tokens
        })

    def __repr__(self):
        return super().__repr__()

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Do forward and get loss on the network

        Parameters
        ----------
        batch : Dict
            The batched data

        Returns
        ----------
        loss_dict : Dict[str, torch.Tensor]
            The dictionary of loss values
        """
        loss_dict = self._get_loss(batch=batch['batch'])
        return loss_dict

    def predict(self, batch: Dict) -> List[Dict]:
        """
        Predict a batch

        Parameters
        ----------
        batch : Dict
            The batched data

        Returns
        ----------
        pred_data : List[Dict]
            The prediction results (batch size x dict)
        """
        from amparse.model import parser_generator as generator
        with torch.no_grad():
            # Passing only one model, do not use ensemble
            pred_data = generator.ensemble(
                models=[self],
                batch=batch['batch'],
                oracle_span=self.config.evaluate_with_oracle_span
            )
        return pred_data

    def plm_encode(self, batch) -> torch.Tensor:
        """
        Encode the input text by a pre-trained language model (PLM) and get the embeddings

        Parameters
        ----------
        batch : Dict
            The batched data

        Returns
        ----------
        embed : torch.Tensor
            The embed representation of the input text (batch, seq, dim)
        """
        input_target = batch['input_target']
        # Encode input text
        res = self.encoder(
            token_ids=input_target['token_ids'],
            sentences=input_target['sentences'],
            sentence_masks=input_target['sentence_masks'],
            sentence_global_masks=input_target['sentence_global_masks'],
            mask=input_target['mask'],
            global_mask=input_target['global_mask'],
        )
        return res['embed']  # (batch, seq, dim)

    def span_output(self, exs: torch.Tensor, fw: str) -> torch.Tensor:
        # exs: (batch, seq, dim)
        # mask: (batch, seq)
        # fw: framework name
        o = self.bio_out[fw](exs)  # (batch, seq, n_out)
        return o

    def span_predict(self, exs: torch.Tensor, span_out: torch.Tensor, mask: torch.Tensor,
                     gold: Optional[torch.Tensor], oracle: bool = False) -> Dict[str, torch.Tensor or List[slice]]:
        # exs: (batch, seq, dim)
        # span_out: (batch, seq, n_out)
        # mask: (batch, seq)
        # gold: (batch, seq)
        if oracle or (self.training and gold is not None):
            # Use gold anchors when training
            arg_outs = [g[g != self.vocab.token2id(P.PAD)] for g in gold]
        else:
            arg_outs = [torch.argmax(o[m], dim=-1) for o, m in zip(span_out, mask)]

        span_exs, span_mask, top_mask, spans = [], [], [], []
        for ex, outs, m in zip(exs, arg_outs, mask):
            assert len(outs) != 0

            # Predict spans
            outs = outs.cpu().numpy().tolist()
            str_bio = self.bio_vocab.ids2tokens(outs)
            str_bio = util.modify_bio_sequence(str_bio)
            bio_spans = util.bio_sequence_to_spans(str_bio)  # BIO sequence -> span slices

            if bio_spans:
                # Apply the mean pooling to obtain span representation
                span_ex = torch.stack([torch.mean(ex[m][s], dim=0) for s in bio_spans])  # (n_span, dim)
                # Append <s> repr. into the last of span representations (this repr. is for the imaginary top)
                span_ex = torch.cat((span_ex, ex[0].unsqueeze(0)), dim=0)  # (n_span + 1, dim)

                assert torch.equal(
                    span_ex[0],
                    torch.mean(ex[m][bio_spans[0].start: bio_spans[0].stop], dim=0)
                 )
            else:
                span_ex = ex[0].unsqueeze(0)  # (n_span + 1, dim)

            spans.append(bio_spans)
            span_exs.append(span_ex)

            span_m = torch.ones(size=(span_ex.shape[0],)).type(torch.bool)
            top_m = span_m.clone()
            span_m[-1] = False
            top_m[:-1] = False

            assert sum(span_m) + sum(top_m) == len(bio_spans) + 1

            span_mask.append(span_m)
            top_mask.append(top_m)

        span_exs = pad_sequence(span_exs, batch_first=True, padding_value=0)  # (batch, n_span + 1, dim)
        span_mask = pad_sequence(span_mask, batch_first=True, padding_value=False)  # (batch + 1, n_span)
        top_mask = pad_sequence(top_mask, batch_first=True, padding_value=False)  # (batch + 1, n_span)

        assert torch.equal(
            span_exs[0][top_mask[0]][0],
            exs[0][0]
        )

        return {
            'span_exs': span_exs,  # (batch, n_span + 1, dim)
            'span_mask': span_mask,  # (batch, n_span + 1)
            'top_mask': top_mask,  # (batch, n_span + 1)
            'span': spans,  # batch x n_span x slice
        }

    def proposition_output(self, span_exs: torch.Tensor, fw: str) -> torch.Tensor:
        # span_exs: (batch, n_span + 1, dim)
        # fw: framework name
        o = self.proposition_out[fw](span_exs)  # (batch, n_span + 1, n_out)
        return o

    def proposition_predict(self, prop_out: torch.Tensor, span_mask: torch.Tensor, fw: str) -> List[str]:
        # prop_out: (batch, n_span + 1, n_out)
        # span_mask: (batch, n_span)
        # fw: framework name
        pred = [o[m].cpu().detach().numpy() for o, m in zip(prop_out, span_mask)]  # batch x (n_span, n_out)
        pred = [self.fw_2_proposition_vocab[fw].ids2tokens(np.argmax(o, axis=1)) for o in pred]
        return pred

    def edge_output(self, span_exs: torch.Tensor, fw: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # exs: (batch, seq, dim)
        # span_exs: (batch, n_span + 1, dim)
        # fw: framework name
        h_arc = self.arc_mlp_h[fw](span_exs)  # (batch, n_span + 1, dim)
        d_arc = self.arc_mlp_d[fw](span_exs)  # (batch, n_span + 1, dim)
        o_arc = self.arc_biaff[fw](h_arc, d_arc)  # (batch, n_span + 1, n_span + 1)

        h_rel = self.rel_mlp_h[fw](span_exs)  # (batch, n_span + 1, dim)
        d_rel = self.rel_mlp_d[fw](span_exs)  # (batch, n_span + 1, dim)
        o_rel = self.rel_biaff[fw](h_rel, d_rel)
        if len(o_rel.shape) == 4:
            o_rel = o_rel.permute(0, 2, 3, 1)  # (batch, n_span + 1, n_span + 1, n_out)
        else:
            o_rel = o_rel.unsqueeze(1).permute(0, 2, 3, 1)  # (batch, n_span + 1, n_span + 1, n_out)

        return o_arc, o_rel

    def edge_predict(self, arc_out: torch.Tensor, rel_out: torch.Tensor,
                     span_mask: torch.Tensor, top_mask: torch.Tensor) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # arc_out: (batch, n_span + 1, n_span + 1)
        # rel_out: (batch, n_span + 1, n_span + 1, n_out)
        # span_mask: (batch, n_span + 1)
        # top_mask: (batch, n_span + 1)
        lens = [torch.sum(m) for m in span_mask + top_mask]

        arc_out = torch.sigmoid(arc_out)  # (batch, n_span + 1, n_span + 1, n_out)
        rel_out = torch.softmax(rel_out, dim=-1)  # (batch, n_span + 1, n_span + 1, n_out)

        arc_pred = [matrix[:l, :l].cpu().detach().numpy()
                    for matrix, l in zip(arc_out, lens)]  # batch x (n_span + 1, n_span + 1)
        rel_pred = [matrix[:l, :l].cpu().detach().numpy()
                    for matrix, l in zip(rel_out, lens)]  # batch x (n_span + 1, n_span + 1, n_out)

        return arc_pred, rel_pred

    def _get_loss(self, batch) -> Dict[str, torch.Tensor]:
        input_target = batch['input_target']
        g_fw = input_target['fw']  # (batch,)
        g_bio = input_target['bio_ids']  # (batch, seq)
        g_prop = input_target['proposition_ids']  # (batch, n_span)
        g_arc = input_target['edge_arcs']  # (batch, n_span + 1, n_span + 1)
        g_rel = input_target['edge_rels']  # (batch, n_span + 1, n_span + 1)

        assert sum(g_arc[0][0] != self.vocab.token2id(P.PAD)) - 1 == sum(g_prop[0] != self.vocab.token2id(P.PAD))

        # Encode
        exs = self.plm_encode(batch=batch)
        mask = input_target['mask'] * input_target['mask_special_tokens']

        assert sum(g_bio[0] != self.vocab.token2id(P.PAD)) == sum(mask[0])

        # Output layers
        fw_outs = dict()
        for fw in self.fw_vocab.tokens:
            span_out = self.span_output(exs=exs, fw=fw)  # (batch, seq, n_out)
            span_pred = self.span_predict(exs=exs, span_out=span_out, mask=mask, gold=g_bio)
            prop_out = self.proposition_output(span_exs=span_pred['span_exs'], fw=fw)  # (batch, n_span + 1, n_out)
            arc_out, rel_out = self.edge_output(span_exs=span_pred['span_exs'], fw=fw)

            fw_outs[fw] = {
                'span': span_out,  # (batch, seq, n_out)
                'span_mask': span_pred['span_mask'],  # (batch, n_span + 1)
                'top_mask': span_pred['top_mask'],  # (batch, n_span + 1)
                'prop': prop_out,  # (batch, n_span + 1, n_out)
                'arc': arc_out,  # (batch, n_span + 1, n_span + 1)
                'rel': rel_out,  # (batch, n_span + 1, n_span + 1, n_out)
            }

        def _d():
            return {fw: [] for fw in self.fw_vocab.tokens}

        fw_2_span_outs, fw_2_prop_outs, fw_2_arc_outs, fw_2_arc_masks, fw_2_rel_outs = \
            _d(), _d(), _d(), _d(), _d()
        fw_2_span_golds, fw_2_prop_golds, fw_2_arc_golds, fw_2_rel_golds = \
            _d(), _d(), _d(), _d()

        # Get framework-specific outputs
        for i, fw in enumerate(g_fw):
            fwo = fw_outs[fw]
            m_x = mask[i]  # (seq,)
            m_span = fwo['span_mask'][i]  # (n_span + 1,)
            m_top = fwo['top_mask'][i]  # (n_span + 1,)
            fw_2_span_outs[fw].append(fwo['span'][i][m_x])
            fw_2_prop_outs[fw].append(fwo['prop'][i][m_span])  # (n_span, n_out)
            l = torch.sum(m_span + m_top)  # n_span + 1
            fw_2_arc_outs[fw].append(fwo['arc'][i][:l, :l].flatten())

            fw_g_bio = g_bio[i]
            fw_2_span_golds[fw].append(fw_g_bio[fw_g_bio != self.vocab.token2id(P.PAD)])
            fw_g_prop = g_prop[i]
            fw_2_prop_golds[fw].append(fw_g_prop[fw_g_prop != self.vocab.token2id(P.PAD)])
            fw_g_arc = g_arc[i]
            fw_2_arc_golds[fw].append(fw_g_arc[fw_g_arc != self.vocab.token2id(P.PAD)])
            fw_g_rel = g_rel[i]
            fw_g_rel_mask = fw_g_rel != self.vocab.token2id(P.PAD)

            if torch.sum(fw_g_arc > 0) != 0:
                fw_2_arc_masks[fw].append(g_arc[i][:l, :l].flatten().type(torch.bool))
                fw_2_rel_golds[fw].append(fw_g_rel[fw_g_rel_mask])
                fw_2_rel_outs[fw].append(fwo['rel'][i][:l, :l].flatten(end_dim=-2))

        # Get framework-specific loss
        loss_dict = dict()
        for fw in self.fw_vocab.tokens:
            if self.config.tgt_fw == fw:
                fw_weight = self.config.lambda_tgt_fw
            elif self.config.lambda_other_fw is not None:
                fw_weight = self.config.lambda_other_fw
            else:
                fw_weight = 1.0

            # [Loss] Span identification
            if fw_2_span_golds[fw]:
                if f'{fw}_anchor' not in loss_dict:
                    loss_dict[f'{fw}_anchor'] = 0.
                loss_dict[f'{fw}_anchor'] += fw_weight * self.config.lambda_bio * nn.functional.cross_entropy(
                    input=torch.cat(fw_2_span_outs[fw]),
                    target=torch.cat(fw_2_span_golds[fw]),
                    reduction='mean')

            # [Loss] Component classification
            if fw_2_prop_golds[fw] and len(torch.cat(fw_2_prop_golds[fw])) > 0:
                if f'{fw}_label' not in loss_dict:
                    loss_dict[f'{fw}_label'] = 0.
                loss_dict[f'{fw}_label'] += fw_weight * self.config.lambda_proposition * nn.functional.cross_entropy(
                    input=torch.cat(fw_2_prop_outs[fw]),
                    target=torch.cat(fw_2_prop_golds[fw]),
                    reduction='mean')

            if fw_2_arc_golds[fw] and len(torch.cat(fw_2_arc_golds[fw])) > 0:

                # [Loss] Edge link identification
                if f'{fw}_edge_link' not in loss_dict:
                    loss_dict[f'{fw}_edge_link'] = 0.
                loss_dict[f'{fw}_edge_link'] += \
                    fw_weight * self.config.lambda_arc * nn.functional.binary_cross_entropy_with_logits(
                        input=torch.cat(fw_2_arc_outs[fw]),
                        target=torch.cat(fw_2_arc_golds[fw]),
                        reduction='mean'
                    )

                # [Loss] Edge label classification
                if sum([len(m) for m in fw_2_arc_masks[fw]]) != 0:
                    arc_masks = torch.cat(fw_2_arc_masks[fw])
                    rel_outs = torch.cat(fw_2_rel_outs[fw])
                    if f'{fw}_edge_label' not in loss_dict:
                        loss_dict[f'{fw}_edge_label'] = 0.
                    loss_dict[f'{fw}_edge_label'] += fw_weight * self.config.lambda_rel * nn.functional.cross_entropy(
                        input=rel_outs[arc_masks],
                        target=torch.cat(fw_2_rel_golds[fw]),
                        reduction='mean')

        return loss_dict

    def save_model(
            self,
            path: str,
            optimizer,
            scheduler,
            warmup_steps: int,
            total_steps: int,
            current_epoch: int,
            current_step: int,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'config': self.config,
            'numericalizer': self.numericalizer,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'warmup_steps': warmup_steps,
            'total_steps': total_steps,
            'current_epoch': current_epoch,
            'current_step': current_step,
        }, path)

    @classmethod
    def load_model(cls, path,
                   optimizer_cls=AdamW,
                   fn_scheduler=get_linear_schedule_with_warmup,
                   load_optimizers=True,
                   device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        state = torch.load(path, map_location=device)
        config = state['config']
        numericalizer = state['numericalizer']
        warmup_steps = state['warmup_steps']
        total_steps = state['total_steps']
        current_epoch = state['current_epoch'] if 'current_epoch' in state else None
        current_step = state['current_step'] if 'current_step' in state else None
        model = cls(config, numericalizer)
        model.load_state_dict(state['state_dict'], False)

        if load_optimizers and 'optimizer' in state:
            optimizer = optimizer_cls(model.parameters())
            optimizer.load_state_dict(state['optimizer'])
            for opt_state in optimizer.state.values():
                for opt_k, opt_v in opt_state.items():
                    if isinstance(opt_v, torch.Tensor):
                        opt_state[opt_k] = opt_v.to(device)
        else:
            optimizer = None

        if load_optimizers and 'scheduler' in state:
            scheduler = fn_scheduler(optimizer, warmup_steps, total_steps)
            scheduler.load_state_dict(state['scheduler'])
        else:
            scheduler = None

        return config, model, numericalizer, optimizer, scheduler, \
               warmup_steps, total_steps, current_epoch, current_step

    @staticmethod
    def load_config(path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        config = state['config']
        return config
