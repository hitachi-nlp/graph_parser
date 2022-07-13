# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from typing import List
import torch

from amparse.model.parser import AMParser


@torch.no_grad()
def ensemble(models: List[AMParser], batch, oracle_span: bool = False):
    assert len(models) > 0

    repr_model = models[0]
    input_target = batch['input_target']

    exs_list = []
    mask_list = []
    for model in models:
        model.eval()
        exs_list.append(model.plm_encode(batch=batch))
        mask_list.append(input_target['mask'] * input_target['mask_special_tokens'])

    # Output layers
    fw_outs = dict()
    for fw in repr_model.fw_vocab.tokens:
        span_out = []
        for model, exs in zip(models, exs_list):
            span_out.append(
                model.span_output(exs=exs, fw=fw)  # (batch, seq, n_out)
            )
        span_out = torch.mean(torch.stack(span_out), dim=0)

        prop_out = []
        arc_out = []
        rel_out = []
        for model, exs, mask in zip(models, exs_list, mask_list):
            if oracle_span:
                g_bio = input_target['bio_ids']  # (batch, seq)
            else:
                g_bio = None
            span_pred = model.span_predict(exs=exs, span_out=span_out, mask=mask, gold=g_bio, oracle=oracle_span)
            prop_out.append(
                model.proposition_output(span_exs=span_pred['span_exs'], fw=fw)  # (batch, n_span, n_out)
            )
            ao, ro = model.edge_output(span_exs=span_pred['span_exs'], fw=fw)
            arc_out.append(ao)
            rel_out.append(ro)

        prop_out = torch.mean(torch.stack(prop_out), dim=0)
        arc_out, rel_out = torch.mean(torch.stack(arc_out), dim=0), torch.mean(torch.stack(rel_out), dim=0)
        prop_pred = repr_model.proposition_predict(prop_out=prop_out, span_mask=span_pred['span_mask'], fw=fw)
        arc_pred, rel_pred = repr_model.edge_predict(arc_out=arc_out, rel_out=rel_out,
                                                     span_mask=span_pred['span_mask'], top_mask=span_pred['top_mask'])

        fw_outs[fw] = {
            'span_exs': span_pred['span_exs'],  # batch x (n_span + 1) x seq
            'span_mask': span_pred['span_mask'],  # batch x (n_span + 1) x seq
            'span': span_pred['span'],  # batch x seq
            'prop': prop_pred,  # batch x n_span
            'arc': arc_pred,  # batch x (n_span + 1, n_span + 1)
            'rel': rel_pred,  # batch x (n_span + 1, n_span+ 1, n_out)
        }

    outs = []
    for i, _ in enumerate(input_target['tokens']):
        out = dict()

        # Get the offset mapping
        offsets = input_target['offset_mapping'][i]

        # Get the outputs
        for fw in repr_model.fw_vocab.tokens:
            span_exs = fw_outs[fw]['span_exs'][i]  # n_span x dim
            span_mask = fw_outs[fw]['span_mask'][i]  # n_span
            spans = [slice(offsets[s.start][0], offsets[s.stop - 1][1]) for s in fw_outs[fw]['span'][i]]
            out[fw] = {
                'span_exs': span_exs[span_mask].cpu().detach().numpy().tolist(),
                'span': spans,
                'proposition': fw_outs[fw]['prop'][i],
                'arc': fw_outs[fw]['arc'][i],
                'rel': fw_outs[fw]['rel'][i],
            }

        outs.append(out)
    return outs
