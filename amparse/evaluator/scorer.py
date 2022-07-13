# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import argparse
import json
import io
import copy
from typing import List, Dict, Tuple, Set
from distutils.util import strtobool


class Scorer:
    def __init__(self):
        self.s = 0
        self.g = 0
        self.c = 0
        return

    def add(self, system: Set[Tuple], gold: Set[Tuple]):
        self.s += len(system)
        self.g += len(gold)
        self.c += len(gold & system)
        return

    @property
    def p(self):
        return self.c / self.s if self.s else 0.

    @property
    def r(self):
        return self.c / self.g if self.g else 0.

    @property
    def f(self):
        p = self.p
        r = self.r
        return (2. * p * r) / (p + r) if p + r > 0 else 0.0

    def dump(self):
        return {
            'g': self.g,
            's': self.s,
            'c': self.c,
            'p': self.p,
            'r': self.r,
            'f': self.f
        }


def _read_mrp(f_io: io.TextIOWrapper):
    jds = [json.loads(l) for l in f_io.readlines() if l]
    return jds


def eval_anchor(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        scorer.add(
            system=set([
                (s_mrp['id'], node['anchors'][0]['from'], node['anchors'][0]['to'])
                for node in s_mrp['nodes']]),
            gold=set([
                (g_mrp['id'], node['anchors'][0]['from'], node['anchors'][0]['to'])
                for node in g_mrp['nodes']]),
        )
    return scorer.dump()


def eval_top(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        s_nid2anc = {node['id']: (node['anchors'][0]['from'], node['anchors'][0]['to']) for node in s_mrp['nodes']}
        g_nid2anc = {node['id']: (node['anchors'][0]['from'], node['anchors'][0]['to']) for node in g_mrp['nodes']}
        scorer.add(
            system=set([
                (s_mrp['id'],) + s_nid2anc[top] for top in s_mrp['tops']]),
            gold=set([
                (g_mrp['id'],) + g_nid2anc[top] for top in g_mrp['tops']]),
        )
    return scorer.dump()


def eval_label(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    labels = set()
    for g_mrp in g_mrps:
        labels |= set([n['label'] for n in g_mrp['nodes'] if 'label' in n])
    label_scores = dict()
    for label in labels:
        scorer = Scorer()
        for s_mrp, g_mrp in zip(s_mrps, g_mrps):
            scorer.add(
                system=set([
                    (
                        s_mrp['id'],
                        node['anchors'][0]['from'],
                        node['anchors'][0]['to'],
                    )
                    for node in s_mrp['nodes']
                    if 'label' in node and node['label'] == label
                ]),
                gold=set([
                    (
                        g_mrp['id'],
                        node['anchors'][0]['from'],
                        node['anchors'][0]['to'],
                    )
                    for node in g_mrp['nodes']
                    if 'label' in node and node['label'] == label
                ]),
            )
        label_scores[label] = scorer.dump()

    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        scorer.add(
            system=set([
                (
                    s_mrp['id'],
                    node['anchors'][0]['from'],
                    node['anchors'][0]['to'],
                    node['label']
                )
                for node in s_mrp['nodes']
                if 'label' in node
            ]),
            gold=set([
                (
                    g_mrp['id'],
                    node['anchors'][0]['from'],
                    node['anchors'][0]['to'],
                    node['label']
                )
                for node in g_mrp['nodes']
                if 'label' in node
            ]),
        )
    label_scores['total'] = scorer.dump()
    return label_scores


def eval_edge(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    # Align node anchor and edge
    s_n2anc, g_n2anc = dict(), dict()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        for node in s_mrp['nodes']:
            anc = node['anchors'][0]
            s_n2anc[(s_mrp['id'], node['id'])] = (anc['from'], anc['to'])
        for node in g_mrp['nodes']:
            anc = node['anchors'][0]
            g_n2anc[(g_mrp['id'], node['id'])] = (anc['from'], anc['to'])
    # Obtain edge labels
    edge_labels = set()
    for g_mrp in g_mrps:
        labels = [e['label'] for e in g_mrp['edges'] if 'label' in e]
        edge_labels |= set(labels)
    # Calculate label scores
    label_scores = dict()
    for label in edge_labels:
        scorer = Scorer()
        for s_mrp, g_mrp in zip(s_mrps, g_mrps):
            scorer.add(
                system=set([
                    (
                        s_mrp['id'],
                        *s_n2anc[(s_mrp['id'], edge['source'])],
                        *s_n2anc[(s_mrp['id'], edge['target'])]
                    )
                    for edge in s_mrp['edges']
                    if 'label' in edge and edge['label'] == label
                ]),
                gold=set([
                    (
                        g_mrp['id'],
                        *g_n2anc[(g_mrp['id'], edge['source'])],
                        *g_n2anc[(g_mrp['id'], edge['target'])]
                    )
                    for edge in g_mrp['edges']
                    if 'label' in edge and edge['label'] == label
                ]),
            )
        label_scores[label] = scorer.dump()

    link_scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        link_scorer.add(
            system=set([
                (
                    s_mrp['id'],
                    *s_n2anc[(s_mrp['id'], edge['source'])],
                    *s_n2anc[(s_mrp['id'], edge['target'])]
                )
                for edge in s_mrp['edges']
            ]),
            gold=set([
                (
                    g_mrp['id'],
                    *g_n2anc[(g_mrp['id'], edge['source'])],
                    *g_n2anc[(g_mrp['id'], edge['target'])]
                )
                for edge in g_mrp['edges']
            ]),
        )
    label_scores['link'] = link_scorer.dump()

    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        scorer.add(
            system=set([
                (
                    s_mrp['id'],
                    *s_n2anc[(s_mrp['id'], edge['source'])],
                    *s_n2anc[(s_mrp['id'], edge['target'])],
                    (edge['label'] if 'label' in edge else '')
                )
                for edge in s_mrp['edges']
            ]),
            gold=set([
                (
                    g_mrp['id'],
                    *g_n2anc[(g_mrp['id'], edge['source'])],
                    *g_n2anc[(g_mrp['id'], edge['target'])],
                    (edge['label'] if 'label' in edge else '')
                )
                for edge in g_mrp['edges']
            ]),
        )
    label_scores['total'] = scorer.dump()
    return label_scores


def relieve_overlap(s_mrp: Dict, g_mrp: Dict, overlap: float):
    s_mrp = copy.deepcopy(s_mrp)
    g_anchors = [n['anchors'][0] for n in g_mrp['nodes']]
    for s_node in s_mrp['nodes']:
        s_anc = s_node['anchors'][0]
        max_overlap_len = 0
        for g_anc in g_anchors:
            max_len = max(g_anc['to'] - g_anc['from'], s_anc['to'] - s_anc['from'])
            overlap_len = len(set(range(g_anc['from'], g_anc['to'])) & set(range(s_anc['from'], s_anc['to'])))
            overlap_rate = float(overlap_len) / max_len
            if overlap_rate > overlap and overlap_rate > max_overlap_len:
                s_node['anchors'][0] = g_anc
                max_overlap_len = overlap_rate
    return s_mrp


def relieve_space(s_mrp: Dict, g_mrp: Dict):
    s_mrp = copy.deepcopy(s_mrp)
    g_mrp = copy.deepcopy(g_mrp)
    txt = s_mrp['input']

    for s_node in s_mrp['nodes']:
        while True:
            s_anc = s_node['anchors'][0]
            if txt[s_anc['from']] == ' ':
                s_node['anchors'] = [{'from': s_anc['from'] + 1, 'to': s_anc['to']}]
            else:
                break
        while True:
            s_anc = s_node['anchors'][0]
            if txt[s_anc['to'] - 1] == ' ':
                s_node['anchors'] = [{'from': s_anc['from'], 'to': s_anc['to'] - 1}]
            else:
                break

    for g_node in g_mrp['nodes']:
        while True:
            g_anc = g_node['anchors'][0]
            if txt[g_anc['from']] == ' ':
                g_node['anchors'] = [{'from': g_anc['from'] + 1, 'to': g_anc['to']}]
            else:
                break
        while True:
            g_anc = g_node['anchors'][0]
            if txt[g_anc['to'] - 1] == ' ':
                g_node['anchors'] = [{'from': g_anc['from'], 'to': g_anc['to'] - 1}]
            else:
                break

    return s_mrp, g_mrp


def add_top_scores_into_edge_scores(res_top: Dict, res_edge: Dict):
    res_edge = copy.deepcopy(res_edge)
    for edge_metric in ['total', 'link']:
        g, s, c = (res_top[metric] + res_edge[edge_metric][metric] for metric in ['g', 's', 'c'])
        p, r = c / s, c / g
        f = (2. * p * r) / (p + r)
        res_edge[f'{edge_metric}-top-as-edge'] = {
            'g': g, 's': s, 'c': c, 'p': p, 'r': r, 'f': f}
    return res_edge


def main(args):
    # Read mrp files
    _s_mrps, _g_mrps = _read_mrp(args.system), _read_mrp(args.gold)
    args.system.close()
    args.gold.close()

    # Validate
    s_mrps, g_mrps = [], []
    for g_mrp in _g_mrps:
        g_finds = [g for g in _g_mrps if g['id'] == g_mrp['id']]
        assert len(g_finds) == 1, f'{g_mrp["id"]} duplicates in gold input'
        s_finds = [s for s in _s_mrps if s['id'] == g_mrp['id']]
        assert s_finds, f'{g_mrp["id"]} was not found in system input'
        assert len(s_finds) == 1, f'{g_mrp["id"]} duplicates in system input'
        s_mrps.append(s_finds[0])
        g_mrps.append(g_finds[0])

    if args.overlap is not None:
        new_s_mrps = []
        for s_mrp, g_mrp in zip(s_mrps, g_mrps):
            new_s_mrps.append(
                relieve_overlap(s_mrp=s_mrp, g_mrp=g_mrp, overlap=args.overlap)
            )
        s_mrps = new_s_mrps

    if args.space:
        new_s_mrps, new_g_mrps = [], []
        for s_mrp, g_mrp in zip(s_mrps, g_mrps):
            new_s_mrp, new_g_mrp = relieve_space(s_mrp=s_mrp, g_mrp=g_mrp)
            new_s_mrps.append(new_s_mrp)
            new_g_mrps.append(new_g_mrp)
        s_mrps = new_s_mrps
        g_mrps = new_g_mrps

    res_top = eval_top(s_mrps=s_mrps, g_mrps=g_mrps)
    res_anchor = eval_anchor(s_mrps=s_mrps, g_mrps=g_mrps)
    res_label = eval_label(s_mrps=s_mrps, g_mrps=g_mrps)
    res_edge = eval_edge(s_mrps=s_mrps, g_mrps=g_mrps)
    if args.top_as_edge:
        res_edge = add_top_scores_into_edge_scores(res_top=res_top, res_edge=res_edge)

    res = {
        'tops': res_top,
        'anchors': res_anchor,
        'labels': res_label,
        'edges': res_edge,
    }
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'')
    parser.add_argument(
        '-system',
        type=argparse.FileType("r", encoding='utf-8'),
        default=None,
        help=u'Path to the system file'
    )
    parser.add_argument(
        '-gold',
        type=argparse.FileType("r", encoding='utf-8'),
        default=None,
        help='Path to the gold file'
    )
    parser.add_argument(
        '--top_as_edge',
        type=strtobool,
        default=False,
        help='[Experimental] Evaluate tops as edges'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=None,
        help='[Experimental] Span overlap to relieve'
    )
    parser.add_argument(
        '--space',
        type=strtobool,
        default=0,
        help='Relieve span mis-matches due to space characters'
    )
    args = parser.parse_args()
    main(args)
