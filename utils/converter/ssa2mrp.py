# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from typing import Dict
import os
import json
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import amparse.common.util as util


@dataclass
class Arguments:
    """
    Arguments
    """
    input: str = field(
        default=None,
        metadata={'help': 'Path to an input json file'},
    )
    output: str = field(
        default=None,
        metadata={'help': 'The output mrp file path'},
    )
    language: str = field(
        default='en',
        metadata={'help': 'The used language'},
    )
    clean_span: bool = field(
        default=False,
        metadata={'help': 'Whether to discards invalid spans'},
    )
    framework: str = field(
        default=None,
        metadata={'help': 'The framework (i.e., corpus) name, e.g., opener_en'},
    )


def to_mrp(jd: Dict, conf: Arguments):
    mrp = {
        "id": jd['sent_id'],
        "input": jd['text'],
        "framework": conf.framework,
        "time": "2021-12-28",
        "flavor": 0,
        "version": 1.0,
        "language": conf.language,
        "provenance": "https://github.com/jerbarnes/semeval22_structured_sentiment",
        "source": "https://github.com/jerbarnes/semeval22_structured_sentiment",
        "nodes": [],
        "edges": [],
        "tops": []
    }

    # Filter opinions

    def _validate_span(surfaces, spans) -> bool:
        for _surface, _span in zip(surfaces, spans):
            _start, _stop = _span.split(':')
            _start, _stop = int(_start), int(_stop)
            if jd['text'][_start: _stop] != _surface:
                return False
        return True

    opinions = []
    for opinion in jd['opinions']:
        if conf.clean_span:
            if not _validate_span(*opinion['Polar_expression']):
                print(f'Invalid Polar_expression {opinion}')
                return None
            if 'Target' in opinion and not _validate_span(*opinion['Target']):
                print(f'Invalid Target span {opinion}')
                return None
            if 'Source' in opinion and not _validate_span(*opinion['Source']):
                print(f'Invalid Source span {opinion}')
                return None
        opinions.append(opinion)

    # Get fine-grained span nodes

    split_positions = []
    all_positions = set()
    for opinion in opinions:
        assert 'Polar_expression' in opinion
        assert 'Polarity' in opinion
        for field in ['Source', 'Target', 'Polar_expression']:
            if opinion[field] == [[], []]:
                continue
            for surface, indexes in zip(*opinion[field]):
                if not indexes:
                    continue
                frm, to = indexes.split(':')
                frm, to = int(frm), int(to)
                split_positions += [frm, to]
                all_positions |= set(range(frm, to))
        split_positions = sorted(list(set(split_positions)))

    prev_pos = 0
    for pos in split_positions:
        rng = (prev_pos, pos)
        if not mrp['input'][slice(*rng)].strip():
            prev_pos = pos
            continue
        while mrp['input'][rng[0]] == ' ':
            rng = (rng[0] + 1, rng[1])
        while rng[1] <= len(mrp['input']) and mrp['input'][rng[1] - 1] == ' ':
            rng = (rng[0], rng[1] - 1)

        if set(range(*rng)) & all_positions:
            mrp['nodes'].append({
                'id': len(mrp['nodes']),
                'label': 'Span',
                'anchors': [{'from': rng[0], 'to': min(rng[1], len(jd['text']))}]
            })
        prev_pos = pos

    # Generate edges

    for opinion in opinions:
        assert 'Polar_expression' in opinion
        assert 'Polarity' in opinion

        opinion_nodes = []
        edge_labels = []
        for field in ['Source', 'Target', 'Polar_expression']:
            if opinion[field] == [[], []]:
                continue

            assert len(opinion[field]) == 2

            edge_label = f'{field}'
            if field == 'Polar_expression':
                edge_label = f'{opinion["Polarity"]}'

            for surface, indexes in zip(*opinion[field]):
                if not indexes:
                    continue

                frm, to = indexes.split(':')
                frm, to = int(frm), int(to)

                for node in mrp['nodes']:
                    node_span = (node['anchors'][0]['from'], node['anchors'][0]['to'])
                    if set(range(*node_span)) & set(range(frm, to)):
                        opinion_nodes.append(node)
                        edge_labels.append(edge_label)

        assert len(opinion_nodes) == len(edge_labels)

        for node, edge_label in zip(opinion_nodes, edge_labels):
            duplicate_edge = [
                e for e in mrp['edges']
                if (e['source'], e['target']) == (node['id'], opinion_nodes[-1]['id'])
            ]
            if not duplicate_edge:
                mrp['edges'].append({
                    'source': node['id'],
                    'target': opinion_nodes[-1]['id'],
                    'label': f'{edge_label}',
                })
            else:
                assert len(duplicate_edge) == 1
                duplicate_edge = duplicate_edge[0]
                labels = duplicate_edge['label'].split('_')
                labels = sorted(list(set(labels + [f'{edge_label}'])))
                duplicate_edge['label'] = '_'.join(labels)

    mrp = util.reverse_edge(mrp=mrp)

    for node in mrp['nodes']:
        in_edges = [e for e in mrp['edges'] if e['target'] == node['id'] and not e['target'] == e['source']]
        if not in_edges:
            mrp['tops'].append(node['id'])

    return mrp


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)

    with open(conf.input, 'r') as f:
        jds = json.loads(f.read())

    mrps = []
    n_drop = 0
    for jd in jds:
        mrp = to_mrp(jd=jd, conf=conf)
        if mrp is not None:
            mrps.append(mrp)
        else:
            n_drop += 1
    print(f'Dropped {n_drop} graphs')

    util.try_mkdir(os.path.dirname(conf.output))
    util.dump_jsonl(fpath=conf.output, jsonl=mrps)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
