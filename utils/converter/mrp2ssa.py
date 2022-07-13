# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

# Structured sentiment analysis

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
        metadata={'help': 'The input mrp file path'},
    )
    output: str = field(
        default=None,
        metadata={'help': 'The output json path'},
    )


def to_ssa(mrp: Dict):
    ssa = {
        "sent_id": mrp['id'],
        "text": mrp['input'],
        "opinions": [],
    }

    def _nids2ssa(_nids):
        _find_nodes = [n for n in mrp['nodes'] if n['id'] in _nids]
        _find_nodes = sorted(_find_nodes, key=lambda x: (x['anchors'][0]['from'], x['anchors'][0]['to']))
        _res = [[], []]
        for _node in _find_nodes:
            _anc = (_node['anchors'][0]['from'], _node['anchors'][0]['to'])
            _res[0].append(mrp['input'][slice(*_anc)])
            _res[1].append(f"{_anc[0]}:{_anc[1]}")
        return _res

    # Get opinions

    for edge in mrp['edges']:
        # Find self-loops
        if edge['source'] != edge['target']:
            continue
        pol_nid = edge['source']
        edge_labels = edge['label'].split('_')

        # Find polarity
        polarity_nids = [pol_nid]
        if 'Positive' in edge_labels:
            polarity = 'Positive'
        elif 'Negative' in edge_labels:
            polarity = 'Negative'
        elif 'Neutral' in edge_labels:
            polarity = 'Neutral'
        else:
            logging.warning(f'Edge label in {edge} must be either Positive, Negative or Neutral. '
                            f'As a workaround, we convert this into Neutral label')
            polarity = 'Neutral'

        # Find source, target, and polarity nodes
        source_nids, target_nids = [], []
        for edge2 in mrp['edges']:
            if edge2['source'] == edge2['target']:
                continue
            if edge2['source'] != pol_nid:
                continue

            tgt_nid = edge2['target']
            edge_labels2 = edge2['label'].split('_')

            if 'Source' in edge_labels2:
                source_nids.append(tgt_nid)
            if 'Target' in edge_labels2:
                target_nids.append(tgt_nid)
            if 'Positive' in edge_labels2 or 'Negative' in edge_labels2:
                polarity_nids.append(tgt_nid)

        ssa['opinions'].append({
            'Source': _nids2ssa(_nids=source_nids),
            'Target': _nids2ssa(_nids=target_nids),
            'Polar_expression': _nids2ssa(_nids=polarity_nids),
            'Polarity': polarity,
        })
    return ssa


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)

    with open(conf.input, 'r') as f:
        mrps = [json.loads(l) for l in f.readlines() if l.strip()]

    ssas = []
    for mrp in mrps:
        ssas.append(to_ssa(mrp=mrp))

    util.try_mkdir(os.path.dirname(conf.output))

    with open(conf.output, 'w') as f:
        f.write(json.dumps(ssas, ensure_ascii=False))
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
