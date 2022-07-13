# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import os
import glob
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import amparse.common.util as util


@dataclass
class Arguments:
    """
    Arguments
    """
    dir_aasd: str = field(
        default=None,
        metadata={'help': 'The input directory path which contains .conll files'},
    )
    prefix: str = field(
        default='AASD_',
        metadata={'help': 'The prefix for component labels and edges'},
    )
    output: str = field(
        default=None,
        metadata={'help': 'The output mrp file path'},
    )


def read_aasd(conf, conll_path: str):
    tokens, bios, node_labels, edge_labels, trg_distances = [], [], [], [], []
    with open(conll_path, 'r') as f:
        for row in f.readlines():
            row = row.rstrip()
            token, labels = row.split('\t')
            splits = [l for l in labels.split('-')]
            if len(splits) == 4:
                bio, node_label, edge_label, target = splits
                trg_distance = int(target)
            else:
                bio, node_label, edge_label, target = [s for s in splits if s]
                trg_distance = -int(target)
            tokens.append(token)
            bios.append(bio)
            node_labels.append(node_label)
            edge_labels.append(edge_label)
            trg_distances.append(trg_distance)

    # Parse graph
    nodes, edges = [], []
    spans = util.bio_sequence_to_spans(bios)
    bodies = ''
    for span in spans:
        body = ''
        for token in tokens[span.start: span.stop]:
            if token == '-LRB-':
                token = '('
            elif token == '-RRB-':
                token = ')'
            # add to body
            if token in ('.', ',', '?', '!'):
                body += token
            elif body != '':
                body += ' ' + token
            else:
                body += token
        # Node label
        node_label = node_labels[span.start: span.stop][0]
        assert len({node_label} & set(node_labels[span.start: span.stop])) == 1
        # Edge label
        edge_label = edge_labels[span.start: span.stop][0]
        assert len({edge_label} & set(edge_labels[span.start: span.stop])) == 1
        # Edge target
        trg_distance = trg_distances[span.start: span.stop][0]
        assert len({trg_distance} & set(trg_distances[span.start: span.stop])) == 1

        nid = len(nodes)

        nodes.append({
            "id": nid,
            "label": conf.prefix + node_label,
            "anchors": [{"from": len(bodies), "to": len(bodies) + len(body)}],
        })
        if edge_label != 'none':
            assert trg_distance != 0
            edges.append({
                "source": nid,
                "target": nid + trg_distance,
                "label": conf.prefix + edge_label
            })
        else:
            assert trg_distance == 0

        bodies += body + ' '
    bodies = bodies[:-1]  # Remove the space ' '

    tops = []
    for node in nodes:
        out_edges = [e for e in edges if e['source'] == node['id']]
        if not out_edges:
            tops.append(node['id'])

    mrp = {
        "id": os.path.basename(conll_path).replace('.conll', ''),
        "input": bodies,
        "framework": "aasd",
        "time": "2020-08-05",
        "flavor": 0,
        "version": 1.0,
        "language": "en",
        "provenance": "http://scientmin.taln.upf.edu/argmin/scidtb_argmin_annotations.tgz",
        "source": "http://scientmin.taln.upf.edu/argmin/scidtb_argmin_annotations.tgz",
        "nodes": nodes,
        "edges": edges,
        "tops": tops
    }
    mrp = util.reverse_edge(mrp=mrp)
    mrp = util.sort_mrp_elements(mrp=mrp)
    return mrp


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)
    # Load files
    conll_files = glob.glob(os.path.join(conf.dir_aasd, '*.conll'), recursive=True)
    # Sort the files
    conll_files = sorted(conll_files)
    logging.info(conll_files)

    mrps, section_mrps = [], []
    for conll in conll_files:
        mrp = read_aasd(conf=conf, conll_path=conll)
        mrps.append(mrp)

    os.makedirs(os.path.dirname(conf.output), exist_ok=True)
    util.dump_jsonl(fpath=conf.output, jsonl=mrps)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
