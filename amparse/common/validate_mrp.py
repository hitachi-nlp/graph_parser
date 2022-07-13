# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import logging
import collections
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import amparse.common.util as util


@dataclass
class Arguments:
    """
    Arguments
    """
    group1: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'The input mrp file paths of the group1'},
    )
    group2: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={'help': 'The input mrp file paths of the group2'},
    )
    way: Optional[str] = field(
        default='exclusive',
        metadata={'help': 'The way to validate group1 and group2 if group2 is specified.'
                          'Must be "exclusive" or "inclusive"'},
    )


def test_mrp(mrp: Dict):
    if 'id' not in mrp:
        logging.error(f'"id" must be specified.')
        exit()
    if 'input' not in mrp:
        logging.error(f'"input" must be specified in ID:{mrp["id"]}.')
        exit()
    if not isinstance(mrp['input'], str) or not 'input'.strip():
        logging.error(f'"input" must be str and not empty in ID:{mrp["id"]}.')
        exit()
    if not isinstance(mrp['id'], str):
        logging.error(f'"id" must be str in ID:{mrp["id"]}.')
        exit()
    if 'nodes' not in mrp:
        logging.error(f'"nodes" must be specified in ID:{mrp["id"]}')
        exit()
    if not isinstance(mrp['nodes'], list):
        logging.error(f'"nodes" must be a list in ID:{mrp["id"]}')
        exit()

    # Node
    nids = [n['id'] for n in mrp['nodes']]
    if len(set(nids)) != len(nids):
        logging.error(f'Node "id" duplication in ID:{mrp["id"]}.')
        exit()
    nodes = sorted(mrp['nodes'], key=lambda n: (n['anchors'][0]['from'], n['anchors'][0]['to']))
    for i, node in enumerate(nodes):
        if 'id' not in node:
            logging.error(f'Node "id" must be specified in ID:{mrp["id"]}.')
            exit()
        if not isinstance(node['id'], int):
            logging.error(f'Node "id" must be int in NODE:{node["id"]} of ID:{mrp["id"]}.')
            exit()
        if 'anchors' not in node:
            logging.error(f'"anchors" must be specified in NODE:{node["id"]} of ID:{mrp["id"]}.')
            exit()
        if not isinstance(node['anchors'], list):
            logging.error(f'"anchors" must be a list in NODE:{node["id"]} of ID:{mrp["id"]}.')
            exit()
        if len(node['anchors']) != 1:
            logging.error(f'The length of anchors must be 1 in NODE:{node["id"]} of ID:{mrp["id"]}.')
            exit()
        if 'from' not in node['anchors'][0] or 'to' not in node['anchors'][0]:
            logging.error(f'The "anchors" must be {{"from": int, "to": int}} '
                          f'format in NODE:{node["id"]} of ID:{mrp["id"]}.')
            exit()
        anchor = (node['anchors'][0]['from'], node['anchors'][0]['to'])
        if not (0 <= anchor[0] < anchor[1] <= len(mrp['input'])):
            logging.error(f'The anchor {anchor} is out of the index in NODE:{node["id"]} of ID:{mrp["id"]}.'
                          f' (input length is {len(mrp["input"])})')
            exit()
        if i != 0:
            prev_node = nodes[i - 1]
            prev_anchor = (prev_node['anchors'][0]['from'], prev_node['anchors'][0]['to'])
            if not (prev_anchor[1] <= anchor[0]):
                logging.error(f'The anchor duplicates in NODE:{node["id"]} of ID:{mrp["id"]}.')
                exit()
        if 'label' not in node:
            logging.error(f'Node "label" must be specified in NODE:{node["id"]} of ID:{mrp["id"]}.')
            exit()
        if not isinstance(node['label'], str):
            logging.error(f'Node "label" must be str in NODE:{node["id"]} of ID:{mrp["id"]}.')
            exit()

    # Edges
    if 'edges' not in mrp:
        logging.error(f'"edges" must be specified in ID:{mrp["id"]}')
        exit()
    if not isinstance(mrp['edges'], list):
        logging.error(f'"edges" must be a list in ID:{mrp["id"]}')
        exit()
    for edge in mrp['edges']:
        if 'source' not in edge:
            logging.error(f'Edge "source" must be specified in ID:{mrp["id"]}.')
            exit()
        if not isinstance(edge['source'], int):
            logging.error(f'Edge "source" must be int. ID:{mrp["id"]}.')
            exit()
        if edge['source'] not in nids:
            logging.error(f'Edge source {edge["source"]} not found in node ids. ID:{mrp["id"]}.')
            exit()
        if 'target' not in edge:
            logging.error(f'Edge "target" must be specified.. ID:{mrp["id"]}.')
            exit()
        if not isinstance(edge['target'], int):
            logging.error(f'Edge "target" must be int. ID:{mrp["id"]}.')
            exit()
        if edge['target'] not in nids:
            logging.error(f'Edge target {edge["target"]} not found in node ids. ID:{mrp["id"]}.')
            exit()
        if 'label' not in edge:
            logging.error(f'Edge "label" must be specified in '
                          f'EDGE:({edge["source"]}, {edge["target"]}) of ID:{mrp["id"]}.')
            exit()
        if not isinstance(edge['label'], str):
            logging.error(f'Edge "label" must be specified in '
                          f'EDGE:({edge["source"]}, {edge["target"]}) of ID:{mrp["id"]}.')
            exit()
    edges = [(e['source'], e['target']) for e in mrp['edges']]
    if len(set(edges)) != len(edges):
        logging.error(f'Edge duplication in ID:{mrp["id"]}.')
        exit()

    # Tops
    if 'tops' not in mrp:
        logging.error(f'"tops" must be specified in ID:{mrp["id"]}')
        exit()
    if not isinstance(mrp['tops'], list):
        logging.error(f'"tops" must be a list in ID:{mrp["id"]}')
        exit()
    for top in mrp['tops']:
        if not isinstance(top, int):
            logging.error(f'Top {top} must be int. ID:{mrp["id"]}.')
            exit()
        if top not in nids:
            logging.error(f'Top {top} not found in node ids. ID:{mrp["id"]}.')
            exit()
    if len(set(mrp['tops'])) != len(mrp['tops']):
        logging.error(f'Top duplication in ID:{mrp["id"]}.')
        exit()
    return


def run(conf: Arguments):
    util.setup_logger(None)
    g1_mrps, g2_mrps = [], []
    for g_mrps, group in ((g1_mrps, conf.group1), (g2_mrps, conf.group2)):
        for fpath in group:
            mrps = util.read_mrp(fpath)
            if len(mrps) == 0:
                logging.warning(f'{fpath} is empty. Is this your intended input?')
                continue
            [
                test_mrp(m)
                for m in mrps
            ]
            ids = [m['id'] for m in mrps]
            c = collections.Counter(ids)
            if len(set(ids)) != len(mrps):
                logging.error(f'ID duplication: {c.most_common()[:3]} at {fpath}')
                exit()
            g_ids = [m['id'] for m in g_mrps]
            if len(set(ids) & set(g_ids)) != 0:
                logging.error(f'ID duplication: {set(ids) & set(g_ids)} at {fpath}')
                exit()
            g_mrps += mrps

    if conf.group2:
        g1_ids = [m['id'] for m in g1_mrps]
        g2_ids = [m['id'] for m in g2_mrps]
        assert conf.way in ('inclusive', 'exclusive'), 'The option "way" must be in "inclusive" or "exclusive"'
        if conf.way == 'exclusive':
            if len(set(g1_ids) & set(g2_ids)) != 0:
                logging.error(f'Group1 and 2 ID duplication: {set(g1_ids) & set(g2_ids)}')
                exit()
        elif conf.way == 'inclusive':
            if not (len(set(g1_ids) & set(g2_ids)) == len(g1_ids) == len(g2_ids)):
                logging.error(f'Group1 ID missing: {[i for i in g2_ids if i not in g1_ids]}')
                logging.error(f'Group2 ID missing: {[i for i in g1_ids if i not in g2_ids]}')
                exit()

    logging.info(f'[OK] Validated {conf.group1} - {conf.group2} - {conf.way}')
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    run(conf)
