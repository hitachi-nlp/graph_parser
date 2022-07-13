# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

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


def main(conf: Arguments):
    # Set logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)

    mrps = util.read_mrp(mrp_path=conf.input)

    node_labels = dict()
    edge_labels = dict()

    for mrp in mrps:
        for node in mrp['nodes']:
            if 'label' in node:
                nlabel = node['label']
                if nlabel not in node_labels:
                    node_labels[nlabel] = 0
                node_labels[nlabel] += 1

        for edge in mrp['edges']:
            if 'label' in edge:
                elabel = edge['label']
                if elabel not in edge_labels:
                    edge_labels[elabel] = 0
                edge_labels[elabel] += 1

    print(f'Node labels: {node_labels.items()}')
    print(f'Edge labels: {edge_labels.items()}')
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
