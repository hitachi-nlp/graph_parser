# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import os
import logging
from joblib import Parallel, delayed
from typing import List, Optional
import subprocess
from main import read_graphs  # mtool function
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
    log: str = field(
        default='./log/graph_viz/',
        metadata={'help': 'The output directory path'},
    )
    formats: Optional[List[str]] = field(
        default_factory=lambda: ['pdf', 'png', 'svg'],
        metadata={'help': 'The log directory path'},
    )
    cores: Optional[int] = field(
        default=5,
        metadata={'help': 'The number of cores to use'},
    )


def dump_image(conf, fmt: str, graph, dot_path: str):
    fmt_dir = os.path.join(conf.log, f'{fmt}/{graph.framework}')
    fmt_path = os.path.join(fmt_dir, os.path.basename(dot_path).replace('.dot', f'.{fmt}'))
    os.makedirs(os.path.dirname(fmt_path), exist_ok=True)
    try:
        subprocess.check_call(['dot', f'-T{fmt}', dot_path, '-o', fmt_path])
    except:
        logging.error(f'Could not dump: {graph.id} ({graph.framework})')


def main(conf: Arguments):
    # Set logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)
    with open(conf.input, 'r') as f:
        graphs, _ = read_graphs(f, format='mrp')

    dot_dir = os.path.join(conf.log, 'dot/')

    dot_files = []
    for graph in graphs:
        dot_path = os.path.join(dot_dir, f'{graph.framework}/', graph.id.replace(os.sep, '_') + '.dot')
        os.makedirs(os.path.dirname(dot_path), exist_ok=True)
        with open(dot_path, 'w') as f:
            graph.dot(f, ids=False, strings=True)
        dot_files.append((dot_path, graph))

    for fmt in conf.formats:
        if fmt == 'dot':
            continue
        logging.info(f'Dumping {fmt} images...')
        Parallel(n_jobs=conf.cores)(
            delayed(dump_image)(conf, fmt, graph, dot_path)
            for dot_path, graph in dot_files
        )
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
