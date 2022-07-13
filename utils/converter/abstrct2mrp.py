# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from dataclasses import dataclass, field
from transformers import HfArgumentParser
import os
import glob
import logging

import amparse.common.util as util


@dataclass
class Arguments:
    """
    Arguments
    """
    dir_abstrct: str = field(
        default=None,
        metadata={'help': 'The input directory path which contains .txt and .ann files'},
    )
    prefix: str = field(
        default='AbstRCT_',
        metadata={'help': 'The prefix for component labels and edges'},
    )
    output: str = field(
        default=None,
        metadata={'help': 'The output mrp file path'},
    )


def read_abstrct(conf, ann_path: str, txt_path: str):
    mrp = util.read_brat(ann_path=ann_path, txt_path=txt_path,
                         framework='abstrct', prefix=conf.prefix,
                         source='https://gitlab.com/tomaye/abstrct/')
    # Re-label the Premise label into Evidence label
    for node in mrp['nodes']:
        if 'premise' in node['label'].lower():
            node['label'] = f'{conf.prefix}Evidence'
    mrp = util.reverse_edge(mrp=mrp)
    mrp = util.sort_mrp_elements(mrp=mrp)
    return mrp


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)
    # Load files
    ann_files = glob.glob(os.path.join(conf.dir_abstrct, '*.ann'), recursive=True)
    txt_files = glob.glob(os.path.join(conf.dir_abstrct, '*.txt'), recursive=True)
    # Sort the files
    ann_files, txt_files = sorted(ann_files), sorted(txt_files)
    assert len(ann_files) == len(txt_files)
    logging.info(ann_files)
    logging.info(txt_files)

    mrps, section_mrps = [], []
    for ann, txt in zip(ann_files, txt_files):
        mrp = read_abstrct(conf=conf, ann_path=ann, txt_path=txt)
        mrps.append(mrp)

    os.makedirs(os.path.dirname(conf.output), exist_ok=True)
    util.dump_jsonl(fpath=conf.output, jsonl=mrps)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
