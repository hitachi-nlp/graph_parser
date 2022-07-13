# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import os
import random
from typing import List, Dict, Optional
import copy
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
    output1: str = field(
        default=None,
        metadata={'help': 'The output path for the first file'},
    )
    output2: str = field(
        default=None,
        metadata={'help': 'The output path for the second file'},
    )
    output2_rate: Optional[float] = field(
        default=.1,
        metadata={'help': 'The ratio of output2'},
    )
    min_output2: Optional[int] = field(
        default=1,
        metadata={'help': 'The number of min samples for the output2'},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'The random seed for splitting'},
    )


def split_dataset(conf: Arguments, mrps: List[Dict]):
    mrps = copy.deepcopy(mrps)
    mrps = sorted(mrps, key=lambda x: x['id'])

    random.seed(conf.seed)
    random.shuffle(mrps)

    n = max(int(len(mrps) * conf.output2_rate), conf.min_output2)
    mrps2, mrps1 = mrps[:n], mrps[n:]
    return mrps1, mrps2


def main(conf: Arguments):
    mrps = util.read_mrp(mrp_path=conf.input)
    mrps1, mrps2 = split_dataset(conf=conf, mrps=mrps)
    util.try_mkdir(os.path.dirname(conf.output1))
    util.try_mkdir(os.path.dirname(conf.output2))
    util.dump_jsonl(fpath=conf.output1, jsonl=mrps1)
    util.dump_jsonl(fpath=conf.output2, jsonl=mrps2)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)

