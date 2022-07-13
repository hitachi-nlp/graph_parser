# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from dataclasses import dataclass, field
import os
import random
from typing import List, Dict, Optional
import copy
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
        metadata={'help': 'The output mrp file path'},
    )
    output_rate: Optional[float] = field(
        default=1.0,
        metadata={'help': 'The ratio of sampling'},
    )
    min_output: Optional[int] = field(
        default=1,
        metadata={'help': 'The number of min samples'},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'The random seed for sampling'},
    )


def sampling(conf: Arguments, mrps: List[Dict]):
    mrps = copy.deepcopy(mrps)

    if conf.output_rate >= 1.:
        print(f'Sampling was not conducted because output_rate >= 1.')
        return mrps

    mrps = sorted(mrps, key=lambda x: x['id'])

    random.seed(conf.seed)
    random.shuffle(mrps)

    n = max(int(len(mrps) * conf.output_rate), conf.min_output)
    return mrps[:n]


def main(conf: Arguments):
    mrps = util.read_mrp(mrp_path=conf.input)
    mrps = sampling(conf=conf, mrps=mrps)
    util.try_mkdir(os.path.dirname(conf.output))
    util.dump_jsonl(fpath=conf.output, jsonl=mrps)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
