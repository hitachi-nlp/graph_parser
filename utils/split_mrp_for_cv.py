# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import json
import os
import math
import random
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
        metadata={'help': 'The input jsonline file path'},
    )
    dir_output: str = field(
        default=None,
        metadata={'help': 'The output directory path'},
    )
    iter: int = field(
        default=1,
        metadata={'help': 'The number of CV iterations'},
    )
    fold: int = field(
        default=5,
        metadata={'help': 'The number of folds in a CV'},
    )
    dev_rate: float = field(
        default=.1,
        metadata={'help': 'The development data ratio in training data'},
    )
    seed: int = field(
        default=42,
        metadata={'help': 'The random seed for splitting'},
    )


def chunker_list(seq, size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main(conf: Arguments):
    assert conf.iter > 0, 'The iter must be 1 or larger'
    assert conf.fold > 1, 'The fold must be 2 or larger'
    os.makedirs(conf.dir_output, exist_ok=True)

    ids = []
    jds = []
    with open(conf.input, 'r') as f:
        for line in f.readlines():
            if not line:
                continue
            jd = json.loads(line)
            jds.append(jd)
            ids.append(jd['id'])
    assert len(ids) == len(set(ids)), f'Found id duplications {[x for x in ids if ids.count(x) > 1]}'
    ids = sorted(ids)

    random.seed(conf.seed)

    i_fold = 0
    for _ in range(conf.iter):
        random.shuffle(ids)
        folds = list(chunker_list(ids, math.ceil(len(ids) / conf.fold)))
        assert len(folds) == conf.fold

        val_ids = []

        for fold_ids in folds:
            train_jds = [jd for jd in jds if jd['id'] not in fold_ids]
            test_jds = [jd for jd in jds if jd['id'] in fold_ids]
            assert len(train_jds) + len(test_jds) == len(jds)

            dev_ids = [jd['id'] for jd in train_jds]
            random.shuffle(dev_ids)
            dev_ids = dev_ids[:int(len(train_jds) * conf.dev_rate)]
            train_jds, dev_jds = [jd for jd in train_jds if jd['id'] not in dev_ids], \
                                 [jd for jd in train_jds if jd['id'] in dev_ids]

            # Validation
            train_ids = [jd['id'] for jd in train_jds]
            dev_ids = [jd['id'] for jd in dev_jds]
            test_ids = [jd['id'] for jd in test_jds]
            assert not (set(train_ids) & set(dev_ids) & set(test_ids))
            assert len(train_jds) + len(test_jds) + len(dev_jds) == len(jds)

            assert not (set(val_ids) & set(test_ids))
            val_ids += test_ids

            base_name = os.path.basename(conf.input).split('.')[0]
            util.dump_jsonl(fpath=os.path.join(conf.dir_output, f'{base_name}.cv{i_fold}.train.mrp'), jsonl=train_jds)
            util.dump_jsonl(fpath=os.path.join(conf.dir_output, f'{base_name}.cv{i_fold}.dev.mrp'), jsonl=dev_jds)
            util.dump_jsonl(fpath=os.path.join(conf.dir_output, f'{base_name}.cv{i_fold}.test.mrp'), jsonl=test_jds)
            i_fold += 1

        assert len(val_ids) == len(ids) == len(set(val_ids) & set(ids))
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)

