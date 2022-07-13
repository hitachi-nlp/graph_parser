# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import tempfile
from typing import List, Dict, Tuple
import os
import csv
import glob
import random
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import amparse.common.util as util


@dataclass
class Arguments:
    """
    Arguments
    """
    dir_aaec: str = field(
        default=None,
        metadata={'help': 'The path to the input directory which contains .txt and .ann files'},
    )
    prefix: str = field(
        default='AAEC_',
        metadata={'help': 'The prefix for component labels and edges'},
    )
    dir_output: str = field(
        default=None,
        metadata={'help': 'The output directory path'},
    )
    seed: int = field(
        default=42,
        metadata={'help': 'The random seed to shuffle ids'},
    )
    level: str = field(
        default='para',
        metadata={'help': '"essay" or "para"'},
    )
    dev_rate: float = field(
        default=.1,
        metadata={'help': 'The development set ratio in training data'},
    )


def filter_with_span(ann_lines: List[str], span: Tuple[int, int]):
    new_lines = []
    node_ids = []
    for ann_line in ann_lines:
        annots = ann_line.split('\t')
        if len(annots) < 2:
            assert False, 'Invalid ann format'
        if annots[1].startswith('AnnotatorNotes'):
            continue
        # Add ADU
        if annots[0].startswith('T'):
            adu_data = annots[1].split(' ')
            if len(adu_data) == 3:
                adu_type, start, stop = adu_data
            else:
                start, stop = adu_data[1], adu_data[-1]
                adu_type = ''
            start, stop = int(start), int(stop)
            if span[0] <= start <= stop <= span[1]:
                new_lines.append(f'{annots[0]}\t{adu_type} {start - span[0]} {stop - span[0]}\n')
                node_ids.append(int(annots[0][1:]))
        # Add edge
        elif annots[0].startswith('R'):
            edge_label, src, tgt = annots[1].split(' ')
            src = int(src.replace('Arg1:', '')[1:])
            tgt = int(tgt.replace('Arg2:', '')[1:])
            if src in node_ids or tgt in node_ids:
                assert src in node_ids and tgt in node_ids, "Seems wired (edges do not stride paragraphs..)"
                new_lines.append(ann_line)
        # Add major claim
        elif annots[0].startswith('A'):
            _, src, stance = annots[1].split(' ')
            src = int(src[1:])
            if src in node_ids:
                new_lines.append(ann_line)
    return new_lines


def read_aaec(conf: Arguments, ann_path: str, txt_path: str):
    source = 'https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp'
    mrps = []

    essay_mrp = util.read_brat(
        ann_path=ann_path,
        txt_path=txt_path,
        framework='aaec_essay',
        prefix=conf.prefix,
        source=source
    )
    essay_mrp = util.reverse_edge(mrp=essay_mrp)
    essay_mrp = util.sort_mrp_elements(mrp=essay_mrp)
    if conf.level.lower() == 'essay':
        mrps.append(essay_mrp)
    elif conf.level.lower() == 'para':
        claim_str2stance = {
            essay_mrp['input'][
                n['anchors'][0]['from']: n['anchors'][0]['to']
            ]: n['label']
            for n in essay_mrp['nodes'] if f'{conf.prefix}Claim' in n['label']
        }
        with open(ann_path, 'r') as f:
            ann_lines = f.readlines()
        with open(txt_path, 'r') as f:
            txt = f.read()
        txt_lines = txt.split('\n')
        while not txt_lines[-1].strip():
            txt_lines.pop(-1)
        first_para = '\n'.join(txt_lines[:3]) + '\n'  # We include the title header
        middle_paras = [l + '\n' for l in txt_lines[3: -1]]
        last_para = txt_lines[-1]
        para_offset = 0
        for i_para, para in enumerate([first_para] + middle_paras + [last_para]):
            paragraph_span = (para_offset, para_offset + len(para))
            para_ann_lines = filter_with_span(ann_lines=ann_lines, span=paragraph_span)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_ann = os.path.join(temp_dir, 'tmp.ann')
                temp_txt = os.path.join(temp_dir, 'tmp.txt')
                util.dump_text(fpath=temp_txt, text=para, newline_at_end=False)
                util.dump_text(fpath=temp_ann, text=''.join(para_ann_lines), newline_at_end=False)
                mrp = util.read_brat(
                    ann_path=temp_ann,
                    txt_path=temp_txt,
                    framework='aaec_para',
                    prefix=conf.prefix,
                    source=source
                )
                mrp['id'] = os.path.basename(ann_path).replace('.ann', '') + f'_{i_para}'
                # Add stance into a claim
                for node in mrp['nodes']:
                    if node['label'] == f'{conf.prefix}Claim':
                        node['label'] = claim_str2stance[
                            mrp['input'][
                                node['anchors'][0]['from']: node['anchors'][0]['to']
                            ]
                        ]
                mrp = util.reverse_edge(mrp=mrp)
                mrp = util.sort_mrp_elements(mrp=mrp)
                mrps.append(mrp)
            para_offset += len(para)
    else:
        assert False, '--level must be "essay" or "para"'
    return mrps


def split_dataset(conf: Arguments, mrps: List[Dict]):
    train_ids = []
    with open(os.path.join(conf.dir_aaec, '../train-test-split.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=';')
        rows = [row for row in reader][1:]  # Skip header
        for _id, label in rows:
            if label == 'TRAIN':
                train_ids.append(_id)
    train_ids = sorted(train_ids)

    train_mrps = [jd for jd in mrps if jd['id'].split('_')[0] in train_ids]
    test_mrps = [jd for jd in mrps if jd['id'].split('_')[0] not in train_ids]

    random.seed(conf.seed)
    random.shuffle(train_ids)
    dev_ids = train_ids[:int(len(train_ids) * conf.dev_rate)]
    train_mrps, dev_mps = [jd for jd in train_mrps if jd['id'].split('_')[0] not in dev_ids], \
                          [jd for jd in train_mrps if jd['id'].split('_')[0] in dev_ids]
    return train_mrps, dev_mps, test_mrps


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)
    random.seed(conf.seed)
    # Load files
    ann_files = glob.glob(os.path.join(conf.dir_aaec, '*.ann'), recursive=True)
    txt_files = glob.glob(os.path.join(conf.dir_aaec, '*.txt'), recursive=True)
    # Sort the files
    ann_files, txt_files = sorted(ann_files), sorted(txt_files)
    assert len(ann_files) == len(txt_files)
    logging.info(ann_files)
    logging.info(txt_files)

    mrps = []
    for ann, txt in zip(ann_files, txt_files):
        mrps += read_aaec(conf=conf, ann_path=ann, txt_path=txt)

    util.dump_jsonl(fpath=os.path.join(conf.dir_output, f'aaec_{conf.level}.mrp'), jsonl=mrps)

    # Split dataset
    train_mrps, dev_mps, test_mrps = split_dataset(conf=conf, mrps=mrps)
    util.dump_jsonl(fpath=os.path.join(conf.dir_output, f'aaec_{conf.level}_train.mrp'), jsonl=train_mrps)
    util.dump_jsonl(fpath=os.path.join(conf.dir_output, f'aaec_{conf.level}_dev.mrp'), jsonl=dev_mps)
    util.dump_jsonl(fpath=os.path.join(conf.dir_output, f'aaec_{conf.level}_test.mrp'), jsonl=test_mrps)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
