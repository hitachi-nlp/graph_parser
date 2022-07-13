# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import copy
import datetime
import dataclasses
from enum import Enum
import json
import logging
import math
import os
import time
import random
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Generator
import coloredlogs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import (AdamW,
                                       get_linear_schedule_with_warmup)

from amparse.common import params as P
from amparse.loader.corpus import Corpus
from amparse.loader.data_loader import AMDataLoader, construct_loader
from amparse.loader.numericalizer import AMNumericalizer


def setup_logger(log_dir: Optional[str], name: Optional[str] = 'train.log'):
    """
    Setup logging directory and file

    Parameters
    ----------
    log_dir : Optional[str]
        The directory for logging (automatically generated if specified)
    name : Optional[str]
        The path for the log file (automatically generated if specified)
    """
    coloredlogs.CAN_USE_BOLD_FONT = True

    logger = logging.getLogger()

    coloredlogs.install(
        level=logging.INFO,
        logger=logger,
        fmt='%(asctime)s %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_dir is not None:
        try_mkdir(log_dir)
        handler = logging.FileHandler(os.path.join(log_dir, name), 'w', 'utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)


def setup_tensorboard(log_dir: str) -> SummaryWriter:
    """
    Setup tensorboard for logging

    Parameters
    ----------
    log_dir : str
        The path for the directory of tensorboard logger
    """
    if not os.path.exists(log_dir):
        try_mkdir(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    return writer


def set_seed(seed: int):
    """
    Setup random seed.
    Note that even using this function, we can not strictly ensure deterministic property due to
    differences in computational environments.

    Parameters
    ----------
    seed : int
        The random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    return


def try_mkdir(dirpath: str):
    """
    Try to make a directory.

    Parameters
    ----------
    dirpath : str
        The path for the directory to make
    """
    if dirpath in ['./', '.']:
        return
    try:
        os.makedirs(dirpath, exist_ok=True)
    except:
        pass
    return


def dump_config(config) -> str:
    """
    Get a serialized JSON representation of the config

    Parameters
    ----------
    config : AMParserConfig
        The config instance

    Returns
    ----------
    res : str
        The serialized JSON representation of the config
    """
    d = dict()
    for f in dataclasses.fields(config.__class__):
        val = getattr(config, f.name)
        if isinstance(val, Enum):
            val = val.value
        d[f.name] = val

    return json.dumps(d, indent=4, ensure_ascii=False, skipkeys=True)


def decode_mst(np_array) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Decode edge matrix by the Chu/Liu Edmonds algorithm

    Parameters
    ----------
    np_array : numpy.array
        The input energy matrix

    Returns
    ----------
    head : int
        The decoded head
    links : List[Tuple[int, int]]
        The decoded links
    """
    from amparse.common.chu_liu_edmonds import decode_mst

    assert len(np_array.shape) == 2 and np_array.shape[0] == np_array.shape[1]

    arr = np_array.T
    arr = np.insert(arr, 0, 0, axis=0)
    arr = np.insert(arr, 0, 0, axis=1)

    instance_heads, head_type = decode_mst(arr, arr.shape[0], has_labels=False)

    links = []
    head = None
    for child, parent in enumerate(instance_heads[1:]):
        if parent == 0:
            head = child
            continue
        links.append((child, parent - 1))

    return head, links


def build_corpora(
        ftrain: str,
        fvalid: str,
        ftest: Optional[str] = None,
) -> Tuple[Corpus, Corpus, Corpus]:
    """
    Build corpora from train, validation and test files

    Parameters
    ----------
    ftrain : str
        The path for the train file
    fvalid : Optional[str]
        The path for the validation file
    ftest : Optional[str]
        The path for the test file

    Returns
    ----------
    train_corpus : Corpus
        The train corpus instance
    valid_corpus : Corpus
        The validation corpus instance
    test_corpus : Corpus
        The test corpus instance
    """
    def _build_from_path(_fpath: str):
        # Load corpus
        _dumps = []
        if _fpath is not None and os.path.exists(_fpath):
            with open(_fpath, 'r') as f:
                _dumps = [l for l in f.readlines() if l]
        _corpus = Corpus.load_from_dump(dumps=_dumps)
        return _corpus

    # Load train corpus
    assert ftrain and os.path.exists(ftrain), f'Train file {ftrain} does not exist.'
    train_corpus = _build_from_path(ftrain)

    # Load valid corpus
    if fvalid:
        assert os.path.exists(fvalid), f'Valid file {fvalid} does not exist.'
    valid_corpus = _build_from_path(fvalid)

    # Load test corpus
    if ftest:
        assert os.path.exists(ftest), f'Test file {ftest} does not exist.'
    test_corpus = _build_from_path(ftest)

    return train_corpus, valid_corpus, test_corpus


def build_dataset_loaders(
        train_corpus: Corpus,
        valid_corpus: Corpus,
        test_corpus: Corpus,
        numericalizer: AMNumericalizer,
        batch_size: int,
        eval_batch_size: int,
) -> Tuple[AMDataLoader, AMDataLoader, AMDataLoader]:
    """
    Build data loaders from train, validation and test corpora

    Parameters
    ----------
    train_corpus : Corpus
        The train corpus
    valid_corpus : Corpus
        The validation corpus
    test_corpus : Corpus
        The test corpus
    numericalizer : AMNumericalizer
        The numericalizer instance
    batch_size : int
        The batch size for training
    eval_batch_size : int
        The batch size for prediction

    Returns
    ----------
    train_loader : AMDataLoader
        The train corpus instance
    valid_loader : AMDataLoader
        The validation corpus instance
    test_loader : AMDataLoader
        The test corpus instance
    """
    train_loader = construct_loader(
        train_corpus,
        numericalizer=numericalizer,
        batch_size=min(batch_size, len(train_corpus)),
        shuffle=True
    )

    valid_loader = construct_loader(
        valid_corpus,
        numericalizer=numericalizer,
        batch_size=max(min(eval_batch_size, len(valid_corpus)), 1),
        shuffle=False
    )

    test_loader = construct_loader(
        test_corpus,
        numericalizer=numericalizer,
        batch_size=max(min(eval_batch_size, len(test_corpus)), 1),
        shuffle=False
    )

    return train_loader, valid_loader, test_loader


def create_file(fpath: str):
    """
    Creates a new file from path. The file will be cleared if it exists.

    Parameters
    ----------
    fpath : str
        The file path
    """
    try_mkdir(os.path.dirname(fpath))
    with open(fpath, 'w'):
        pass


def add_write_file(fpath: str, content: str):
    """
    Add a string content to the file

    Parameters
    ----------
    fpath : str
        The file path
    content : str
        The string content to be added
    """
    try_mkdir(os.path.dirname(fpath))
    with open(fpath, 'a') as f:
        f.write(content)
        f.write('\n')


def write_file(fpath: str, content: str):
    """
    Write a string content to the file

    Parameters
    ----------
    fpath : str
        The file path
    content : str
        The string content to be added
    """
    try_mkdir(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        f.write(content)
        f.write('\n')


def build_optimizers(
        model: torch.nn.Module,
        warmup_ratio: float,
        total_epochs: int,
        total_samples: int,
        batch_size: int,
        lr: float,
        beta1: float,
        beta2: float,
        fn_scheduler=get_linear_schedule_with_warmup,
) -> Tuple[AdamW, torch.optim.lr_scheduler.LambdaLR, int, int]:
    """
    Build optimizer for the model

    Parameters
    ----------
    model : torch.nn.Module
        The model instance
    warmup_ratio : float
        The warm up ratio for learning rate
    total_epochs : int
        The total epoch number
    total_samples : int
        The number of total training samples (i.e., data size)
    batch_size : int
        The batch size
    lr : float
        The learning rate
    beta1 : float
        Adam beta1
    beta2 : float
        Adam beta2
    fn_scheduler :
        The function for creating a learning rate scheduler

    Returns
    ----------
    optimizer : AdamW
        The optimizer
    scheduler : torch.optim.lr_scheduler.LambdaLR
        The learning rate scheduler
    total_steps : int
        The number of total training steps
    warmup_steps : int
        The number of total warm up steps
    """
    # Obtain the number of train steps
    total_steps, warmup_steps = get_warmup_steps(
        warmup_ratio=warmup_ratio,
        total_epochs=total_epochs,
        total_samples=total_samples,
        batch_size=min(total_samples, batch_size)
    )

    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler = fn_scheduler(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    return optimizer, scheduler, total_steps, warmup_steps


def get_warmup_steps(warmup_ratio: float, total_epochs: int, total_samples: int, batch_size: int):
    """
    Get total steps and warmup steps calculated from given warm up ratio, total epochs, total samples and batch size.

    Parameters
    ----------
    warmup_ratio : float
        The warm up ratio for learning rate
    total_epochs : int
        The total epoch number
    total_samples : int
        The number of total training samples (i.e., data size)
    batch_size : int
        The batch size

    Returns
    ----------
    total_steps : int
        The number of total training steps
    warmup_steps : int
        The number of total warm up steps
    """
    steps_per_epoch = math.ceil(total_samples / batch_size)
    total_steps = int(total_epochs * steps_per_epoch)
    warmup_steps = math.ceil(warmup_ratio * total_steps)
    return total_steps, warmup_steps


def count_model_parameters(model: torch.nn.Module):
    """
    Get total parameters in the model

    Parameters
    ----------
    model : torch.nn.Module
        The model instance

    Returns
    ----------
    n_parameters : int
        The number of total parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_model_trainable_parameters(model):
    """
    Get total trainable parameters in the model

    Parameters
    ----------
    model : torch.nn.Module
        The model instance

    Returns
    ----------
    n_parameters : int
        The number of total trainable parameters
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def now() -> str:
    """
    Get current time in string

    Returns
    ----------
    str_time : str
        The current time in %Y-%m-%d (%H:%M) format
    """
    return datetime.datetime.now().strftime('%Y-%m-%d (%H:%M)')


def split_into_n_sized_chunks(lst: List, n: int) -> Generator:
    """
    Split the given input list into n-sized chunks

    Example usage:
    >>> list(split_into_n_sized_chunks([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]

    Parameters
    ----------
    lst : List
        The input list
    n : int
        The size of a chunk

    Returns
    ----------
    lst : List
        The list of n-sized chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def anchor_span(spans: List[Tuple], span: List or Tuple, ignore_span: Optional[Tuple] = None) -> Tuple[int, int]:
    """
    Find token-level anchors from character-level token spans

    Example usage:
    >>> anchor_span(spans=[(0, 5), (7, 12), (15, 17), (20, 23)], span=(7, 17), ignore_span=None)
    (1, 3)

    Parameters
    ----------
    spans : List[Tuple]
        The token spans in character-level
    span : List or Tuple
        The span in character-level to be anchored
    ignore_span : Optional[Tuple]
        Ignored span

    Returns
    ----------
    token_indices : Tuple[int, int]
        The indices (start, end) for the token spans
    """
    start, stop = span[0], span[1]
    if ignore_span is None:
        ignore_span = (-99999, -99999)
    i_start = [i for i, s in enumerate(spans) if start == s[0] and s != ignore_span]
    i_stop = [i for i, s in enumerate(spans) if stop == s[1] and s != ignore_span]
    if i_start and i_stop:
        return i_start[0], i_stop[0] + 1

    anc = [i for i, s in enumerate(spans) if (start <= s[0] < s[1] <= stop) and s != ignore_span]
    if not anc:
        anc = [i for i, s in enumerate(spans) if (s[0] <= start < stop <= s[1]) and s != ignore_span]
    if not anc:
        anc = [i for i, s in enumerate(spans) if (s[0] <= start < s[1] or s[0] < stop <= s[1]) and s != ignore_span]
    if not anc:
        return None
    assert len(anc) >= 1
    return anc[0], anc[-1] + 1


def modify_bio_sequence(bio_sequence: List[str]) -> List[str]:
    """
    Modify the invalid BIO sequence as much as we can

    Example usage:
    >>> ''.join(modify_bio_sequence(bio_sequence=['I', 'I', 'O', 'I', 'B', 'I', 'O']))
    'BIOBBIO'

    Parameters
    ----------
    bio_sequence : List[str]
        The BIO sequence

    Returns
    ----------
    new_bio_list : List[str]
        The modified BIO sequence
    """
    new_bio_list = []
    for i, b in enumerate(bio_sequence):

        new_bio = b
        if b == P.BI_I and i == 0:
            new_bio = P.BI_B
        elif b == P.BI_I and i != 0:
            if bio_sequence[i - 1] == P.BI_O:
                new_bio = P.BI_B

        new_bio_list.append(new_bio)
    return new_bio_list


def bio_sequence_to_spans(bio_sequence: List[str]) -> List[slice]:
    """
    Convert the BIO sequence into span slices

    Example usage:
    >>> bio_sequence_to_spans(bio_sequence=['B', 'I', 'O', 'B', 'B', 'I', 'O'])
    [slice(0, 2, None), slice(3, 4, None), slice(4, 6, None)]

    Parameters
    ----------
    bio_sequence : List[str]
        The BIO sequence

    Returns
    ----------
    spans : List[slice]
        The converted BIO span slices
    """
    spans = []
    is_in_boundary = False
    for i, bio in enumerate(bio_sequence):

        if bio == P.BI_B:
            if is_in_boundary:
                spans[-1] = slice(spans[-1].start, i)
            spans.append(slice(i, i + 1))
            is_in_boundary = True

        elif bio == P.BI_O and is_in_boundary:
            spans[-1] = slice(spans[-1].start, i)
            is_in_boundary = False

        elif is_in_boundary and i == len(bio_sequence) - 1:
            spans[-1] = slice(spans[-1].start, i + 1)

    return spans


def dump_mrp_image(mrp_path: str, index: int, save_path: str, image_format: str = 'svg'):
    """
    Save an mrp image of a data point of the mrp file

    Parameters
    ----------
    mrp_path : str
        The path for the mrp file
    index : int
        The index (i.e., line num) of the mrp file to select
    save_path : str
        The path for the saved image
    image_format : str
        The image format (e.g., "svg", "pdf" and "jpg")
    """
    with tempfile.TemporaryDirectory() as temp_path:
        dump_path = os.path.join(temp_path, f'{index + 1}.mrp')

        subprocess.Popen(
            f'head -n {index + 1} {mrp_path} | tail -n 1 > {dump_path}',
            shell=True,
        )

        p = subprocess.Popen(
            f'exec mtool --n 1 --strings --read mrp --write dot {dump_path} {dump_path}.dot',
            shell=True,
        )
        try:
            p.communicate(timeout=3.0)
        except subprocess.TimeoutExpired as e:
            logging.error(f'"mtool" timeout in dump_mrp_image')
        finally:
            p.kill()

        p = subprocess.Popen(
            f'exec dot -T{image_format} {dump_path}.dot > {save_path}',
            shell=True
        )
        try:
            p.communicate(timeout=3.0)
        except subprocess.TimeoutExpired as e:
            logging.error(f'"dot" timeout in dump_mrp_image')
        finally:
            p.kill()
    return


def dump_jsonl(fpath: str, jsonl: List[Dict]):
    """
    Save a file in the jsonline (mrp) format

    Parameters
    ----------
    fpath : str
        The path for the saved file
    jsonl : List[Dict]
        The list of dictionary to be saved
    """
    try_mkdir(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        for jl in jsonl:
            line = json.dumps(jl, ensure_ascii=False)
            f.write(f'{line}\n')


def read_mrp(mrp_path: str) -> List[Dict]:
    """
    Read an mrp file

    Parameters
    ----------
    mrp_path : str
        The path for the mrp file
    """
    mrp = []
    if os.path.exists(mrp_path):
        with open(mrp_path, 'r') as f:
            for i, l in enumerate(f.readlines()):
                if l:
                    try:
                        mrp.append(json.loads(l))
                    except:
                        logging.error(f'{i}-th line of {mrp_path} is invalid')
                        exit()
    return mrp


def dump_text(fpath: str, text: str, newline_at_end: bool = True):
    """
    Save a text file

    Parameters
    ----------
    fpath : str
        The path for the saved file
    text : str
        The string content to be saved
    newline_at_end : bool
        Whether to include newline character at the end of the file
    """
    try_mkdir(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        f.write(text + ('\n' if newline_at_end else ''))


def setup_gpu(model: torch.nn.Module) -> torch.nn.Module:
    """
    Try to set the GPU device for the model

    Parameters
    ----------
    model : torch.nn.Module
        The model instance

    Returns
    ----------
    model : torch.nn.Module
        The model instance loaded on the device if available
    """
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        return model.to('cuda')
    else:
        logging.info('GPUs are not available. Running on CPU.')
        return model


def device() -> str:
    """
    Get available device

    Returns
    ----------
    device : str
        Device name: "cuda" if available, otherwise "cpu"
    """
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        return 'cuda'
    else:
        return 'cpu'


def read_brat(ann_path: str, txt_path: str, framework: str, prefix: str = '', source: str = 'N/A') -> Dict:
    """
    Read the brat formatted file and text file, converting them into the mrp dictionary

    Parameters
    ----------
    ann_path : str
        The path for the brat annotation file (.ann)
    txt_path : str
        The path for the text file associated with the ann_path
    framework : str
        The name of the framework
    prefix : str
        The name of the label prefix
    source : str
        The source of the dataset, e.g., URL

    Returns
    ----------
    mrp : Dict
        The converted mrp dictionary
    """
    with open(ann_path, 'r') as f:
        ann_lines = f.readlines()
    with open(txt_path, 'r') as f:
        txt = f.read()

    nodes, edges, tops = [], [], []
    major_claims = []
    for ann_line in ann_lines:

        annots = ann_line.split('\t')

        if len(annots) < 2:
            assert False, 'Invalid ann format at {}'.format(ann_path)
        if annots[1].startswith('AnnotatorNotes'):
            continue

        # Add component
        if annots[0].startswith('T'):
            adu_data = annots[1].split(' ')

            if len(adu_data) == 3:
                adu_type, start, stop = adu_data
            else:
                adu_type = adu_data[0]
                start, stop = adu_data[1], adu_data[-1]

            node = {
                "id": int(annots[0][1:]),
                "label": adu_type,
                "anchors": [{"from": int(start), "to": int(stop)}],
            }
            nodes.append(node)

            if adu_type == 'MajorClaim':
                major_claims.append(node)

        # Add relation
        elif annots[0].startswith('R'):
            edge_label, src, trg = annots[1].split(' ')
            src = int(src.replace('Arg1:', '')[1:])
            trg = int(trg.replace('Arg2:', '')[1:])

            find = [e for e in edges if e['source'] == src and e['target'] == trg]
            if find:
                logging.warning(f'Found duplication: {ann_path}, {find}')
            else:
                edges.append({"source": src, "target": trg, "label": edge_label})

    if major_claims:
        # Assign a stance (For or Against) for each Claim
        nid2node = {n['id']: n for n in nodes}

        for ann_line in ann_lines:
            annots = ann_line.split('\t')
            if annots[1].startswith('AnnotatorNotes'):
                continue

            # Add stance for a claim
            if annots[0].startswith('A'):
                _, src, stance = annots[1].split(' ')
                src = src[1:]
                stance = stance.strip()
                nid2node[int(src)]['label'] += f':{stance}'

    # Reassign node id to make the id starts with zero
    nodes = sorted(nodes, key=lambda x: x['anchors'][0]['from'])
    nid2newid = {n['id']: i for i, n in enumerate(nodes)}
    for node in nodes:
        node['id'] = nid2newid[node['id']]
        node['label'] = prefix + node['label']
    for edge in edges:
        edge['source'] = nid2newid[edge['source']]
        edge['target'] = nid2newid[edge['target']]
        edge['label'] = prefix + edge['label']

    tops = []
    for node in nodes:
        out_edges = [e for e in edges if e['source'] == node['id']]
        if not out_edges:
            tops.append(node['id'])

    mrp = {
        "id": os.path.basename(ann_path).replace('.ann', ''),
        "input": txt,
        "framework": framework,
        "time": "2020-08-05",
        "flavor": 0,
        "version": 1.0,
        "language": "en",
        "provenance": source,
        "source": source,
        "nodes": nodes,
        "edges": edges,
        "tops": tops,
    }
    return mrp


def reverse_edge(mrp: Dict) -> Dict:
    """
    Reverse edges (exchange source and target)

    Example usage:
    >>> reverse_edge(mrp={'edges': [{'source': 0, 'target': 1}, {'source': 2, 'target': 3}]})
    {'edges': [{'source': 1, 'target': 0}, {'source': 3, 'target': 2}]}

    Parameters
    ----------
    mrp : Dict
        The input mrp dictionary

    Returns
    ----------
    mrp : Dict
        The output mrp dictionary
    """
    mrp = copy.deepcopy(mrp)
    new_edges = []
    for e in mrp['edges']:
        e['source'], e['target'] = e['target'], e['source']
        new_edges.append(e)
    mrp['edges'] = new_edges
    return mrp


def sort_mrp_elements(mrp: Dict) -> Dict:
    """
    Sort mrp tops, nodes and edges by ID

    Parameters
    ----------
    mrp : Dict
        The input mrp dictionary

    Returns
    ----------
    mrp : Dict
        The output mrp dictionary
    """
    mrp = copy.deepcopy(mrp)
    mrp['tops'] = sorted(mrp['tops'])
    mrp['nodes'] = sorted(mrp['nodes'], key=lambda x: x['id'])
    mrp['edges'] = sorted(mrp['edges'], key=lambda x: (x['source'], x['target']))
    return mrp


def from_pretrained(model_cls, limits: int = 100, sleep: float = 3.0, **kwargs):
    """
    Download the pre-trained model or load it from the cache.
    This method also includes automatic "re-try" function when the connection failed.

    Parameters
    ----------
    model_cls :
        PretrainedModel or PretrainedTokenizer
    limits : int
        The number of max re-tries
    sleep : float
        The time to sleep before the re-try
    **kwargs :

    Returns
    ----------
    pretrained_model_or_tokenizer :
        The loaded pretrained model or tokenizer
    """
    for i in range(limits):
        try:
            return model_cls.from_pretrained(**kwargs)
        except:
            logging.info(f'from_pretrained failed. Waiting until network connection recovers... {i}/{limits}')
            os.environ["CURL_CA_BUNDLE"] = ""
            time.sleep(sleep)
