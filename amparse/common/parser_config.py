# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AMParserConfig:
    """
    Arguments for model training
    """
    log: Optional[str] = field(
        default=f'./log/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
        metadata={'help': 'The directory path for logging'},
    )
    ftrain: str = field(
        default=None,
        metadata={'help': 'Train file (.mrp)'},
    )
    fvalid: Optional[str] = field(
        default=None,
        metadata={'help': 'Validation file (.mrp)'},
    )
    ftest: Optional[str] = field(
        default=None,
        metadata={'help': 'Test file (.mrp)'},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'The random seed for model and training initialization'},
    )
    build_numericalizer_on_entire_corpus: Optional[bool] = field(
        default=False,
        metadata={'help': 'Builds numericalizer from the ftrain, fvalid and ftest files.'},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={'help': 'Makes input tokens lower cased'},
    )
    attention_window: Optional[int] = field(
        default=512,
        metadata={'help': 'The attention window-size for Longformer'},
    )
    split_document: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to split overlength text into multiple segments'
                          ' (experimental / could contain significant bugs)'},
    )
    max_encode: Optional[int] = field(
        default=4096,
        metadata={'help': 'The max token length for the encoder'},
    )

    embed_dropout: Optional[float] = field(
        default=.1,
        metadata={'help': 'The dropout ratio of the input embeddings'},
    )
    dim_mlp: Optional[int] = field(
        default=768,
        metadata={'help': 'The dimension of MLP layers'},
    )
    mlp_dropout: Optional[float] = field(
        default=.1,
        metadata={'help': 'The dropout ratio of MLP layers'},
    )
    dim_biaffine: Optional[int] = field(
        default=768,
        metadata={'help': 'The dimension of the biaffine layers'},
    )
    model_name_or_path: Optional[str] = field(
        default='roberta-base',
        metadata={'help': 'The model path or name.'
                          ' If there exists a corresponding local path, the local model will be loaded.'
                          ' Otherwise, the pre-trained model from Huggingface will be used.'},
    )
    postprocessor: Optional[str] = field(
        default="default:default,aaec:aaec,aaec_essay:aaec,aaec_para:aaec,mtc:mtc,cdcp:cdcp,abstrct:abstrct,aasd:aasd,tree:tree,trees:trees,graph:graph,ssa:ssa",
        metadata={'help': 'The postprocessor key and value for each framework.'},
    )
    evaluate_with_oracle_span: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to evaluate node labels and edges with oracle spans.'},
    )

    # Loss multipliers
    lambda_bio: Optional[float] = field(
        default=1.,
        metadata={'help': 'The loss weight for the BIO labeling (span identification)'},
    )
    lambda_proposition: Optional[float] = field(
        default=1.,
        metadata={'help': 'The loss weight for node labeling (component classification)'},
    )
    lambda_arc: Optional[float] = field(
        default=1.,
        metadata={'help': 'The loss weight for link detection'},
    )
    lambda_rel: Optional[float] = field(
        default=1.,
        metadata={'help': 'The loss weight for relation classification'},
    )
    tgt_fw: Optional[str] = field(
        default=None,
        metadata={'help': 'The target framework name used for the loss weighting'},
    )
    lambda_tgt_fw: Optional[float] = field(
        default=1.0,
        metadata={'help': 'The loss weight for the target framework'},
    )
    lambda_other_fw: Optional[float] = field(
        default=None,
        metadata={'help': 'The loss weight for NON-target frameworks'},
    )

    # Optimizers
    lr: Optional[float] = field(
        default=5e-5,
        metadata={'help': 'The learning rate'},
    )
    beta1: Optional[float] = field(
        default=.9,
        metadata={'help': 'Adam beta1'},
    )
    beta2: Optional[float] = field(
        default=.998,
        metadata={'help': 'Adam beta2'},
    )
    warmup_ratio: Optional[float] = field(
        default=.1,
        metadata={'help': 'The linear warmup ratio'},
    )
    clip: Optional[float] = field(
        default=5.,
        metadata={'help': 'The value of gradient clipping'},
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={'help': 'The batch size for training'},
    )
    eval_batch_size: Optional[int] = field(
        default=4,
        metadata={'help': 'The batch size for prediction and evaluation'},
    )
    epochs: Optional[int]= field(
        default=20,
        metadata={'help': 'The number of epochs'},
    )
    terminate_epochs: Optional[int] = field(
        default=20,
        metadata={'help': 'The number of terminate epochs. '
                          'Training will be forced to terminate by this epoch number.'},
    )
    evaluate_epochs: Optional[int] = field(
        default=2,
        metadata={'help': 'The number of epochs for evaluation. '
                          'Validation will be conducted every this epochs.'},
    )
    
    disable_evaluation: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to disable evaluations.'
                          ' If true, validation and test files will be ignored.'},
    )
    disable_saving_large_files: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to disable saving large files.'
                          ' If true, the model and training progress will not be saved.'},
    )
