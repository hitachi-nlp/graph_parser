# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import sys
from transformers import HfArgumentParser
import logging
import os
import pprint
from joblib import Parallel, delayed
from typing import List, Optional
from dataclasses import dataclass, field
import datetime

import torch
import torch.utils.data
from transformers.optimization import (AdamW, get_linear_schedule_with_warmup)

from amparse.common import util as util
from amparse.loader.corpus import Corpus
from amparse.evaluator.metric import AMMetric
from amparse.loader.data_loader import construct_loader
from amparse.model import parser_generator as generator
from amparse.model.parser import AMParser
from amparse.loader.fields import InputTargetField
from amparse.common.parser_config import AMParserConfig


@dataclass
class Arguments:
    """
    Arguments
    """
    models: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'Model file paths to predict/ensemble.'
                          ' Ensemble is activated only when two or more models are specified.',
                  'nargs': '+'},
    )
    input: str = field(
        default=None,
        metadata={'help': 'The input mrp file path'},
    )
    log: Optional[str] = field(
        default=f'./log/predict_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
        metadata={'help': 'The output directory path'},
    )
    image: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to output visualized images or not.'},
    )
    batch_size: Optional[int] = field(
        default=16,
        metadata={'help': 'The batch size'},
    )
    oracle_span: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to use oracle span or not.'},
    )
    overwrite_framework: Optional[str] = field(
        default=None,
        metadata={'help': 'Overwrite framework if specified.'},
    )
    overwrite_postprocessor: Optional[str] = field(
        default=None,
        metadata={'help': 'Overwrite postprocessor if specified.'},
    )


class AMPredictor:
    def __init__(self, args: Arguments):
        args.models = [m for m in args.models if os.path.exists(m)]
        self.args: Arguments = args
        self.models = []
        self.repr_numericalizer = None
        self.repr_conf = None
        return

    @torch.no_grad()
    def predict_and_build_metric(self, models, numericalizer, config: AMParserConfig, loader):
        metric = AMMetric(numericalizer=numericalizer, config=config)

        for i_batch, batch in enumerate(loader):
            pred_data = generator.ensemble(models=models, batch=batch['batch'], oracle_span=self.args.oracle_span)
            for sentence, d in zip(batch['origin'], pred_data):
                metric.add(sentence, d)

            logging.info(f'Predicted: batch: {i_batch + 1}/{len(loader)}')

        return metric

    def predict(self):
        util.setup_logger(log_dir=self.args.log, name='prediction.log')
        logging.info(util.dump_config(config=self.args))

        assert len(self.args.models) != 0, f'--models is not specified. Specify at least one model path.'

        logging.info(f'Using following models:\n{pprint.pformat(self.args.models)}')
        if len(self.args.models) > 1:
            logging.info('Average ensemble is enabled.')

        for model in self.args.models:
            assert os.path.exists(model), f'"{model}" does not exist.'
        assert os.path.exists(self.args.input), f'"{self.args.input}" does not exist.'

        # Build fields and model
        if not self.models:
            for i, model_path in enumerate(self.args.models):
                config, model, numericalizer, _, _, _, _, _, _ \
                    = AMParser.load_model(
                        path=model_path,
                        optimizer_cls=AdamW,
                        fn_scheduler=get_linear_schedule_with_warmup,
                        load_optimizers=False,
                    )

                model = util.setup_gpu(model)
                model.eval()

                if self.repr_numericalizer is None:
                    self.repr_numericalizer = numericalizer
                    self.repr_conf: AMParserConfig = config

                # Show the model info
                logging.info(config)
                logging.info(f'Model ({i}): {model}')
                logging.info(f'Model parameters: {util.count_model_parameters(model)}')
                self.models.append(model)

        with open(self.args.input, 'r') as f:
            dumps = [l for l in f.readlines() if l]
        pred_corpus = Corpus.load_from_dump(dumps=dumps)

        # Overwrite the framework
        ofw = self.args.overwrite_framework
        if ofw:
            for sentence in pred_corpus.sentences:
                sentence.framework = ofw
                logging.info(f'Overwrite framework into : {ofw}')

        # Overwrite the post-processor
        opp = self.args.overwrite_postprocessor
        if opp:
            overwritten_postprocessor = []
            for s in self.repr_conf.postprocessor.split(','):
                k, _ = s.split(':')
                overwritten_postprocessor.append(f'{k}:{opp}')
            self.repr_conf.postprocessor = ",".join(overwritten_postprocessor)
            logging.info(f'Overwrite postprocessor into : {opp}')

        # Use the dummy label when evaluate on the oracle span
        if self.args.oracle_span:
            if ofw:
                input_target_field: InputTargetField = self.repr_numericalizer.field_by_name('input_target')
                dummy_label = input_target_field.fw_2_proposition_vocab[ofw].tokens[0]
                for sentence in pred_corpus.sentences:
                    for node in sentence.data['nodes']:
                        node['label'] = dummy_label
                logging.info(f'Replacing all node labels into a dummy label "{dummy_label}"')
        else:
            for sentence in pred_corpus.sentences:
                sentence.data['nodes'] = []

        for sentence in pred_corpus.sentences:
            sentence.data['edges'] = []
        for sentence in pred_corpus.sentences:
            sentence.data['tops'] = []

        logging.info(f'Prediction set: {len(pred_corpus):5} samples')
        logging.info(f'\t-----')
        logging.info(f'Prediction sample:')
        logging.info(f"\t{pred_corpus[0].data['id']}")
        logging.info(f"\t{pred_corpus[0].data['framework']}")
        logging.info(f"\t{pred_corpus[0].data['input']}")

        # Build the prediction dataset
        pred_dataset_loader: torch.utils.data.DataLoader = construct_loader(
            pred_corpus,
            numericalizer=self.repr_numericalizer,
            batch_size=self.args.batch_size,
            shuffle=False
        )

        logging.info(f'Prediction started....')

        metric = self.predict_and_build_metric(
            models=self.models,
            numericalizer=self.repr_numericalizer,
            config=self.repr_conf,
            loader=pred_dataset_loader
        )

        mrp_path = os.path.join(self.args.log, 'prediction.mrp')
        metric.dump_pred_mrp(save_path=mrp_path)
        logging.info(f'Saved predictions mrp at "{mrp_path}"')

        if self.args.image:
            dump_args = []
            for i, res in enumerate(metric.prediction_results):
                _id = res['prediction']['mrp']['id']
                fmt = 'pdf'
                dirpath = os.path.join(self.args.log, f'images/{fmt}/')
                fname = _id.replace(os.sep, '_')
                os.makedirs(dirpath, exist_ok=True)
                dump_args.append([mrp_path, i, os.path.join(dirpath, f'{fname}.{fmt}'), fmt])
            try:
                Parallel(n_jobs=12)(
                    delayed(util.dump_mrp_image)(*dump_arg)
                    for dump_arg in dump_args
                )
            except:
                pass

        logging.info(f'Prediction finished')
        return


def main():
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        config = parser.parse_args_into_dataclasses()[0]
    predictor = AMPredictor(args=config)
    predictor.predict()
    return


if __name__ == '__main__':
    main()
