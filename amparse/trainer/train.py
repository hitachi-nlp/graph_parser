# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import logging
import os
import sys
from transformers import HfArgumentParser
from transformers.optimization import (AdamW, get_linear_schedule_with_warmup)

from amparse.common import util as util
from amparse.loader.corpus import Corpus
from amparse.common.parser_config import AMParserConfig
from amparse.loader.fields import InputTargetField
from amparse.loader.numericalizer import AMNumericalizer
from amparse.model.parser import AMParser
from amparse.trainer.base_trainer import BaseTrainer


class AMTrainer(BaseTrainer):
    def __init__(self, config: AMParserConfig):
        super(AMTrainer, self).__init__(config)
        util.setup_logger(log_dir=self.config.log, name='train.log')
        return

    def train(self):
        # Dump config info
        logging.info(self.config)
        with open(os.path.join(self.config.log, 'config.json'), 'w') as f:
            f.write(util.dump_config(self.config))

        if os.path.exists(self.config.model_name_or_path):
            # Load the local model
            logging.info(f'Loading existing model and numericalizer from "{self.config.model_name_or_path}"')
            _, model, numericalizer, _, _, _, _, _, _ = AMParser.load_model(
                path=self.config.model_name_or_path,
                optimizer_cls=AdamW,
                fn_scheduler=get_linear_schedule_with_warmup,
                load_optimizers=True,
                device=util.device(),
            )

            logging.info(f'Existing model config: {model.config}')
            logging.info(f'Overwrite the existing model config by the new one')
            model.config = self.config
            logging.info(f'Current model config: {model.config}')

            # Build corpus
            train_corpus, valid_corpus, test_corpus = util.build_corpora(
                ftrain=self.config.ftrain,
                fvalid=self.config.fvalid,
                ftest=self.config.ftest,
            )

            # Build optimizer
            optimizer, scheduler, total_steps, warmup_steps = util.build_optimizers(
                model=model,
                warmup_ratio=self.config.warmup_ratio,
                total_epochs=self.config.epochs,
                total_samples=len(train_corpus),
                batch_size=self.config.batch_size,
                lr=self.config.lr,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
            )

            # Set seed
            util.set_seed(seed=self.config.seed)
            logging.info(f'Seed: {self.config.seed}')

            model = util.setup_gpu(model)

            # Build datasets
            train_loader, valid_loader, test_loader = util.build_dataset_loaders(
                train_corpus=train_corpus,
                valid_corpus=valid_corpus,
                test_corpus=test_corpus,
                numericalizer=numericalizer,
                batch_size=self.config.batch_size,
                eval_batch_size=self.config.eval_batch_size,
            )
        else:
            # Build corpus
            train_corpus, valid_corpus, test_corpus = util.build_corpora(
                ftrain=self.config.ftrain,
                fvalid=self.config.fvalid,
                ftest=self.config.ftest,
            )

            # Set seed
            util.set_seed(seed=self.config.seed)
            logging.info(f'Seed: {self.config.seed}')

            # Build fields
            cat_corpus = Corpus.load_from_sentences(
                train_corpus.sentences + valid_corpus.sentences + test_corpus.sentences
            )

            # Build a new numericalizer
            logging.info(f'Build new numericalizer with "{self.config.model_name_or_path}" tokenizer')
            numericalizer = AMNumericalizer(
                corpus=train_corpus if not self.config.build_numericalizer_on_entire_corpus else cat_corpus,
                fields=[
                    InputTargetField(name='input_target',
                                     model_name_or_path=self.config.model_name_or_path,
                                     do_lower_case=self.config.do_lower_case,
                                     max_encode=self.config.max_encode),
                ]
            )

            # Build datasets
            train_loader, valid_loader, test_loader = util.build_dataset_loaders(
                train_corpus=train_corpus,
                valid_corpus=valid_corpus,
                test_corpus=test_corpus,
                numericalizer=numericalizer,
                batch_size=self.config.batch_size,
                eval_batch_size=self.config.eval_batch_size,
            )

            # Build model
            logging.info(f'Build new model with "{self.config.model_name_or_path}"')
            model: AMParser = AMParser(self.config, numericalizer=numericalizer)

            model = util.setup_gpu(model)

            # Build optimizer
            optimizer, scheduler, total_steps, warmup_steps = util.build_optimizers(
                model=model,
                warmup_ratio=self.config.warmup_ratio,
                total_epochs=self.config.epochs,
                total_samples=len(train_corpus),
                batch_size=self.config.batch_size,
                lr=self.config.lr,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
            )

        logging.info(f'Built loader (train) {train_loader}')
        logging.info(f'Built loader (validation) {valid_loader}')
        logging.info(f'Built loader (test) {test_loader}')

        if len(valid_loader) == 0:
            logging.warning('Validation set is empty. The validation score will be 0.')

        logging.info(f'Train set: {len(train_corpus):5} samples')
        logging.info(f'Valid set: {len(valid_corpus):5} samples')
        logging.info(f'Test set: {len(test_corpus):5} samples')
        logging.info(f'\t-----')
        logging.info(f'Train sample:')
        logging.info(f"\t{train_corpus[0].data['id']}")
        logging.info(f"\t{train_corpus[0].data['input']}")

        logging.info(f'Model: {model}')
        logging.info(f'{numericalizer}')
        logging.info(f'Model parameters: {util.count_model_parameters(model)}')
        logging.info(f'Model trainable parameters: {util.count_model_trainable_parameters(model)}')

        tb_writer = util.setup_tensorboard(os.path.join(self.config.log, 'tensorboard/'))

        self.build_trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            numericalizer=numericalizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            clip=self.config.clip,
            tb_writer=tb_writer,
        )

        logging.info(f'Total steps: {total_steps}')
        logging.info(f'Warmup steps: {warmup_steps}')
        logging.info(f'Fine-tuning started....')

        self.run_train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader
        )

        logging.info(f'Fine-tuning finished')


def main():
    parser = HfArgumentParser(AMParserConfig)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        config = parser.parse_args_into_dataclasses()[0]
    trainer = AMTrainer(config=config)
    trainer.train()
    exit()


if __name__ == '__main__':
    main()
