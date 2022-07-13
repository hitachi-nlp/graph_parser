# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import os
import json
import logging
from typing import Dict, Tuple, List
import torch
from torch import nn as nn

import amparse.common.util as util
from amparse.model.parser import AMParser
from amparse.common.parser_config import AMParserConfig
from amparse.evaluator.metric import AMMetric
from amparse.loader.data_loader import AMDataLoader


class BaseTrainer:
    def __init__(self, config: AMParserConfig):
        self.config: AMParserConfig = config
        self.model: AMParser = None
        self.optimizer = None
        self.scheduler = None
        self.numericalizer = None
        self.warmup_steps = 0
        self.total_steps = 0
        self.tb_writer = None
        self._train_steps = 0
        self.clip = 1.
        return

    def build_trainer(self,
                      model: AMParser,
                      scheduler,
                      optimizer,
                      numericalizer,
                      warmup_steps=0,
                      total_steps=0,
                      clip=1.,
                      tb_writer=None,
                      ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.numericalizer = numericalizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.clip = clip
        self.tb_writer = tb_writer
        """
        Set the basic trainer information
        
        Parameters
        ----------
        model : AMParser
            The model instance
        scheduler : torch.optim.lr_scheduler.LambdaLR
            The learning rate scheduler
        optimizer : AdamW
            The optimizer instance
        numericalizer : AMNumericalizer
            The numericalizer instance
        warmup_steps : int
            The number of warm up steps for learning rate scheduler
        total_steps : int
            The number of total steps
        clip : float
            The value of gradient clipping
        tb_writer : SummaryWriter
            The tensorboard summary writer
        """
        return

    @property
    def current_step(self):
        return self._train_steps

    def _tick(self):
        self._train_steps += 1
        return

    def _train_epoch(self, loader):
        self.model.train()

        # Loop of one-epoch
        for batch in loader:
            self.optimizer.zero_grad()

            # Compute loss
            loss_dict: Dict = self.model(batch)
            loss = torch.sum(torch.stack(list(loss_dict.values())))

            assert not torch.isnan(loss).any(), f'Detected loss nan: {loss_dict}'

            # Backward
            loss.backward()

            if self.clip > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.scheduler.step()
            self._tick()

            last_lr = self.scheduler.get_last_lr()[0]
            if self.tb_writer:
                self.tb_writer.add_scalar(f'lr', last_lr, self.current_step)
                for k, v in loss_dict.items():
                    self.tb_writer.add_scalar(f'train_loss/{k}', v, self.current_step)

        self.optimizer.zero_grad()
        return

    @torch.no_grad()
    def predict_and_build_metric(self, loader):
        self.model.eval()
        metric = AMMetric(self.numericalizer, self.config)
        for batch in loader:
            pred_data = self.model.predict(batch)
            for sentence, d in zip(batch['origin'], pred_data):
                metric.add(sentence, d)
        return metric

    def run_train(self, train_loader: AMDataLoader, valid_loader: AMDataLoader, test_loader: AMDataLoader):
        """
        Do training, validation and test with given data loaders

        Parameters
        ----------
        train_loader : AMDataLoader
            The training data loader
        valid_loader : AMDataLoader
            The validation data loader
        test_loader : AMDataLoader
            The test data loader
        """
        # Paths for the output files
        model_file = os.path.join(self.config.log, 'model')
        valid_eval_file = os.path.join(self.config.log, 'evaluation/valid.jsonl')
        util.create_file(valid_eval_file)
        test_eval_file = os.path.join(self.config.log, 'evaluation/test.jsonl')
        util.create_file(test_eval_file)
        predicted_valid_file = os.path.join(self.config.log, 'prediction/valid.mrp')
        predicted_test_file = os.path.join(self.config.log, 'prediction/test.mrp')

        # For creating key names for tensorboard
        def _get_entire_key_value_in_dict(_d: Dict, _key_chain: str = '')\
                -> List[Tuple[str, object]]:
            _chains = []
            for _k, _v in _d.items():
                if not isinstance(_v, dict):
                    _chains.append((f'{_key_chain}/{_k}' if _key_chain else _k, _v))
                else:
                    _chains += _get_entire_key_value_in_dict(_v, f'{_key_chain}/{_k}' if _key_chain else _k)
            return _chains

        # Training epochs
        best_e, best_f = 1, 0.
        for epoch in range(1, self.config.epochs + 1):

            if epoch > self.config.terminate_epochs:
                logging.info(f'Terminated with epoch: {epoch} (see config)')
                break

            # Train one-epoch
            self._train_epoch(train_loader)
            logging.info(f'Epoch finished: '
                         f'epoch={epoch}/{self.config.epochs}, steps={self.current_step}/{self.total_steps}')

            if epoch % self.config.evaluate_epochs != 0:
                continue

            if self.config.disable_evaluation:
                if not self.config.disable_saving_large_files:
                    self.model.save_model(
                        path=model_file,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        warmup_steps=self.warmup_steps,
                        total_steps=self.total_steps,
                        current_epoch=epoch,
                        current_step=self.current_step,
                    )
                    logging.info(f'Saved model at "{model_file}"')
                    best_e = epoch
                continue

            # Do validation
            if len(valid_loader) > 0:
                valid_metric: AMMetric = self.predict_and_build_metric(valid_loader)
                val_scores = valid_metric.compute_scores()
                if val_scores is None:
                    logging.warning(f'[Valid] Evaluation failed')
                    continue

                logging.info(f'[Valid] Scores: {val_scores} at epoch {epoch}')

                # Log into tensorboard
                if self.tb_writer:
                    for keychain, v in _get_entire_key_value_in_dict(val_scores):
                        self.tb_writer.add_scalar(f'val_metric/{keychain}', v, self.current_step)

                # Log validation result
                util.add_write_file(
                    valid_eval_file,
                    content=json.dumps({'scores': val_scores, 'epoch': epoch}, ensure_ascii=False))

                val_f = val_scores['all']
                logging.info(f'[Valid] Avg. F1: {val_f} at epoch {epoch}')
            else:
                logging.warning(f'[Valid] There is no validation set,'
                                f' so the best model will always be selected from the latest one.')
                val_f = 0.

            # Update the best model if the validation score is better than the current best validation score
            if val_f >= best_f:
                logging.info(f'[Valid] Update the best model based on the validation score')
                best_e, best_f = epoch, val_f

                # Save the best model
                if not self.config.disable_saving_large_files:
                    self.model.save_model(
                        path=model_file,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        warmup_steps=self.warmup_steps,
                        total_steps=self.total_steps,
                        current_epoch=epoch,
                        current_step=self.current_step,
                    )
                    logging.info(f'[Valid] Saved model at "{model_file}"')

                if len(valid_loader) > 0:
                    if len(test_loader) == 0 and not self.config.disable_saving_large_files:
                        valid_metric.dump_metric_data(
                            save_dir=os.path.join(self.config.log, f'progress/valid/epoch={epoch}/'),
                            max_num=10
                        )
                    valid_metric.dump_pred_mrp(save_path=predicted_valid_file)
                    logging.info(f'[Valid] Saved validation prediction at "{predicted_valid_file}"')

                # Do test prediction
                if len(test_loader) > 0:
                    test_metric = self.predict_and_build_metric(test_loader)
                    test_scores = test_metric.compute_scores()
                    if not self.config.disable_saving_large_files:
                        test_metric.dump_metric_data(
                            save_dir=os.path.join(self.config.log, f'progress/test/epoch={epoch}/'),
                            max_num=30
                        )

                    # Save test results
                    util.write_file(
                        test_eval_file,
                        content=json.dumps({'scores': test_scores, 'epoch': epoch}, ensure_ascii=False))

                    logging.info(f'[Test] Scores: {test_scores} at epoch {best_e}')
                    test_metric.dump_pred_mrp(save_path=predicted_test_file)
                    logging.info(f'[Test] Saved test prediction at "{predicted_test_file}"')
                    if self.tb_writer:
                        for keychain, v in _get_entire_key_value_in_dict(test_scores):
                            self.tb_writer.add_scalar(f'test_metric/{keychain}', v, self.current_step)

        self.tb_writer.close()

        finished_file = os.path.join(self.config.log, 'finished.json')
        util.create_file(finished_file)
        util.add_write_file(
            finished_file,
            content=json.dumps({'best_f': best_f, 'best_epoch': best_e}, ensure_ascii=False)
        )
        return

