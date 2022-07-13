# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from typing import Dict, List
import shlex
import os
import subprocess
import json
import tempfile

from amparse.common import util as util
from amparse.common.parser_config import AMParserConfig
from amparse.loader.corpus import Sentence
from amparse.loader.fields import InputTargetField
from amparse.loader.numericalizer import AMNumericalizer
import amparse.postprocess as postprocess
import amparse.evaluator.scorer as scorer


class AMMetric:
    def __init__(self, numericalizer: AMNumericalizer, config: AMParserConfig):
        self.numericalizer: AMNumericalizer = numericalizer
        self.config: AMParserConfig = config
        self.field: InputTargetField = self.numericalizer.field_by_name('input_target')
        self._golds: List[Dict] = []
        self._preds: List[Dict] = []

    def __len__(self):
        return len(self._preds)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_predictions={len(self)})'

    def add(self, sentence: Sentence, pred_data: Dict):
        """
        Add predicted data by reconstruction

        Parameters
        ----------
        sentence : Sentence
            The origin sentence instance to predict
        pred_data : Dict
            The predicted data
        """
        fw = sentence.framework

        # Apply the post-processor for reconstruction
        pp_strs: List[str] = self.config.postprocessor.split(',')
        pp_dict = {s.split(':')[0]: s.split(':')[1] for s in pp_strs}
        assert fw in pp_dict, f'Specify the postprocessor for "{fw}". ' \
                              f'We could not find from {self.config.postprocessor} ({pp_dict})'
        pp_cls = postprocess.BasePostProcess.by_name(pp_dict[fw])
        pp = pp_cls()

        # Obtain predicted mrp result
        pred_mrp = pp.reconstruct(sentence=sentence, pred_data=pred_data[fw], edge_vocab=self.field.fw_2_edge_vocab[fw])

        # Add prediction result
        self._preds.append({
            'mrp': pred_mrp,
        })
        # Add gold
        self._golds.append({
            'mrp': {
                "id": sentence.id,
                "input": sentence.data['input'],
                "flavor": 0,
                "language": sentence.data['language'] if 'language' in sentence.data else '',
                "framework": sentence.framework,
                "time": util.now(),
                "version": sentence.data['version'] if 'version' in sentence.data else '',
                "provenance": sentence.data['provenance'] if 'provenance' in sentence.data else '',
                "source": sentence.data['source'] if 'source' in sentence.data else '',
                "nodes": sentence.data['nodes'],
                "edges": sentence.data['edges'],
                "tops": sentence.data['tops'],
            },
        })

    def compute_scores(self) -> Dict:
        """
        Compute metrics (e.g., precision, recall and F1) by the scorer

        Returns
        ----------
        scores : Dict
            Evaluation results
        """
        self._validate()

        scores = dict()
        for fw in self.field.fw_vocab.tokens:
            scores[fw] = self._run_scorer(fw=fw)

            span_f = scores[fw]['anchors']['f']
            prop_f = scores[fw]['labels']['total']['f']
            edge_f = scores[fw]['edges']['total']['f']
            all_f = (span_f + prop_f + edge_f) / 3.
            scores[fw]['all'] = all_f

        fw_scores = scores.items()
        span_f = sum([s['anchors']['f'] for fw, s in fw_scores]) / len(fw_scores)
        prop_f = sum([s['labels']['total']['f'] for fw, s in fw_scores]) / len(fw_scores)
        edge_f = sum([s['edges']['total']['f'] for fw, s in fw_scores]) / len(fw_scores)
        all_f = (span_f + prop_f + edge_f) / 3.
        scores['all'] = all_f

        return scores

    @property
    def prediction_results(self) -> List[Dict]:
        """
        Get the prediction results

        Returns
        ----------
        res : List[Dict]
            The list of prediction results. Each element in the list contains "prediction" and "gold" dictionaries.
            Example usage of extracting predicted data in MRP: ```res["prediction"]["mrp"]```
        """
        self._validate()

        res = []
        for p, g in zip(self._preds, self._golds):
            res.append({'prediction': p, 'gold': g})

        return res

    def dump_pred_mrp(self, save_path: str):
        """
        Save predicted results as an mrp file

        Parameters
        ----------
        save_path : str
            The path to save the mrp file
        """
        util.dump_jsonl(fpath=save_path, jsonl=[p['mrp'] for p in self._preds])
        return

    def dump_gold_mrp(self, save_path: str):
        """
        Save the gold mrp file

        Parameters
        ----------
        save_path : str
            The path to save the mrp file
        """
        util.dump_jsonl(fpath=save_path, jsonl=[g['mrp'] for g in self._golds])
        return

    def dump_metric_data(self, save_dir: str, max_num: int = 50):
        """
        Save the intermediate data

        Parameters
        ----------
        save_dir : str
            The directory path to save
        max_num : int
            The maximum number of saving images
        """
        util.try_mkdir(os.path.dirname(save_dir))

        # Save the prediction and gold results as mrp files
        path_mrp_s = os.path.join(save_dir, 's.mrp')
        path_mrp_g = os.path.join(save_dir, 'g.mrp')
        self.dump_pred_mrp(save_path=path_mrp_s)
        self.dump_gold_mrp(save_path=path_mrp_g)

        # Get first n prediction results
        n = min(max_num, len(self))
        results = self.prediction_results[:n]

        # Save visualized graphs
        fmt = 'pdf'
        dirpath = os.path.join(save_dir, f'{fmt}/')
        util.try_mkdir(dirpath)
        for i in range(n):
            _id = results[i]['prediction']['mrp']['id']
            fname = _id.replace(os.sep, '_')
            try:
                util.dump_mrp_image(
                    mrp_path=path_mrp_s, index=i, save_path=os.path.join(dirpath, f'{fname}.s.{fmt}'),
                    image_format=fmt
                )
                util.dump_mrp_image(
                    mrp_path=path_mrp_g, index=i, save_path=os.path.join(dirpath, f'{fname}.g.{fmt}'),
                    image_format=fmt
                )
            except:
                pass

    def _validate(self):
        assert len(self._preds) == len(self._golds), 'The number of predictions and golds should be the same.'

    def _run_scorer(self, fw: str) -> Dict:
        """
        Run the scorer script for evaluation of a specific framework (fw)

        Returns
        ----------
        res : Dict
            The evaluation results
        """
        with tempfile.TemporaryDirectory() as temp_path:
            system_path = os.path.join(temp_path, 'system.mrp')
            gold_path = os.path.join(temp_path, 'gold.mrp')
            util.dump_jsonl(fpath=system_path, jsonl=[d['mrp'] for d in self._preds if d['mrp']['framework'] == fw])
            util.dump_jsonl(fpath=gold_path, jsonl=[d['mrp'] for d in self._golds if d['mrp']['framework'] == fw])
            subproc = subprocess.run(
                shlex.split(
                    f'python {scorer.__file__} -system {system_path} -gold {gold_path}'
                ),
                encoding='utf-8',
                stdout=subprocess.PIPE
            )
        return json.loads(subproc.stdout)

