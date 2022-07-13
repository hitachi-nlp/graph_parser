# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import copy
import numpy as np
from typing import Dict

from amparse.loader.corpus import Sentence
from amparse.loader.vocab import AMVocab
from amparse.postprocess.base_postprocess import BasePostProcess
import amparse.common.util as util
import amparse.common.params as P


@BasePostProcess.register("ssa")
class SsaPostProcess(BasePostProcess):
    def reconstruct(self, sentence: Sentence, pred_data: Dict, edge_vocab: AMVocab):
        nodes = []
        for span, prop in zip(pred_data['span'], pred_data['proposition']):
            nodes.append({
                'id': len(nodes),
                'anchors': [{"from": span.start, "to": span.stop}],
                'label': prop,
            })
        arc = copy.deepcopy(pred_data['arc'])
        assert 0 <= np.min(arc) <= np.max(arc) <= 1
        arc[:, -1] = -1000.
        arc[-1, :] = -1000.
        links = list(zip(*np.nonzero(arc >= 0.5)))
        rel = copy.deepcopy(pred_data['rel'])
        rel[:, :, edge_vocab.token2id(P.TOP)] = -1000.

        edges = []
        for frm, to in links:
            frm, to = int(frm), int(to)
            rel_id = np.argmax(rel[frm, to])
            rel_label = edge_vocab.id2token(rel_id)
            assert rel_label != P.TOP
            edges.append({
                'source': frm,
                'target': to,
                'label': rel_label
            })

        tops = []
        for node in nodes:
            in_edges = [e for e in edges if e['target'] == node['id'] and e['source'] != e['target']]
            if not in_edges:
                tops.append(node['id'])

        d = {
            "id": sentence.id,
            "input": sentence.data['input'],
            "flavor": 0,
            "language": sentence.data['language'] if 'language' in sentence.data else '',
            "framework": sentence.framework,
            "time": util.now(),
            "version": sentence.data['version'] if 'version' in sentence.data else '',
            "provenance": sentence.data['provenance'] if 'provenance' in sentence.data else '',
            "source": sentence.data['source'] if 'source' in sentence.data else '',
            "tops": tops,
            "edges": edges,
            "nodes": nodes,
        }
        return d
