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


@BasePostProcess.register("abstrct")
class AbstrctPostProcess(BasePostProcess):
    def reconstruct(self, sentence: Sentence, pred_data: Dict, edge_vocab: AMVocab):
        nodes = []
        for span, prop in zip(pred_data['span'], pred_data['proposition']):
            nodes.append({
                'id': len(nodes),
                'anchors': [{"from": span.start, "to": span.stop}],
                'label': prop,
            })
        arc = copy.deepcopy(pred_data['arc'])
        arc[:, -1] = -1000.
        arc[np.eye(len(arc)).astype(np.bool)] = -1000.
        head, rev_links = util.decode_mst(arc.T)
        links = [(to, frm) for frm, to in rev_links]
        rel = copy.deepcopy(pred_data['rel'])

        tops = []
        edges = []
        for frm, to in links:
            frm, to = int(frm), int(to)
            if frm == head:
                tops.append(to)
            else:
                rel_id = np.argmax(rel[frm, to])
                rel_label = edge_vocab.id2token(rel_id)
                edges.append({
                    'source': frm,
                    'target': to,
                    'label': rel_label
                })

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
