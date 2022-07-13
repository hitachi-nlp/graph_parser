# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import unittest
import os
import shlex
import subprocess
import json
from typing import Dict, List
import tempfile
import amparse.common.util as util
import amparse.evaluator.scorer as scorer


def eval_mrp(s: List[Dict], g: List[Dict]):
    with tempfile.TemporaryDirectory() as d:
        fs = os.path.join(d, 's')
        fg = os.path.join(d, 'g')
        util.dump_jsonl(fpath=fs, jsonl=s)
        util.dump_jsonl(fpath=fg, jsonl=g)
        subproc = subprocess.run(
            shlex.split(f'python {scorer.__file__} -system {fs} -gold {fg}'),
            encoding='utf-8',
            stdout=subprocess.PIPE
        )
    return json.loads(subproc.stdout)


class TestAMScorer(unittest.TestCase):
    """
    Test class of evaluator/scorer.py
    """

    def test1(self):
        # Exact match
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        gold = system
        res = eval_mrp(system, gold)
        self.assertEqual(1., res['anchors']['p'])
        self.assertEqual(1., res['anchors']['r'])
        self.assertEqual(1., res['anchors']['f'])
        self.assertEqual(1., res['labels']['Premise']['p'])
        self.assertEqual(1., res['labels']['Premise']['r'])
        self.assertEqual(1., res['labels']['Premise']['f'])
        self.assertEqual(1., res['labels']['Claim']['p'])
        self.assertEqual(1., res['labels']['Claim']['r'])
        self.assertEqual(1., res['labels']['Claim']['f'])
        self.assertEqual(1., res['labels']['total']['p'])
        self.assertEqual(1., res['labels']['total']['r'])
        self.assertEqual(1., res['labels']['total']['f'])
        self.assertEqual(1., res['edges']['Support']['p'])
        self.assertEqual(1., res['edges']['Support']['r'])
        self.assertEqual(1., res['edges']['Support']['f'])
        self.assertEqual(1., res['edges']['total']['p'])
        self.assertEqual(1., res['edges']['total']['r'])
        self.assertEqual(1., res['edges']['total']['f'])

    def test2(self):
        # Partial anchor match
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        gold = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 30}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 105}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        res = eval_mrp(system, gold)
        self.assertEqual(1 / 4, res['anchors']['p'])
        self.assertEqual(1 / 3, res['anchors']['r'])
        self.assertEqual((2 * (1/4) * (1/3)) / (1/4 + 1/3), res['anchors']['f'])
        self.assertEqual(1 / 2, res['labels']['Premise']['p'])
        self.assertEqual(1 / 2, res['labels']['Premise']['r'])
        self.assertEqual((2 * (1/2) * (1/2)) / (1/2 + 1/2), res['labels']['Premise']['f'])
        self.assertEqual(0 / 2, res['labels']['Claim']['p'])
        self.assertEqual(0 / 1, res['labels']['Claim']['r'])
        self.assertEqual(0., res['labels']['Claim']['f'])
        self.assertEqual(1 / 4, res['labels']['total']['p'])
        self.assertEqual(1 / 3, res['labels']['total']['r'])
        self.assertEqual((2 * (1/3) * (1/4)) / (1/3 + 1/4), res['labels']['total']['f'])
        self.assertEqual(0 / 1, res['edges']['Support']['p'])
        self.assertEqual(0 / 1, res['edges']['Support']['r'])
        self.assertEqual(0., res['edges']['Support']['f'])
        self.assertEqual(0 / 1, res['edges']['total']['p'])
        self.assertEqual(0 / 1, res['edges']['total']['r'])
        self.assertEqual(0., res['edges']['total']['f'])

    def test3(self):
        # Partial label match
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        gold = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                    {'id': 3, 'label': 'Claim', 'anchors': [{'from': 30, 'to': 40}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Claim', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        res = eval_mrp(system, gold)
        self.assertEqual(4 / 4, res['anchors']['p'])
        self.assertEqual(4 / 5, res['anchors']['r'])
        self.assertEqual((2 * (4/4) * (4/5)) / (4/4 + 4/5), res['anchors']['f'])
        self.assertEqual(1 / 2, res['labels']['Premise']['p'])
        self.assertEqual(1 / 1, res['labels']['Premise']['r'])
        self.assertEqual((2 * (1/2) * (1/1)) / (1/2 + 1/1), res['labels']['Premise']['f'])
        self.assertEqual(2 / 2, res['labels']['Claim']['p'])
        self.assertEqual(2 / 4, res['labels']['Claim']['r'])
        self.assertEqual((2 * (2/2) * (2/4)) / (2/2 + 2/4), res['labels']['Claim']['f'])
        self.assertEqual(3 / 4, res['labels']['total']['p'])
        self.assertEqual(3 / 5, res['labels']['total']['r'])
        self.assertEqual((2 * (3/4) * (3/5)) / (3/4 + 3/5), res['labels']['total']['f'])
        self.assertEqual(1., res['edges']['Support']['p'])
        self.assertEqual(1., res['edges']['Support']['r'])
        self.assertEqual(1., res['edges']['Support']['f'])
        self.assertEqual(1., res['edges']['total']['p'])
        self.assertEqual(1., res['edges']['total']['r'])
        self.assertEqual(1., res['edges']['total']['f'])

    def test4(self):
        # Partial edges match
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 2, 'label': 'Support'},
                    {'source': 0, 'target': 1, 'label': 'Attack'},
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        gold = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'},
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        res = eval_mrp(system, gold)
        self.assertEqual(1, res['anchors']['p'])
        self.assertEqual(1, res['anchors']['r'])
        self.assertEqual(1, res['anchors']['f'])
        self.assertEqual(1, res['labels']['Premise']['p'])
        self.assertEqual(1, res['labels']['Premise']['r'])
        self.assertEqual(1, res['labels']['Premise']['f'])
        self.assertEqual(1, res['labels']['Claim']['p'])
        self.assertEqual(1, res['labels']['Claim']['r'])
        self.assertEqual(1, res['labels']['Claim']['f'])
        self.assertEqual(1, res['labels']['total']['p'])
        self.assertEqual(1, res['labels']['total']['r'])
        self.assertEqual(1, res['labels']['total']['f'])
        self.assertEqual(0 / 1, res['edges']['Support']['p'])
        self.assertEqual(0 / 1, res['edges']['Support']['r'])
        self.assertEqual(0, res['edges']['Support']['f'])
        self.assertEqual(0 / 2, res['edges']['total']['p'])
        self.assertEqual(0 / 1, res['edges']['total']['r'])
        self.assertEqual(0, res['edges']['total']['f'])

    def test5(self):
        # Partial edges match 2
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 2, 'label': 'Support'},
                    {'source': 0, 'target': 1, 'label': 'Attack'},
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        gold = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Attack'},
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        res = eval_mrp(system, gold)
        self.assertEqual(1, res['anchors']['p'])
        self.assertEqual(1, res['anchors']['r'])
        self.assertEqual(1, res['anchors']['f'])
        self.assertEqual(1, res['labels']['Premise']['p'])
        self.assertEqual(1, res['labels']['Premise']['r'])
        self.assertEqual(1, res['labels']['Premise']['f'])
        self.assertEqual(1, res['labels']['Claim']['p'])
        self.assertEqual(1, res['labels']['Claim']['r'])
        self.assertEqual(1, res['labels']['Claim']['f'])
        self.assertEqual(1, res['labels']['total']['p'])
        self.assertEqual(1, res['labels']['total']['r'])
        self.assertEqual(1, res['labels']['total']['f'])
        self.assertEqual(1 / 1, res['edges']['Attack']['p'])
        self.assertEqual(1 / 1, res['edges']['Attack']['r'])
        self.assertEqual(1, res['edges']['Attack']['f'])
        self.assertEqual(1 / 2, res['edges']['total']['p'])
        self.assertEqual(1 / 1, res['edges']['total']['r'])
        self.assertEqual((2 * (1/2) * (1/1)) / (1/2 + 1/1), res['edges']['total']['f'])

    def test6(self):
        # Duplication with different document ID
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            }
        ]
        gold = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        res = eval_mrp(system, gold)
        self.assertEqual(3 / 6, res['anchors']['p'])
        self.assertEqual(3 / 4, res['anchors']['r'])
        self.assertEqual((2 * (3/6) * (3/4)) / (3/6 + 3/4), res['anchors']['f'])
        self.assertEqual(1 / 2, res['labels']['Premise']['p'])
        self.assertEqual(1 / 2, res['labels']['Premise']['r'])
        self.assertEqual((2 * (1/2) * (1/2)) / (1/2 + 1/2), res['labels']['Premise']['f'])
        self.assertEqual(2 / 4, res['labels']['Claim']['p'])
        self.assertEqual(2 / 2, res['labels']['Claim']['r'])
        self.assertEqual((2 * (2/4) * (2/2)) / (2/4 + 2/2), res['labels']['Claim']['f'])
        self.assertEqual(3 / 6, res['labels']['total']['p'])
        self.assertEqual(3 / 4, res['labels']['total']['r'])
        self.assertEqual((2 * (3/6) * (3/4)) / (3/6 + 3/4), res['labels']['total']['f'])
        self.assertEqual(1 / 2, res['edges']['Support']['p'])
        self.assertEqual(1 / 1, res['edges']['Support']['r'])
        self.assertEqual((2 * (1/2) * (1/1)) / (1/2 + 1/1), res['edges']['Support']['f'])
        self.assertEqual(1 / 2, res['edges']['total']['p'])
        self.assertEqual(1 / 1, res['edges']['total']['r'])
        self.assertEqual((2 * (1/2) * (1/1)) / (1/2 + 1/1), res['edges']['total']['f'])

    def test7(self):
        # Anchor and label duplication
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                    {'id': 3, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 100, 'to': 122}]},
                ],
                'edges': [],
                'tops': [0],
            }
        ]
        gold = system
        res = eval_mrp(system, gold)
        self.assertEqual(1., res['anchors']['p'])
        self.assertEqual(1., res['anchors']['r'])
        self.assertEqual(1., res['anchors']['f'])
        self.assertEqual(1., res['labels']['Premise']['p'])
        self.assertEqual(1., res['labels']['Premise']['r'])
        self.assertEqual(1., res['labels']['Premise']['f'])
        self.assertEqual(1., res['labels']['Claim']['p'])
        self.assertEqual(1., res['labels']['Claim']['r'])
        self.assertEqual(1., res['labels']['Claim']['f'])
        self.assertEqual(1., res['labels']['total']['p'])
        self.assertEqual(1., res['labels']['total']['r'])
        self.assertEqual(1., res['labels']['total']['f'])
        self.assertEqual(1., res['edges']['Support']['p'])
        self.assertEqual(1., res['edges']['Support']['r'])
        self.assertEqual(1., res['edges']['Support']['f'])
        self.assertEqual(1., res['edges']['total']['p'])
        self.assertEqual(1., res['edges']['total']['r'])
        self.assertEqual(1., res['edges']['total']['f'])

    def test8(self):
        # edges duplication
        system = [
            {
                'id': 0,
                'nodes': [
                    {'id': 0, 'label': 'Premise', 'anchors': [{'from': 10, 'to': 15}]},
                    {'id': 1, 'label': 'Claim', 'anchors': [{'from': 20, 'to': 25}]},
                    {'id': 2, 'label': 'Claim', 'anchors': [{'from': 3, 'to': 6}]},
                ],
                'edges': [
                    {'source': 0, 'target': 1, 'label': 'Support'},
                    {'source': 0, 'target': 1, 'label': 'Support'}
                ],
                'tops': [0],
            },
            {
                'id': 1,
                'nodes': [
                    {'id': 0, 'label': 'Claim', 'anchors': [{'from': 100, 'to': 122}]},
                    {'id': 1, 'label': 'Premise', 'anchors': [{'from': 25, 'to': 50}]},
                ],
                'edges': [
                    {'source': 1, 'target': 0, 'label': 'Attack'},
                    {'source': 1, 'target': 0, 'label': 'Attack'},
                ],
                'tops': [0],
            }
        ]
        gold = system
        res = eval_mrp(system, gold)
        self.assertEqual(1., res['anchors']['p'])
        self.assertEqual(1., res['anchors']['r'])
        self.assertEqual(1., res['anchors']['f'])
        self.assertEqual(1., res['labels']['Premise']['p'])
        self.assertEqual(1., res['labels']['Premise']['r'])
        self.assertEqual(1., res['labels']['Premise']['f'])
        self.assertEqual(1., res['labels']['Claim']['p'])
        self.assertEqual(1., res['labels']['Claim']['r'])
        self.assertEqual(1., res['labels']['Claim']['f'])
        self.assertEqual(1., res['labels']['total']['p'])
        self.assertEqual(1., res['labels']['total']['r'])
        self.assertEqual(1., res['labels']['total']['f'])
        self.assertEqual(1., res['edges']['Support']['p'])
        self.assertEqual(1., res['edges']['Support']['r'])
        self.assertEqual(1., res['edges']['Support']['f'])
        self.assertEqual(1., res['edges']['Attack']['p'])
        self.assertEqual(1., res['edges']['Attack']['r'])
        self.assertEqual(1., res['edges']['Attack']['f'])
        self.assertEqual(1., res['edges']['total']['p'])
        self.assertEqual(1., res['edges']['total']['r'])
        self.assertEqual(1., res['edges']['total']['f'])

