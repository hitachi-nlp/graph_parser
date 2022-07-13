"""
LICENSE Note

The following method uses modified code of https://github.com/kuribayashi4/span_based_argumentation_parser
- read_mircrotexts
The following codes are obtained from https://github.com/kuribayashi4/span_based_argumentation_parser
- argmicro/emnlp2015/util/arggraph.py
- argmicro/emnlp2015/util/folds.py
The original license (MIT) of the codes is as follows:

---
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
---

We also note that the original codes of the above mentioned codes are by Peldszus and Stede 2015 available at https://github.com/peldszus/emnlp2015
"""

import os
import glob
import logging
import random
from collections import defaultdict
from utils.converter.argmicro.emnlp2015.util.folds import folds
from utils.converter.argmicro.emnlp2015.util.arggraph import ArgGraph

from dataclasses import dataclass, field
from transformers import HfArgumentParser

import amparse.common.util as util

TOT_MTC = 112


@dataclass
class Arguments:
    """
    Arguments
    """
    dir_mtc: str = field(
        default=None,
        metadata={'help': 'The input directory path which contains .txt and .xml files'},
    )
    prefix: str = field(
        default='MTC_',
        metadata={'help': 'The prefix for component labels and edges'},
    )
    seed: int = field(
        default=42,
        metadata={'help': 'The random seed for validation splitting'},
    )
    output_cv_prefix: str = field(
        default=None,
        metadata={'help': 'The path prefix of output CV mrp files'},
    )
    output: str = field(
        default=None,
        metadata={'help': 'The output mrp file path'},
    )
    dev_rate: float = field(
        default=.1,
        metadata={'help': 'The ratio of development data in training data'},
    )
    language: str = field(
        default='en',
        metadata={'help': 'The used language'},
    )


def read_mircrotexts(conf: Arguments):
    def _make_author_folds(all_tids, tid2id):
        folds_50 = defaultdict(list)
        set_all_tids = set(all_tids)

        for i_fold, test_tids in enumerate(folds):
            i_iteration = i_fold // 5
            set_test_tids = set(test_tids)
            assert i_iteration < 10
            assert len(set_test_tids) < 26

            train_tids = set_all_tids - set_test_tids
            train_ids = list([tid2id[tid] for tid in train_tids])
            test_ids = list([tid2id[tid] for tid in test_tids])
            folds_50[i_iteration].append([train_ids, test_ids])
        folds_50 = dict(folds_50)
        return folds_50

    all_tids = sorted(list(set([tid for fold in folds for tid in fold])))
    set_all_tids = set(all_tids)
    assert len(set_all_tids) == 112

    tid2id = defaultdict(lambda: len(tid2id)*2+1)
    for tid in all_tids:
        tid2id[tid]
    tid2id = dict(tid2id)

    folds_50 = _make_author_folds(all_tids=all_tids, tid2id=tid2id)
    mrps = []
    for tid, text_i in tid2id.items():
        g = ArgGraph()
        g.load_from_xml(os.path.join(conf.dir_mtc, f'{tid}.xml'))

        # text information
        relations = g.get_adus_as_dependencies()
        relations = sorted(relations, key=lambda x: int(x[0].replace('a', '')))

        body = ''
        tops = [g.get_central_claim()]
        nodes = []
        edges = []
        for ac_i, relation in enumerate(relations):
            adu_id = relation[0].lstrip("a")
            text = g.nodes["e"+adu_id]["text"]

            nodes.append({
                'id': relation[0],
                'label': g.get_adu_role(relation[0]),
                'anchors': [{"from": len(body), "to": len(body) + len(text)}],
            })
            body += text
            if ac_i != len(relations) - 1:
                body += ' '

            relation_type = relation[2]
            target = relation[1]
            if relation_type == "add":
                while(relation_type == "add"):
                    for rel in relations:
                        if rel[0] == target:
                            relation_type = rel[2]
                            target = rel[1]
                        if relation_type != "add":
                            break
            if relation_type != 'ROOT':
                edges.append({
                    'source': relation[0],
                    'target': target,
                    'label': relation_type
                })

        # Reassign node id to make the id starts with zero
        nodes = sorted(nodes, key=lambda x: x['anchors'][0]['from'])
        nid2newid = {n['id']: i for i, n in enumerate(nodes)}
        for node in nodes:
            node['id'] = nid2newid[node['id']]
            node['label'] = conf.prefix + node['label']
        for edge in edges:
            edge['source'] = nid2newid[edge['source']]
            edge['target'] = nid2newid[edge['target']]
            edge['label'] = conf.prefix + edge['label']
        tops = [nid2newid[t] for t in tops]

        assert body == g.get_unsegmented_text()

        mrp = {
            "id": tid,
            "input": body,
            "framework": "mtc",
            "time": "2020-08-05",
            "flavor": 0,
            "version": 1.0,
            "language": conf.language,
            "provenance": "https://github.com/peldszus/arg-microtexts",
            "source": "https://github.com/peldszus/arg-microtexts",
            "nodes": nodes,
            "edges": edges,
            "tops": tops
        }
        mrp = util.reverse_edge(mrp=mrp)
        mrp = util.sort_mrp_elements(mrp=mrp)
        mrps.append(mrp)
    return mrps, folds_50, tid2id


def dump_cv_mrps(output_cv_prefix: str, dev_rate: float, mrps, folds_50, tid2id):
    i_fold = 0
    for i_iter, _folds in sorted(folds_50.items(), key=lambda x: x[0]):

        val_ids = []

        for train_ids, test_ids in _folds:
            train_jds = [[m for m in mrps if tid2id[m['id']] == _id][0]
                         for _id in train_ids]
            test_jds = [[m for m in mrps if tid2id[m['id']] == _id][0]
                         for _id in test_ids]
            assert len(train_jds) + len(test_jds) == TOT_MTC
            train_jds = sorted(train_jds, key=lambda x: x['id'])
            test_jds = sorted(test_jds, key=lambda x: x['id'])
            dev_ids = [jd['id'] for jd in train_jds]
            random.shuffle(dev_ids)
            dev_ids = dev_ids[:int(len(train_jds) * dev_rate)]
            train_jds, dev_jds = [jd for jd in train_jds if jd['id'] not in dev_ids], \
                                 [jd for jd in train_jds if jd['id'] in dev_ids]

            # Validation
            assert len(train_jds) + len(test_jds) + len(dev_jds) == TOT_MTC
            train_ids = [jd['id'] for jd in train_jds]
            dev_ids = [jd['id'] for jd in dev_jds]
            test_ids = [jd['id'] for jd in test_jds]
            assert not (set(train_ids) & set(dev_ids) & set(test_ids))
            assert not (set(val_ids) & set(test_ids))
            val_ids += test_ids

            util.dump_jsonl(fpath=os.path.join(f'{output_cv_prefix}.cv{i_fold}.train.mrp'), jsonl=train_jds)
            util.dump_jsonl(fpath=os.path.join(f'{output_cv_prefix}.cv{i_fold}.dev.mrp'), jsonl=dev_jds)
            util.dump_jsonl(fpath=os.path.join(f'{output_cv_prefix}.cv{i_fold}.test.mrp'), jsonl=test_jds)
            i_fold += 1

        # Validation
        all_ids = [m['id'] for m in mrps]
        assert len(val_ids) == len(all_ids) == len(set(val_ids) & set(all_ids))
    return


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)

    random.seed(conf.seed)

    # Load files
    xml_files = glob.glob(os.path.join(conf.dir_mtc, '*.xml'), recursive=True)
    # Sort the files
    logging.info(xml_files)

    mrps, folds_50, tid2id = read_mircrotexts(conf=conf)
    mrps = sorted(mrps, key=lambda x: x['id'])
    assert len(mrps) == TOT_MTC

    util.try_mkdir(os.path.dirname(conf.output))
    util.dump_jsonl(fpath=conf.output, jsonl=mrps)

    dump_cv_mrps(output_cv_prefix=conf.output_cv_prefix, mrps=mrps,
                 folds_50=folds_50, tid2id=tid2id, dev_rate=conf.dev_rate)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
