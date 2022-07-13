"""
LICENSE Note

The following methods use modified code of https://github.com/vene/marseille/
- link_transitive
- merge_spans
- merge_prop_labels
The original license of the code is as follows:

---

BSD 3-Clause License

Copyright (c) 2017, Vlad Niculae
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import glob
import json
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import amparse.common.util as util


@dataclass
class Arguments:
    """
    Arguments
    """
    dir_cdcp: str = field(
        default=None,
        metadata={'help': 'The input directory path which contains .txt and .ann.json files'},
    )
    prefix: str = field(
        default='CDCP_',
        metadata={'help': 'The prefix for component labels and edges'},
    )
    output: str = field(
        default=None,
        metadata={'help': 'The output directory path'},
    )


def read_cdcp(conf: Arguments, ann_path: str, txt_path: str):
    assert os.path.basename(ann_path).split('.')[0] == os.path.basename(txt_path).split('.')[0]
    # Read json file
    with open(ann_path, 'r') as f:
        lines = [line for line in f]
        assert len(lines) == 1
        jd = json.loads(lines[0])
    # Read body file
    with open(txt_path, 'r') as f:
        lines = [line for line in f]
        assert len(lines) == 1
        txt = lines[0]

    props, prop_offsets, links, prop_labels = merge_spans(json_dict=jd)
    links = link_transitive(links=links)

    nodes = []
    for i, (label, offset) in enumerate(zip(prop_labels, prop_offsets)):
        while txt[offset[0]] == ' ':
            offset = (offset[0] + 1, offset[1])
        nodes.append({
            "id": i,
            "label": conf.prefix + label,
            "anchors": [{"from": offset[0], "to": offset[1]}],
        })

    edges = []
    for src, trg, label in links:
        edges.append({
            "source": src,
            "target": trg,
            "label": conf.prefix + label
        })

    tops = []
    for node in nodes:
        out_edges = [e for e in edges if e['source'] == node['id']]
        if not out_edges:
            tops.append(node['id'])

    mrp = {
        "id": os.path.basename(ann_path).replace('.ann.json', ''),
        "input": txt,
        "framework": 'cdcp',
        "time": "2020-08-05",
        "flavor": 0,
        "version": 1.0,
        "language": "en",
        "provenance": 'https://facultystaff.richmond.edu/~jpark/',
        "source": 'https://facultystaff.richmond.edu/~jpark/',
        "nodes": nodes,
        "edges": edges,
        "tops": tops,
    }
    mrp = util.reverse_edge(mrp=mrp)
    mrp = util.sort_mrp_elements(mrp=mrp)
    return mrp


def link_transitive(links):
    """perform transitive closure of links.
    For input [(1, 2), (2, 3)] the output is [(1, 2), (2, 3), (1, 3)]
    """
    links = set(links)
    while True:
        new_links = [(src_a, trg_b, label_b)
                     for src_a, trg_a, label_a in links for src_b, trg_b, label_b in links
                     if trg_a == src_b
                     and (src_a, trg_b, label_b) not in links]
        if new_links:
            links.update(new_links)
        else:
            break
    return links


def merge_spans(json_dict, include_nonarg=True):
    """Normalization needed for CDCP data because of multi-prop spans"""

    # flatten multi-prop src spans like (3, 6) into new propositions
    # as long as they never overlap with other links. This inevitably will
    # drop some data but it's a very small number.

    # function fails if called twice because
    #    precondition: doc.links = [((i, j), k)...]
    #    postcondition: doc.links = [(i, k)...]

    url = {int(key): val for key, val in json_dict['url'].items()}
    prop_labels = json_dict['prop_labels']
    prop_offsets = [(int(a), int(b)) for a, b in json_dict['prop_offsets']]
    reasons = [((int(a), int(b)), int(c)) for (a, b), c in json_dict['reasons']]
    evidences = [((int(a), int(b)), int(c)) for (a, b), c in json_dict['evidences']]
    links = reasons + evidences
    link_labels = ['reason' for _ in reasons] + ['evidence' for _ in evidences]

    new_links = []
    new_props = {}
    new_prop_offsets = {}

    dropped = 0

    for ((start, end), trg), l_label in zip(links, link_labels):

        if start == end:
            new_props[start] = (start, end)
            new_prop_offsets[start] = prop_offsets[start]

            new_props[trg] = (trg, trg)
            new_prop_offsets[trg] = prop_offsets[trg]

            new_links.append((start, trg, l_label))

        elif start < end:
            # multi-prop span. Check for problems:

            problems = []
            for (other_start, other_end), other_trg in links:
                if start == other_start and end == other_end:
                    continue

                # another link coming out of a subset of our span
                if start <= other_start <= other_end <= end:
                    problems.append(((other_start, other_end), other_trg))

                # another link coming into a subset of our span
                if start <= other_trg <= end:
                    problems.append(((other_start, other_end), other_trg))

            if not len(problems):
                if start in new_props:
                    assert (start, end) == new_props[start]

                new_props[start] = (start, end)
                new_prop_offsets[start] = (prop_offsets[start][0],
                                           prop_offsets[end][1])

                new_props[trg] = (trg, trg)
                new_prop_offsets[trg] = prop_offsets[trg]

                new_links.append((start, trg, l_label))

            else:
                # Since we drop the possibly NEW span, there is no need
                # to remove any negative links.
                dropped += 1

    if include_nonarg:
        used_props = set(k for a, b in new_props.values()
                         for k in range(a, b + 1))
        for k in range(len(prop_offsets)):
            if k not in used_props:
                new_props[k] = (k, k)
                new_prop_offsets[k] = prop_offsets[k]

    mapping = {key: k for k, key in enumerate(sorted(new_props))}
    res_props = [val for _, val in sorted(new_props.items())]
    res_prop_offsets = [val for _, val in sorted(new_prop_offsets.items())]
    res_links = [(mapping[src], mapping[trg], label) for src, trg, label in new_links]
    res_prop_labels = [merge_prop_labels(prop_labels[a:1 + b]) for a, b in res_props]

    return res_props, res_prop_offsets, res_links, res_prop_labels


def merge_prop_labels(labels):
    """After joining multiple propositions, we need to decide the new type.
    Rules:
        1. if the span is a single prop, keep the label
        2. if the span props have the same type, use that type
        3. Else, rules from Jon: policy>value>testimony>reference>fact
    """

    if len(labels) == 1:
        return labels[0]

    labels = set(labels)

    if len(labels) == 1:
        return next(iter(labels))

    if 'policy' in labels:
        return 'policy'
    elif 'value' in labels:
        return 'value'
    elif 'testimony' in labels:
        return 'testimony'
    elif 'reference' in labels:
        return 'reference'
    elif 'fact' in labels:
        return 'fact'
    else:
        raise ValueError("weird labels: {}".format(" ".join(labels)))


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)
    # Load files
    ann_files = glob.glob(os.path.join(conf.dir_cdcp, '*.ann.json'), recursive=True)
    txt_files = glob.glob(os.path.join(conf.dir_cdcp, '*.txt'), recursive=True)
    # Sort the files
    ann_files, txt_files = sorted(ann_files), sorted(txt_files)
    assert len(ann_files) == len(txt_files)
    logging.info(ann_files)
    logging.info(txt_files)

    mrps = []
    for ann, txt in zip(ann_files, txt_files):
        mrp = read_cdcp(conf=conf, ann_path=ann, txt_path=txt)
        mrps.append(mrp)

    os.makedirs(os.path.dirname(conf.output), exist_ok=True)
    util.dump_jsonl(fpath=conf.output, jsonl=mrps)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
