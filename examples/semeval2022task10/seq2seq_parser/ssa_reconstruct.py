# coding: utf-8
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


import re
import json
import typer
from typing import List, Dict, Tuple
from simalign import SentenceAligner


app = typer.Typer()
# making an instance of our model.
# You can specify the embedding model and all alignment settings in the constructor.
myaligner = SentenceAligner(model="xlmr", token_type="bpe", matching_methods="f")


T5_map= {
    '[Positive]': '<extra_id_0>',
    '[Negative]': '<extra_id_1>',
    '[Neutral]': '<extra_id_2>',
    '[Average]': '<extra_id_3>',
    '[Strong]': '<extra_id_4>',
    '[Weak]': '<extra_id_5>',
    '[TGT]': '<extra_id_6>',
    '[SRC]': '<extra_id_7>',
    '[SEP]': '<extra_id_8>',
    '[Standard]': '<extra_id_9>',
}


def replace_tokens(seq: str) -> str:
    for k, v in T5_map.items():
        seq = seq.replace(v, f' {k} ')
    return seq


def to_list(data: List[str]) -> List[str]:
    buf = []
    lst = []
    for token in data:
        if token == '[SEP]':
            lst.append(' '.join(buf))
            buf = []
            continue
        buf.append(token)

    if len(buf) > 0:
        lst.append(' '.join(buf))
    
    return [lst, []]


def regularize_opinion(opn: Dict) -> Dict:
    for key in ["Polar_expression", "Target", "Source"]:
        if key not in opn:
            opn[key] = [[], []]
    if "Polarity" not in opn:
        opn["Polarity"] = "Positive"

    return opn


def greedy_loader(seq: str) -> Tuple[str, List[Dict]]:
    split_ptn = re.compile(r'\s+')
    if seq.startswith('</s>'):
        seq = seq[len('</s>'):]
    if '</s>' in seq:
        seq = seq.split('</s>')[0]

    for mbart_lang in "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ," \
                      "ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN".split(','):
        seq = seq.replace(mbart_lang, "")
    seq = seq.replace("<pad>", "")

    seq = seq.strip()

    seq = replace_tokens(seq)
    tokens: List[str] = split_ptn.split(seq)

    buf = []
    opnions = []
    current = {}

    for token in tokens:
        if token in ["[Positive]", "[Negative]", "[Neutral]"]:
            current["Polarity"] = token.replace('[', '').replace(']', '')
            if len(buf) > 0:
                current["Polar_expression"] = to_list(buf)
                buf.clear()
            else:
                current["Polar_expression"] = [[], []]
            continue
        if token in ["[Average]", "[Strong]", "[Weak]", "[Standard]"]:
            current["Intensity"] = token.replace('[', '').replace(']', '')
            continue

        if token == '[TGT]':
            if len(buf) > 0:
                current["Target"] = to_list(buf)
                buf.clear()
            else:
                current["Target"] = [[], []]
            continue
        
        if token == '[SRC]':
            if len(buf) > 0:
                current["Source"] = to_list(buf)
                buf.clear()
            else:
                current["Source"] = [[], []]
            if len(current) > 0:
                opnions.append(regularize_opinion(current))
                current = {}
            continue

        buf.append(token)

    if len(current) > 0:
        opnions.append(regularize_opinion(current))

    return seq, opnions


def find_align(index: int, alignments: List[Tuple[int]]):
    for src, tgt in alignments:
        if src == index:
            return tgt
    
    return None


def add_alignment(text: str, opinion: Dict):
    seq = []
    maps = []
    start = 0

    for exp in opinion["Polar_expression"][0]:
        tokens = [t for t in exp.split(" ") if len(t) > 0]
        seq.extend(tokens)
        maps.append((start, start + len(tokens), "Polar_expression", exp))
        start += len(tokens)
    
    for exp in opinion["Target"][0]:
        tokens = [t for t in exp.split(" ") if len(t) > 0]
        seq.extend(tokens)
        maps.append((start, start + len(tokens), "Target", exp))
        start += len(tokens)
        
    for exp in opinion["Source"][0]:
        tokens = [t for t in exp.split(" ") if len(t) > 0]
        seq.extend(tokens)
        maps.append((start, start + len(tokens), "Source", exp))
        start += len(tokens)

    tgt_tokens = text.split(' ')
    tgt_lens = [len(t) for t in tgt_tokens]

    try:
        alignments = myaligner.get_word_aligns(seq, tgt_tokens)['fwd']
    except:
        return opinion

    for begin, end, t, exp in maps:
        spans = []
        for i in range(begin, end):
            tgt = find_align(i, alignments)
            if tgt is not None:
                spans.append(tgt)
        
        if len(spans) > 0:
            _s = min(spans)
            _e = max(spans)

            s = sum(tgt_lens[0:_s]) + _s # add _s because length of space " "
            e = sum(tgt_lens[0:_e+1]) + _e # add _e because length of space " "
            
            repr = text[s:e]
            if repr.strip() != exp.strip():
                i = text.find(exp.strip())
                if i >= 0:
                    s = i
                    e = i + len(exp.strip())

            lst: List[str] = opinion[t][1]
            lst.append(f'{s}:{e}')

    return opinion

    
@app.command()
def reconstruct(genfile: str, jsonl: str, debug: bool = False):
    with open(genfile) as f:
        gen_lines = [l for l in f.readlines() if l.strip()]

    with open(jsonl) as f:
        json_lines = [l for l in f.readlines() if l.strip()]

    assert len(gen_lines) == len(json_lines)

    js_list = []
    for g, js in zip(gen_lines, json_lines):
        g = g.strip()
        pred, opinions = greedy_loader(g)
        js = json.loads(js)
        opinions = [add_alignment(js["text"], op) for op in opinions]
        js['opinions'] = opinions
        if debug:
            js['pred'] = pred
        else:
            js.pop('dump', None)
            js.pop('dump_t5', None)
        js_list.append(js)
    print(json.dumps(js_list, ensure_ascii=False))


if __name__ == "__main__":
    app()
