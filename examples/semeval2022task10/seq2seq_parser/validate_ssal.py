# coding: utf-8
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import json
import typer
import logging
from typing import List, Dict, Optional, Tuple, Set

app = typer.Typer()

logger = logging.getLogger(__name__)


def opinion_to_tuples(opinion: Dict) -> Set[Tuple[str, str, str]]:
    exps: List[str] = opinion["Polar_expression"][0]
    srcs: List[str] = opinion["Source"][0]
    trgs: List[str] = opinion["Target"][0]

    res = []
    for exp in exps:
        res.append(('root', exp.strip(), 'root'))

        for src in srcs:
            res.append((exp.strip(), src.strip(), 'src'))

        for trg in trgs:
            res.append((exp.strip(), trg.strip(), 'trg'))

    return set(res)


def check_opinion_tuples(opinion: Dict, references: List[Dict]) -> Optional[Dict]:
    op_set = opinion_to_tuples(opinion)

    for ref in references:
        rf_set = opinion_to_tuples(ref)
        if rf_set == op_set:
            return ref

    logger.error("no matching opinions: %s in %s", opinion, references)
    return None


def _check_spans(text, a, b):
    for exp, span in zip(a[0], a[1]):
        for rexp, rspan in zip(b[0], b[1]):
            if exp.strip() != rexp.strip():
                continue
            if span != rspan:
                try:
                    span = [int(s) for s in span.split(":")]
                    rspan = [int(s) for s in rspan.split(":")]
                    logger.error("exp: %s, anchor mismatch: original %s: %s, reconstructed %s: %s", exp, rspan, text[rspan[0]:rspan[1]], span, text[span[0]:span[1]])
                except:
                    logger.error("exp %s, anchor mismatch: original %s, reconstructed %s", exp, rspan, span)


def check_polarity(opinion: Dict, reference: Dict) -> None:
    if opinion["Polarity"] != reference["Polarity"]:
        logger.error("Polarity mismatch original %s, reconstructed %s", reference["Polarity"], opinion["Polarity"])
        
    if opinion["Intensity"] != reference["Intensity"]:
        logger.error("Intensity mismatch original %s, reconstructed %s", reference["Intensity"], opinion["Intensity"])


def check_anchors(text: str, opinion: Dict, reference: Dict) -> None:

    logger.info("check Polar_expression")
    _check_spans(text, opinion["Polar_expression"], reference["Polar_expression"])
    
    logger.info("check Source")
    _check_spans(text, opinion["Source"], reference["Source"])
    
    logger.info("check Target")
    _check_spans(text, opinion["Target"], reference["Target"])


def check_data(data: Dict, reference: Dict) -> None:
    logger.info("checking %s", data["sent_id"])
    for opinion in data["opinions"]:
        ref = check_opinion_tuples(opinion, reference["opinions"])

        if ref is None:
            continue

        check_polarity(opinion, ref)
        check_anchors(reference["text"], opinion, ref)
        

@app.command()
def check(gold: str, system: str):
    with open(gold) as f:
        gold = json.load(f)
        gold = {g["sent_id"]:g for g in gold}

    with open(system) as f:
        system = json.load(f)

    for s in system:
        sid = s["sent_id"]
        if sid not in gold:
            logger.error("%s is not found in gold data.", sid)
            continue

        g = gold[sid]
        check_data(s, g)


if __name__ == "__main__":
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    app()
