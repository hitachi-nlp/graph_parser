# coding: utf-8
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


import json
import sys
import typer
from typing import Dict

app = typer.Typer()


def replace_token_t5(text: str) -> str:
    text = text.replace('[Positive]', '<extra_id_0>')
    text = text.replace('[Negative]', '<extra_id_1>')
    text = text.replace('[Neutral]', '<extra_id_2>')
    text = text.replace('[Average]', '<extra_id_3>')
    text = text.replace('[Strong]', '<extra_id_4>')
    text = text.replace('[Weak]', '<extra_id_5>')
    text = text.replace('[TGT]', '<extra_id_6>')
    text = text.replace('[SRC]', '<extra_id_7>')
    text = text.replace('[SEP]', '<extra_id_8>')
    text = text.replace('[Standard]', '<extra_id_9>')

    return text


def check_span(text: str, surface: str, span: str)-> bool:
    span = [int(t) for t in span.split(':')]
    exp = text[span[0]:span[1]]

    if exp == surface:
        return True
    else:
        print(text, surface, exp, file=sys.stderr)
        return False


def check_spans(data: Dict) -> bool:
    text = data["text"]
    for opn in data["opinions"]:
        for k, v in zip(*opn['Source']):
            if not check_span(text, k, v):
                return False

        for k, v in zip(*opn['Polar_expression']):
            if not check_span(text, k, v):
                return False
                
        for k, v in zip(*opn['Target']):
            if not check_span(text, k, v):
                return False

    return True


@app.command()
def convert(fname: str, check_span: bool = typer.Option(True), skip_empty: bool = typer.Option(False)):
    with open(fname) as f:
        for line in f:
            js = json.loads(line)
            if skip_empty:
                if len(js.get("opinions", [])) < 1:
                    continue
                
            if check_span and not check_spans(js):
                continue

            js["dump_t5"] = replace_token_t5(js["dump"])
            print(json.dumps(js))


if __name__ == "__main__":
    app()
