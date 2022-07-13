# coding: utf-8
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


import json
import typer
from typing import List, Dict

app = typer.Typer()


def to_dump(exp: List[str]) -> List[str]:
    res = []

    for token in exp:
        res.append(token)
        res.append('[SEP]')

    if len(res) > 1:
        res.pop()
    return res


def to_ssa(opinions: List[Dict], use_intensity: bool) -> List[str]:
        _dump = []

        for opn in opinions:
            _dump.extend(to_dump(opn["Polar_expression"][0]))
            _dump.append(f'[{opn["Polarity"]}]')
            if use_intensity:
                _dump.append(f'[{opn["Intensity"]}]')
            _dump.extend(to_dump(opn["Target"][0]))
            _dump.append('[TGT]')
            _dump.extend(to_dump(opn["Source"][0]))
            _dump.append('[SRC]')

        return _dump


@app.command()
def ssa(fname: str,  use_intensity: bool= typer.Option(False)):
    with open(fname) as f:
        data = json.load(f)
        for js in data:
            js["dump"] = " ".join(to_ssa(js["opinions"], use_intensity))
            print(json.dumps(js, ensure_ascii=False))


if __name__ == "__main__":
    app()
