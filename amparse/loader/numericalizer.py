# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from typing import List
import torch

from amparse.loader.fields import AMBaseField


class AMNumericalizer(object):
    """
    Numericalizer class
    """
    def __init__(self, corpus, fields: List[AMBaseField]):
        self.corpus = corpus
        self.fields = fields
        self.field_names = [field.name for field in fields]

        # Build the fields
        for f in fields:
            f.build(corpus=corpus)

    def __repr__(self):
        s = f'Numericalizer: (\n'
        s += f'\tid: {id(self)}\n'
        for field in self.fields:
            s += '\n'.join([f'\t{l}' for l in str(field).split('\n')])
            s += '\n'
        s += f'\n)'
        return s

    def field_by_name(self, name: str):
        field = [field for field in self.fields if field.name == name]
        if field:
            return field[0]
        else:
            return None

    def numericalize(self, corpus, **kwargs):
        field2numerics = {}
        for field in self.fields:
            field2numerics[field.name] = field.numericalize(corpus=corpus, **kwargs)
        return field2numerics

    def save(self, path):
        torch.save({
            'numericalizer': self,
        }, path)

    @classmethod
    def load_from(cls, path):
        state = torch.load(path)
        numericalizer = state['numericalizer']
        return numericalizer

