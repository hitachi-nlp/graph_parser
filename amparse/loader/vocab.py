# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import copy
from collections import Counter
from typing import Dict, Optional


class AMVocab(object):
    """
    Vocabulary class
    """
    def __init__(self,
                 counter: Optional[Counter] = None,
                 min_freq: int = 0,
                 special_token2id: Optional[Dict[str, int]] = None
                 ):
        assert special_token2id is None or isinstance(special_token2id, dict)
        assert counter is None or isinstance(counter, Counter)

        self._specials = copy.deepcopy(special_token2id) if special_token2id is not None else dict()
        self._token2id = copy.deepcopy(self._specials)
        self._id2token = {i: token for token, i in self._token2id.items()}

        if counter is not None:
            self.extend([token for token, freq in counter.items() if freq >= min_freq])

    def __len__(self):
        return len(self._token2id)

    def __contains__(self, token):
        return token in self._token2id

    def __repr__(self):
        return f"""{self.__class__.__name__}: (n_all={self.n_vocab},  special_tokens={self.special_tokens[:300]}), content={list(self._token2id.items())[:300]})"""

    @property
    def n_vocab(self):
        return len(self._token2id)

    @property
    def n_specials(self):
        return len(self._specials)

    @property
    def special_tokens(self):
        return list(self._specials.keys())

    @property
    def tokens(self):
        return list(self._token2id.keys())

    @property
    def ids(self):
        return list(self._token2id.values())

    @property
    def max_id(self):
        return max(self._id2token.keys())

    @property
    def min_id(self):
        return min(self._id2token.keys())

    def token2id(self, token):
        return self._token2id[token]

    def tokens2ids(self, tokens):
        return [self._token2id[t] for t in tokens]

    def id2token(self, i):
        return self._id2token[i]

    def ids2tokens(self, ids):
        return [self._id2token[i] for i in ids]

    def extend(self, tokens, auto_sort=True):
        max_id = (max(list(self._token2id.values())) if len(self._token2id) > 0 else -1) + 1
        new_tokens = list(set(tokens).difference(self._token2id.keys()))
        if auto_sort:
            new_tokens = sorted(new_tokens)
        for i, token in enumerate(new_tokens):
            self._token2id[token] = max_id + i
            self._id2token[max_id + i] = token

    def extend_with_dict(self, d: Dict[str, int]):
        for token, i in d.items():
            self._token2id[token] = i
            self._id2token[i] = token

