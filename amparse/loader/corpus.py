# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

import copy
import json
import logging
from typing import Dict, List


class Sentence(object):
    """
    Sentence class
    """
    def __init__(self, data: Dict):
        self.id: str = data['id']
        self.data = data

    @property
    def framework(self) -> str:
        return self.data['framework']

    @framework.setter
    def framework(self, value: str):
        assert isinstance(value, str) and value.strip()
        self.data['framework'] = value

    @classmethod
    def from_json(cls, data: Dict):
        """
        Load sentence from id and data dictionary

        Parameters
        ----------
        data : Dict
            The mrp dictionary

        Returns
        ----------
        sentence : Sentence
            The sentence instance
        """
        data = copy.deepcopy(data)
        return cls(data)


class Corpus(object):
    """
    Corpus class
    """
    def __init__(self, sentences: List[Sentence]):
        super(Corpus, self).__init__()
        self.sentences = sentences
        return

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_sentence={len(self)})'

    def __getitem__(self, index):
        return self.sentences[index]

    @classmethod
    def load_from_sentences(cls, sentences: List[Sentence]):
        """
        Load corpus from sentence instances

        Parameters
        ----------
        sentences : List[Sentence]
            The list of sentence instance

        Returns
        ----------
        corpus : Corpus
            The corpus instance
        """
        return cls(sentences)

    @classmethod
    def load_from_dump(cls, dumps: List[str]):
        """
        Load corpus from the data dump

        Parameters
        ----------
        dumps : List[str]
            The data dump lines (json line for each element in the list)

        Returns
        ----------
        corpus : Corpus
            The corpus instance
        """
        import amparse.common.validate_mrp as validate_mrp

        sentences = []
        for dump in dumps:
            jd = json.loads(dump)

            validate_mrp.test_mrp(mrp=jd)

            assert 'id' in jd, '"id" must be described'
            assert 'input' in jd, '"input" must be described'
            assert 'framework' in jd, '"framework" must be described'

            if not jd['input'].strip():
                logging.warning(f'{jd["input"]} text is empty. Skip loading.')
                continue

            s = Sentence.from_json(data=jd)

            sentences.append(s)

        return cls(sentences)
