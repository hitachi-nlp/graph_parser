# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from collections.abc import Sized
import torch
from torch.utils.data import Dataset

from amparse.loader.numericalizer import AMNumericalizer


class AMDataLoader(torch.utils.data.DataLoader):
    """
    Data loader class
    """
    def __init__(self, *args, **kwargs):
        super(AMDataLoader, self).__init__(*args, **kwargs)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __repr__(self):
        s = f'{self.__class__.__name__} (\n'
        s += f'\tdevice: {self.device}\n'
        s += f'\tnumericalizer_id: {id(self.dataset.numericalizer)}\n'
        s += f')'
        return s

    def _to_device(self, batch_sequence):
        device_batch = dict()
        for field, batch in zip(self.dataset.fields, batch_sequence):
            device_batch[field.name] = field.batchfy(device=self.device, batch=batch)

        return device_batch

    def __iter__(self):
        for raw_batch in super(AMDataLoader, self).__iter__():
            batch, indices = raw_batch
            yield {
                'data_indices': indices,
                'origin': [self.dataset.corpus[i] for i in indices],
                'batch': self._to_device(batch),
            }


class AMDataset(torch.utils.data.Dataset):
    """
    Dataset class
    """
    def __init__(self, corpus: Sized, numericalizer: AMNumericalizer, **kwargs):
        super(AMDataset, self).__init__()

        self.corpus = corpus
        self.numericalizer = numericalizer
        self.fields = numericalizer.fields
        self.numerics = self.numericalizer.numericalize(corpus=corpus, **kwargs)
        self.field_names = [f.name for f in self.fields]

        # Check if the dataset size is valid
        for k in self.field_names:
            assert len(self.numerics[k]) == len(self.corpus)
        assert len(set(self.field_names)) == len(self.field_names) == len(self.fields)

    def __getitem__(self, index):
        # Returns numericalized sample and data index
        return [self.numerics[k][index] for k in self.field_names], index

    def __len__(self):
        return len(self.corpus)

    @staticmethod
    def collate_fn(batch):
        # Returns (i) batched fields and (ii) data indices
        return [field for field in zip(*[b[0] for b in batch])], \
               [b[1] for b in batch]  # Indices


def construct_loader(
        corpus: Sized,
        numericalizer: AMNumericalizer,
        batch_size: int,
        shuffle: bool = False,
        **kwargs) -> AMDataLoader:
    """
    Build data loader from the corpus and numericalizer

    Parameters
    ----------
    corpus : Sized
        The corpus instance
    numericalizer : AMNumericalizer
        The numericalizer instance
    batch_size : int
        The bach size
    shuffle : bool
        Whether to shuffle the batch samples

    Returns
    ----------
    data_loader : AMDataLoader
        The data loader instance
    """
    dataset = AMDataset(corpus=corpus, numericalizer=numericalizer, **kwargs)
    return AMDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        sampler=None
    )
