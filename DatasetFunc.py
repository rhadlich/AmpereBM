#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:04:57 2023

@author: rodrigohadlich
"""

import os
import glob
import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def peek_shapes_hdf5(data_dir):
    files = glob.iglob(os.path.join(data_dir, '*.h5'))
    with h5.File(next(files), "r") as fin:
        data_shape = fin['data'].shape
        label_shape = fin['labels'].shape

    return data_shape, label_shape


# Dataset class
class GetDataset(Dataset):

    def init_reader(self):
        # shuffle
        if self.shuffle:
            self.rng.shuffle(self.all_files)

        # shard dataset
        self.global_size = len(self.all_files)
        if self.allow_uneven_distribution:
            # covers dataset completely, some workers will have more examples than others

            # deal with bulk of files
            num_files_local = self.global_size // self.size
            start_idx = self.rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]

            # deal with remainder of files
            for idx in range(self.size * num_files_local, self.global_size):
                if idx % self.size == self.rank:
                    self.files.append(self.all_files[idx])
        else:
            # here every worker will get the same number of samples,
            # potentially under-sampling the data
            num_files_local = self.global_size // self.size
            start_idx = self.rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]
            self.global_size = self.size * len(self.files)

        # number of files in this worker
        self.local_size = len(self.files)
        print(f'Number of files in rank {self.rank} is {self.local_size}')

    def __init__(self,
                 source,
                 allow_uneven_distribution=False,
                 shuffle=False,
                 size=1,
                 rank=0,
                 seed=12345):
        self.source = source
        self.allow_uneven_distribution = allow_uneven_distribution
        self.shuffle = shuffle
        self.size = size
        self.rank = rank
        self.all_files = sorted([os.path.join(self.source, x) for x in os.listdir(self.source) if
                                 x.endswith('.h5')])  # set file format extension here

        # create seed for shuffling files
        self.rng = np.random.RandomState(seed)

        # init reader
        self.init_reader()

        # get shapes of data and labels
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['data'].shape
            self.label_shape = fin["labels"].shape

        if rank == 0:
            print(f'Initialized dataset with {self.global_size} samples. World size is {size}')

        print(f'Local dataset size in rank {self.rank} is {self.local_size}')

    def __len__(self):
        return self.local_size

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        # load data and project
        with h5.File(filename, "r") as f:
            data = f['data'][...][:, 1:-1]  # order is phi, inj pressure, inj timing, inj duration, cov
            label = f['labels'][...]
            mean = f['mean'][...][1:-1]
            std_dev = f['std'][...][1:-1]

        # pre-process
        data = (data - mean) / std_dev

        data = np.squeeze(data)

        return data, label, filename, idx
