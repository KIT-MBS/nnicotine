import os

import numpy as np
import h5py
import torch

# NOTE adapted from github.com/biolib/openprotein

class proteinnet(torch.utils.data.Dataset):
    def __init__(self, root, mode=train, version=13, thinning=90, download=True, transforms=None):

        # TODO check how the hdf5 encoded files are organized
        filename = root+'casp'+str(version)+'/'+str(thinning)
        if os.path.exists(filename):
            self.h5py_file = h5py.File(filename, 'r')
        else:
            # TODO download text files
            # TODO pre process into hdf5
            pass

        self.num_proteins, self.max_sequence_len = self.h5py_file['primary'].shape

        return

    def __get_item__(self, index):
        # TODO add transforms
        mask = torch.Tensor(self.h5py_file['mask'][index, :]).type(dtype=torch.bool)
        prim = torch.masked_select(torch.Tensor(self.h5py_file['primary'][index, :]).type(dtype=torch.long), mask)
        tert = torch.Tensor(self.h5py_file['tertiary'][index][:int(mask.sum())])
        return prim, tert

    def __len__(self):
        return self.num_proteins


# TODO collating, batching, augmentation
