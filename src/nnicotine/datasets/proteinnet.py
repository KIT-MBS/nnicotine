import os

import numpy as np
import h5py
import torch

from torchvision.datasets.utils import download_and_extract_archive
# from .preprocess import preprocess_proteinnet


# TODO insert md5 sums
proteinnet_archives = {'casp7' : None, 'casp13' : None}

modes = {
        30 : 'training_30',
        50 : 'training_30',
        70 : 'training_30',
        90 : 'training_30',
        95 : 'training_30',
        100: 'training_100',
        'val' : 'validation',
        'test': 'testing',
    }


# TODO map astral domains to pdb entries and chain sections
class ProteinNet(torch.utils.data.Dataset):
    """
    root: root directory of dataset
    mode: either train thinning in [30, 50, 70, 90, 95, 100] or val or test
    version: protinnet version between 7 and 13
    transform: function that takes in a integer encoded amino acid string an returns a sample tensor
    target_transform: function that takes a 3x3 x sequence length numpy array of backbone coordinates and returns a target tensor
    raw: whether to include raw MSAs, only available for proteinnet12
    cbeta: whether to include cbeta coordinates, needs pdb to extract information
    raw_root: location of raw MSA database, ignored if raw is False
    pdb_root: location of pdb, ignore if cbeta is False
    """
    def __init__(self, root, mode=90, version=13, download=False, preprocess=False, transform=None, target_transform=None, raw=False, cbeta=False, raw_root=None, pdb_root=None):
        if mode not in modes:
            raise ValueError("Unsupported mode.")
        self.root = root
        self.version = version
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.url = "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp{}.tar.gz".format(version)

        os.makedirs(os.path.join(root, 'preprocessed'), exist_ok=True)
        self.filename = os.path.join(root, 'preprocessed/casp{}_{}.hdf5'.format(version, mode))
        self.text_filename = os.path.join(root, 'casp{}/{}'.format(version, modes[mode]))
        self.archive_filename = os.path.join(root, 'casp{}.tar.gz'.format(version))

        self.tgz_md5 = proteinnet_archives['casp{}'.format(version)]

        if download:
            self.download()

        if preprocess:
            self.preprocess()

        self.h5pyfile = h5py.File(self.filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape

        return

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        mask = torch.Tensor(self.h5pyfile['mask'][index, :]).type(dtype=torch.bool)
        prim = torch.masked_select(torch.Tensor(self.h5pyfile['primary'][index, :]).type(dtype=torch.long), mask)
        tert = torch.Tensor(self.h5pyfile['tertiary'][index][:int(mask.sum())])
        if self.transform is not None:
            prim = self.transform(prim)
        if self.target_transform is not None:
            tert = self.target_transform(tert)

        return prim, tert

    def __len__(self):
        return self.num_proteins

    def download(self):
        if os.path.exists(os.path.join(self.root, 'casp{}/{}'.format(self.version, modes[self.mode]))):
            return

        download_and_extract_archive(self.url, self.root, filename=self.archive_filename, md5=self.tgz_md5)
        return

    # TODO only get C_alphas instead of all backbone coordinates
    def preprocess(self):
        if os.path.exists(os.path.join(self.root, self.filename)):
            return
        raise
        # preprocess_proteinnet(self.root, filename=self.raw_filename, out_filename=self.filename)
        return
