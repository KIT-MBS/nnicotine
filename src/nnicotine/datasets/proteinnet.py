import os
from collections import OrderedDict

import numpy as np
import h5py
import torch

from torchvision.datasets.utils import download_and_extract_archive
from .preprocess import convert_proteinnet_file


# TODO compute md5 sums
proteinnet_archives = {
        'casp7' : None,
        'casp8' : None,
        'casp9' : None,
        'casp10' : None,
        'casp11' : None,
        'casp12' : None,
        'casp13' : None,
    }

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


class ProteinNet(torch.utils.data.Dataset):
    """
    root: root directory of dataset
    mode: either train thinning in [30, 50, 70, 90, 95, 100] or val or test
    version: protinnet version between 7 and 13
    transform: function that takes in a integer encoded amino acid string an returns a sample tensor
    target_transform: function that takes a 3x3 x sequence length numpy array of backbone coordinates and returns a target tensor
    cbeta: whether to include cbeta coordinates, needs pdb to extract information
    pdb_root: location of pdb, ignore if cbeta is False
    """
    def __init__(self, root, mode=90, version=13, download=False, preprocess=False, transform=None, target_transform=None):
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

        self.num_samples = len(self.h5pyfile['ids'])

    def __getitem__(self, index):
        # TODO should the hdf5file not do this by itself?
        if index >= len(self):
            raise IndexError()
        ID = self.h5pyfile['ids'][index].tostring().decode('utf-8')

        prim = self.h5pyfile[ID]['primary']
        evo = self.h5pyfile[ID]['evolutionary']

        tert_n = self.h5pyfile[ID]['n']
        tert_ca = self.h5pyfile[ID]['ca']
        tert_c= self.h5pyfile[ID]['c']
        mask = self.h5pyfile[ID]['mask']

        # NOTE in the published text versions of proteinnet, the secondary structure is still missing
        # sec = self.h5pyfile[ID]['secondary']

        sample = OrderedDict([('primary', prim), ('evolutionary', evo)])
        target = OrderedDict([('n', tert_n), ('ca', tert_ca), ('c', tert_c), ('mask', mask)])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.num_samples

    def download(self):
        if os.path.exists(os.path.join(self.root, 'casp{}/{}'.format(self.version, modes[self.mode]))):
            return

        download_and_extract_archive(self.url, self.root, filename=self.archive_filename, md5=self.tgz_md5)

    def preprocess(self):
        if os.path.exists(os.path.join(self.root, self.filename)):
            return
        convert_proteinnet_file(os.path.join(self.root, self.text_filename), os.path.join(self.root, self.filename))

    def __del__(self):
        pass

        # TODO manually closing the file raises an ImportError
        # self.h5pyfile.close()
