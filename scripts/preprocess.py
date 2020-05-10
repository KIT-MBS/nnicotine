#!/usr/bin/env python

import os

import h5py

from nnicotine import datasets

data_root_dir = os.environ['DATA_PATH']


infile = os.path.join(data_root_dir, "proteinnet/data/casp12/training_30")
outfile = os.path.join(data_root_dir, "proteinnet/data/preprocessed/casp12_30.hdf5")

datasets.preprocess.convert_proteinnet_file(infile, outfile)

# with h5py.File(outfile, 'r') as f:
#     pass
