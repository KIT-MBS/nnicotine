#!/usr/bin/env python

import os

from nnicotine import datasets

data_root_dir = os.environ['DATA_PATH']

ds = datasets.ProteinNet(os.path.join(data_root_dir, 'proteinnet/data/'), mode=30, version=7, download=True)

print(len(ds))

# TODO histogram sequence lengths and MSA effective number of sequences

for i, (x, y) in enumerate(ds):
    print(i, x)
