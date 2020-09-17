#!/usr/bin/env python

import os

from nnicotine import datasets
from nnicotine import transforms

data_root_dir = os.environ['DATA_PATH']
root = os.path.join(data_root_dir, "nnicotine")
cath_version = '20200708'
mode = "toy"

ds = datasets.CATHDerived(root, mode=mode, version=cath_version, generate=False)

# TODO histogram sequence lengths and MSA effective number of sequences

nsequences = []
alengthts = []

Tsample, Ttarget = transforms.get_transforms()

for i, (x, y) in enumerate(ds):
    # print(i)
    # print(x['sequence'], x['msa'][0].seq)
    nsequences.append(len(x['msa']))
    alengthts.append(x['msa'].get_alignment_length())

    x = Tsample(x)


    break

# TODO plot
