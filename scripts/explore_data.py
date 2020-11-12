#!/usr/bin/env python

import os

from nnicotine import datasets
from nnicotine import transforms


root = os.path.join(os.environ['DATA_PATH'], "nnicotine")
cath_version = '20200708'
mode = "toy"

ds = datasets.CATHDerived(root, mode=mode, version=cath_version, generate=False)

nsequences = []
neffectivesequences = []
lengths = []
gappedlengths = []

Tsample, Ttarget = transforms.get_transforms()

for i, (x, y) in enumerate(ds):
    lenghts.append(len(x['seq']))
    gappedlengths.append(x['msa'].get_alignment_length())
    nsequences.append(len(x['msa'][0]))
    # TODO compute effective sequences

# TODO histogram, plot and save
