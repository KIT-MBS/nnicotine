#!/usr/bin/env python

import os

from nnicotine import datasets

data_root_dir = os.environ['DATA_PATH']
root = os.path.join(data_root_dir, "nnicotine")
pdb_root = os.path.join(data_root_dir, "pdb")

dataset = datasets.CATHDerived(root, mode='toy', generate=True, pdb_root=pdb_root)
