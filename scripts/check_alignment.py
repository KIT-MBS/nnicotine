#!/usr/bin/env python

import os
import gzip

import h5py
from Bio import AlignIO
from Bio import SeqIO

from nnicotine import datasets

test_ids = [
    "2J58_1_A",
    "2QXX_1_A",
    "3EAF_1_A",
    "3RZG_1_A",
    "3VC5_d3vc5a1",
    "4W8Q_1_A",
    "4XHT_1_A",
]


data_root_dir = os.environ['DATA_PATH']

proteinnet_ds = h5py.File(os.path.join(data_root_dir, "proteinnet/data/preprocessed/casp12_30.hdf5"), 'r')

for ID in test_ids:
    print(ID)
    print(proteinnet_ds[ID]['primary'][()].tostring().decode('utf-8'))
    alignment_file = os.path.join(data_root_dir, "proteinnet-raw/raw/{}.a2m.gz".format(ID))
    with gzip.open(alignment_file, 'rt') as f:
        l = None
        alignment = SeqIO.parse(f, 'fasta')
        for record in alignment:
            if l is None:
                print(record.id)
                l = len(record)
                print(l)
            else:
                if len(record) != l:
                    print(record.id)
                    print(len(record))
                    break
