#!/usr/bin/env python

import os

from nnicotine import datasets

partition = "training_30"

infile = os.path.join(data_root_dir, "proteinnet/data/casp12/{}".format(partition))

file_pointer = open(infile, 'r')
next_entry = datasets.preprocess.read_entry(file_pointer)

testids = []

while next_entry is not None:
    ID = next_entry['id']
    if os.path.isfile(os.path.join(data_root_dir, "proteinnet-raw/raw/"+ID+".a2m.gz")):
        testids.append(ID)
    if len(testids) >= 10:
        break

for ID in testids:
    pass

