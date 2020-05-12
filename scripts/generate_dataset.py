#!/usr/bin/env python

import random
import os
import gzip

random.seed(42)


data_root_dir = os.environ['DATA_PATH']

# TODO make split, same pdb codes and same superfamily codes should be in same partition
# TODO generate domain samples
# TODO generate chain samples
# TODO get sequence
# TODO generate MSA
# TODO generate input features
# TODO generate target features

n = 0
superfamily_codes = []
superfamilies = {}
IDs = []
with gzip.open(os.path.join(data_root_dir, "nnicotine/casp14/cath-b-s35-newest.gz"), 'rt') as f:
    for line in f:
        n += 1
        domainid, version, cath_code, boundaries = line.split()
        superfamily = cath_code.split('.')[3]
        if superfamily not in superfamilies:
            superfamilies[superfamily] = 1
        else:
            superfamilies[superfamily] += 1

        IDs.append(domainid)
        superfamily_codes.append(superfamily)


print(n)
print(len(superfamilies))
print(max(superfamilies, key=superfamilies.get), max(superfamilies.values()))
possible_validation_families = [(x, superfamilies[x]) for x in superfamilies if superfamilies[x] < 200]

validation_families = set()
validation_samples = 0
random.shuffle(possible_validation_families)
for f, s in possible_validation_families:
    validation_families.add(f)
    validation_samples += s
    if validation_samples > 1800: break

print(validation_families)

with gzip.open(os.path.join(data_root_dir, "nnicotine/casp14/train.txt.gz"), 'wt') as tf:
    with gzip.open(os.path.join(data_root_dir, "nnicotine/casp14/validation.txt.gz"), 'wt') as vf:
        for ID, family in zip(IDs, superfamily_codes):
            if family in validation_families:
                vf.write(ID+"\n")
            else:
                tf.write(ID+"\n")

