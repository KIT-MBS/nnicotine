#!/usr/bin/env python

import os
import pickle
import fnmatch

from nnicotine import datasets

data_root_dir = os.environ['DATA_PATH']

a2mfiles = set([x for x in os.listdir(os.path.join(data_root_dir, "proteinnet-raw/raw/")) if x.endswith('.a2m.gz')])
alignments = {x[:-7] for x in a2mfiles}

allpartitions = ["training_30", "training_50", "training_70", "training_90", "training_95", "training_100", "validation", "testing"]
trainingpartitions = allpartitions[:-2]

idlists = {}

for partition in allpartitions:
    print(partition)
    if os.path.isfile(os.path.join(data_root_dir, "proteinnet/data/casp12/{}_ids.pkl".format(partition))):
        with open(os.path.join(data_root_dir, "proteinnet/data/casp12/{}_ids.pkl".format(partition)), 'rb') as f:
            ids = pickle.load(f)
    else:
        infile = os.path.join(data_root_dir, "proteinnet/data/casp12/{}".format(partition))
    
        file_pointer = open(infile, 'r')
        next_entry = datasets.preprocess.read_entry(file_pointer)
        ids = []

        while next_entry is not None:
            ID = next_entry['id']
            if partition in ["validation", "testing"]:
                ID = ID.split('#')[1]
            ids.append(ID)
            next_entry = datasets.preprocess.read_entry(file_pointer)
        with open(os.path.join(data_root_dir, "proteinnet/data/casp12/{}_ids.pkl".format(partition)), 'wb') as f:
            pickle.dump(ids, f)
    idlists[partition] = ids

    print("entries in proteinnet textfile: ", len(ids))
    print("of those, alignments found raw: ", len(alignments & set(ids)))

print("\ntotal number of alignments present: ", len(a2mfiles))

missingalignments = set()
# for partition in allpartitions:
for partition in ["validation"]:
    missingalignments = missingalignments | (set(idlists[partition]) - alignments)


for partition in allpartitions:
    alignments = alignments - set(idlists[partition])
print("alignments not referenced in any partition: ", len(alignments))
print("missing alignments: ", len(missingalignments))

for i in range(len(trainingpartitions)-1):
    print(trainingpartitions[i], " in ", trainingpartitions[i+1], set(idlists[trainingpartitions[i]]) <= set(idlists[trainingpartitions[i+1]]))
    print(len(set(idlists[trainingpartitions[i]]) - set(idlists[trainingpartitions[i+1]])))

print(sorted(list(alignments))[:100])
print(sorted(list(missingalignments))[:100])
