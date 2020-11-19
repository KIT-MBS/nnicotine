#!/usr/bin/env python

import os
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from nnicotine import datasets
from nnicotine.utils import compute_msa_weights


root = os.path.join(os.environ['DATA_PATH'], "nnicotine")
cath_version = '20200708'
modes = ["toy"]


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# NOTE preserve sample order

for mode in modes:
    ds = datasets.CATHDerived(root, mode=mode, version=cath_version, generate=False)

    nsequences = []
    neffectivesequences = []
    lengths = []
    gappedlengths = []

    # rank_samples = [int(np.ceil(len(ds)/size))] + [int(len(ds)/size) for i in range(size-1)]
    # offset = 0 if rank == 0 else sum(rank_samples[rank-1:rank])
    limits = [i * int(len(ds)/size) for i in range(size)] + [len(ds)]

    for i in range(limits[rank], limits[rank+1]):
        x, y = ds[i]
        lengths.append(len(x['sequence']))
        gappedlengths.append(x['msa'].get_alignment_length())
        nsequences.append(len(x['msa']))
        neffectivesequences.append((compute_msa_weights(x['msa'])).sum())

    lengths = comm.gather(lengths, root=0)
    gappedlengths = comm.gather(gappedlengths, root=0)
    nsequences = comm.gather(nsequences, root=0)
    neffectivesequences = comm.gather(neffectivesequences, root=0)

    if rank==0:
        # NOTE flatten gathered data
        lengths = np.array(list(chain.from_iterable(lengths)))
        gappedlengths = np.array(list(chain.from_iterable(gappedlengths)))
        nsequences = np.array(list(chain.from_iterable(nsequences)))
        neffectivesequences = np.array(list(chain.from_iterable(neffectivesequences)))

        assert len(lengths) == len(ds)
        # NOTE ds stats save to file
        np.save(f'data/{cath_version}_{mode}_lens.npy', lengths)
        np.save(f'data/{cath_version}_{mode}_glens.npy', gappedlengths)
        np.save(f'data/{cath_version}_{mode}_seq.npy', nsequences)
        np.save(f'data/{cath_version}_{mode}_effseq.npy', neffectivesequences)
