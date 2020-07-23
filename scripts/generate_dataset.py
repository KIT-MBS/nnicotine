#!/usr/bin/env python

import os
from mpi4py import MPI
import subprocess as sp

from nnicotine import datasets

data_root_dir = os.environ['DATA_PATH']
root = os.path.join(data_root_dir, "nnicotine")
pdb_root = os.path.join(data_root_dir, "pdb")
workdir = os.environ["LOCALSCRATCH"]
database = "UniRef30_2020_03"
cath_version = '20200708'

def write_seq_file(id, seq, seqfile):
    os.mkdirs(os.path.join(workdir, '{}'.format(id[1:3])), exist_ok=True)

    with open(seqfile, 'w') as f:
        f.write('>{}\n{}'.format(id, seq))

def hhblits(id, seq):
    seqfile = os.path.join(workdir, '{}/{}.seq'.format(id[1:3], id))
    write_seq_file(id, seq, seqfile)
    a3mfile = os.path.join(workdir, '{}/{}.a3m'.format(id[1:3], id))
    clustalfile = os.path.join(workdir, '{}/{}.clu'.format(id[1:3], id))
    seqfile = os.path.join(workdir, '{}/{}.seq'.format(id[1:3], id))

    # TODO i'm not sure the mact value is the reported E value.
    # TODO trim unalignable parts?
    cp = sp.run(['hhblits', '-cpu', '48', '-i', seqfile, '-d', '{}/{}'.format(workdir, database), '-oa3m', a3mfile, '-mact', '0.001'], check=True)
    cp = sp.run(['reformat.pl', 'a3m', 'clu', a3mfile, clustalfile], check=True)
    return clustalfile


# modes = ['toy', 'train', 'test']
modes = ['toy']

size = MPI.Comm.Get_size()
rank = MPI.Comm.Get_rank()

for mode in modes:
    dataset = datasets.CATHDerived(root, mode=mode, version=cath_version, generate=False, pdb_root=pdb_root)
    ids = dataset.h5pyfile['ids']
    for i in range(rank, len(dataset), size):
        id = ids[i].tostring().decode('utf-8')
        seq = dataset.h5pyfile[id]['sequence'][...].tostring().decode('utf-8')
        alignmentfile = hhblits(id, seq)
        os.mkdirs(os.path.join(root, "{}/{}".format(version, id[1:3])), exist_ok=True)
        sample_outfile = os.path.join(root, "{}/{}/{}.clu".format(version, id[1:3], id))
        sp.run(['rsync', alignmentfile, sample_outfile], check=True)
