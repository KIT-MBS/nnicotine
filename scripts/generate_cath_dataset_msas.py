#!/usr/bin/env python

import os
from mpi4py import MPI
import subprocess as sp

from nnicotine import datasets

data_root_dir = os.environ['DATA_PATH']
root = os.path.join(data_root_dir, "nnicotine")
pdb_root = os.path.join(data_root_dir, "pdb")
workdir = os.environ["LOCALSCRATCH"]
databasename = "UniRef30_2020_03"
# database = "uniprot_sprot"
cath_version = '20200708'

cores = 4 # TODO set to number of cpu cores used by alginment tool

# TODO use biopython for this
def write_seq_file(id, seq, seqfile):
    os.makedirs(os.path.join(workdir, '{}'.format(id[1:3])), exist_ok=True)

    with open(seqfile, 'w') as f:
        f.write('>{}\n{}'.format(id, seq))


def hhblits(id, seq, seqfile):
    # seqfile = os.path.join(workdir, '{}/{}.seq'.format(id[1:3], id))
    # write_seq_file(id, seq, seqfile)
    a3mfile = os.path.join(workdir, '{}/{}.a3m'.format(id[1:3], id))
    outfile = os.path.join(workdir, '{}/{}.sto'.format(id[1:3], id))
    seqfile = os.path.join(workdir, '{}/{}.seq'.format(id[1:3], id))

    database = os.path.join(workdir, 'uniref/'+databasename)

    # TODO trim unalignable parts?
    # TODO E-value?
    cp = sp.run(['hhblits', '-cpu', '{}'.format(cores), '-i', seqfile, '-d', database, '-oa3m', a3mfile], check=True)
    cp = sp.run(['reformat.pl', 'a3m', 'sto', a3mfile, outfile], check=True)
    return outfile


def jackhmmer(id, seq, seqfile):
    # seqfile = os.path.join(workdir, '{}/{}.fasta'.format(id[1:3], id))
    # write_seq_file(id, seq, seqfile)
    outfile = os.path.join(workdir, '{}/{}.sto'.format(id[1:3], id))
    logfile = os.path.join(workdir, '{}/{}.log'.format(id[1:3], id))
    database = os.path.join(workdir, '{}.fasta'.format(databasename))
    seqdb = os.path.join('/home/oskar/data/uniprot', '{}.fasta'.format(database))
    cp = sp.run(['jackhmmer', '-N', '5', '-E', '0.001', '-Z', str(1e8), '--cpu', '{}'.format(cores), '-A', outfile, '-o', logfile, seqfile, seqdb], check=True)
    return outfile



# modes = ['toy', 'train', 'test']
modes = ['toy']

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

# TODO copy database to workdir

for mode in modes:
    print(mode)
    dataset = datasets.CATHDerived(root, mode=mode, version=cath_version, generate=False, pdb_root=pdb_root)
    ids = dataset.h5pyfile['ids']
    for i in range(rank, len(dataset), size):
        id = ids[i].tostring().decode('utf-8')
        print(id)
        seq = dataset.h5pyfile[id]['sequence'][...].tostring().decode('utf-8')
        print(seq)

        # NOTE write input sequence file
        seqfile = os.path.join(workdir, '{}/{}.seq'.format(id[1:3], id))
        write_seq_file(id, seq, seqfile)

        # NOTE run alignment
        # alignmentfile = jackhmmer(id, seq, seqfile)
        alignmentfile = hhblits(id, seq, seqfile)

        os.makedirs(os.path.join(root, "{}/{}/{}".format(cath_version, 'stockholm', id[1:3])), exist_ok=True)
        outfilename = os.path.basename(alignmentfile)
        sample_outfile = os.path.join(root, "{}/{}/{}/{}".format(cath_version, 'stockholm', id[1:3], outfilename))
        sp.run(['rsync', alignmentfile, sample_outfile], check=True)
