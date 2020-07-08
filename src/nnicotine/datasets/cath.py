import os
import sys
from collections import OrderedDict
import gzip
import re

import numpy as np
import h5py
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from .utils import cath_train_test_split, get_pdb_filename, protein_letters_3to1


urlbase = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/"


class CATHDerived(torch.utils.data.Dataset):
    """
    root: root directory of dataset
    mode: toy, train or val. there is no test set. A good test set would be a set of CASP targets released after the CATH version release
    version: either daily, a date in yyyymmdd format or valid CATH plus version (latest at time of writing is v4.2.0) or latest (the most recent CATH plus release)
    generate: whether to generate a dataset (does not automatically generate MSAs, only a list of sequences to generate MSAs from). Default is False.
    transform: transformation to apply to sample input
    target_transform: transformation to apply to sample target


    """

    def __init__(self, root, mode='train', version='daily', generate=False, transform=None, target_transform=None, **kwargs):
        if mode != 'toy': raise NotImplementedError("That stuff comes later")
        self.root = root
        self.mode = mode
        # NOTE there is no validation set for now, test set is for validation, casp set is for 'testing'
        self.modes = ["toy", "train", "test"]
        assert mode in self.modes
        if version == 'daily' or version == 'newest':
            self.url = urlbase + "daily-release/newest/cath-b-s35-newest.gz"
            version = 'newest'
        elif version == 'latest':
            raise NotImplementedError("CATH plus versions are not supported")
            self.url = urlbase + "all-releases/{}/".format()

        elif version[0] == 'v':
            raise NotImplementedError("CATH plus versions are not supported")
            self.url = urlbase + "all-releases/{}/".format(version.replace('.', '_')) # TODO put in the actual file
        else:
            y = version[0:4]
            m = version[4:6]
            d = version[6:]
            self.url = urlbase + "daily-release/archive/cath-b-{}{}{}-s35-all.gz".format(y, m, d)
        self.version = version
        self.versionroot = os.path.join(self.root, self.version)
        os.makedirs(self.versionroot, exist_ok=True)
        self.cathfile = os.path.join(self.versionroot, "cath-b-s35.gz")
        self.h5pyfilename = os.path.join(root, "{}/{}.hdf5".format(version, mode))
        if generate:
            self.download()
            self.preprocess(**kwargs)
        self.h5pyfile = h5py.File(self.h5pyfilename, 'r')
        self.num_samples = len(self.h5pyfile['ids'])

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        domain_id = self.h5pyfile['ids'][index].tostring().decode('utf-8')

        # TODO
        # msa = None
        # gapped_sequence = None

        # sample = OrderedDict(('sequence', gapped_sequence), ('msa', msa))
        # target = OrderedDict(('cb', gapped_cb), ('ca', gapped_ca))
        sample = self.h5pyfile[domain_id]['sequence'][...].tostring().decode('utf-8')
        target = self.h5pyfile[domain_id]['ca'][...]

        return sample, target

    def __len__(self):
        return self.num_samples

    def download(self):
        if os.path.exists(os.path.join(self.versionroot, self.cathfile)):
            return
        print(self.url)
        download_url(self.url, self.versionroot, filename=self.cathfile)

    def preprocess(self, **kwargs):
        if os.path.exists(self.h5pyfilename):
            # TODO
            # print("using existing file")
            # return
            pass
        pdb_root = kwargs.get('pdb_root', os.path.join(self.root, '../pdb'))
        if not os.path.exists(pdb_root):
             raise RuntimeError("A PDB containing structural data on CATH domains is required.")
        lines = []
        with gzip.open(self.cathfile, 'rt') as f:
            for line in f:
                lines.append(line)
        domains = [line.split() for line in lines]
        train_domains, test_domains = cath_train_test_split(domains, **kwargs)

        toy_domains = train_domains[:10]

        for mode, domains in [('toy', toy_domains), ('train', train_domains), ('test', test_domains)]:
            h5pyfilename = os.path.join(self.root, "{}/{}.hdf5".format(self.version, mode))
            with h5py.File(h5pyfilename, 'w') as handle:
                id_dset = handle.create_dataset('ids', (len(domains),), dtype='S7')

                for i, domain in enumerate(tqdm(domains)):
                    domain_id, version, cath_code, boundaries = domain
                    pdb_id = domain_id[:4]
                    id_dset[i] = np.string_(domain_id)
                    if ',' in boundaries:
                        # NOTE if the domain consists of several fragments: pick only the longest one
                        fragment_boundaries = boundaries.split(',')
                        fragment_sizes = []
                        for b in fragment_boundaries:
                            b = b.split(':')[0]
                            lower, upper = re.findall("-*\d+", b)
                            fragment_sizes.append(int(upper)-int(lower))
                        boundaries = fragment_boundaries[np.argmax(fragment_sizes)]
                    boundaries, chain_id = boundaries.split(':')
                    lower, upper = re.findall("-*\d+", boundaries)

                    lower = int(lower)
                    upper = int(upper[1:]) + 1
                    l = upper - lower

                    pdb_file = get_pdb_filename(pdb_id, pdb_root)

                    parser = MMCIFParser()
                    with gzip.open(pdb_file, 'rt') as f:
                        s = parser.get_structure(pdb_id, f)
                    c = s[0][chain_id]
                    # NOTE some cifs are formatted weirdly
                    strand_ids = parser._mmcif_dict["_entity_poly.pdbx_strand_id"]
                    seq_idx = [idx for idx, s in enumerate(strand_ids) if chain_id in s][0]
                    seq = parser._mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"][seq_idx]
                    seq = ''.join(seq.split())
                    residues = []
                    for i in range(lower, upper):
                        if i in c:
                            residues.append(protein_letters_3to1[c[i].get_resname().capitalize()])
                        else:
                            residues.append('-')
                    strucseq = [aa for aa in residues]

                    if '-' in residues:
                        strucseq = _resolve_gaps(strucseq, seq)

                    if ''.join(strucseq) not in seq:
                        degapped_strucseq = ''.join([subseq for subseq in ''.join(strucseq).split('-') if subseq != ''])
                        if degapped_strucseq not in seq:
                            raise RuntimeError("Reconstructed sequence does not match the expected data base sequence:\n{}\nsequence from atom entries: {}\nsequence without unresolved gaps:{}\nreference sequence: {}".format(domain_id, ''.join(strucseq), degapped_strucseq, seq))
                        else:
                            print("Warning: Missing residues could not be filled in from sequence information. Keeping unresolved gaps.", file=sys.stderr)

                    ca_coords = np.full((l, 3), float('nan'), dtype=np.float32)
                    cb_coords = np.full((l, 3), float('nan'), dtype=np.float32)

                    for i in range(lower, upper):
                        if i in c:
                            r = c[i]
                            if 'CA' in r:
                                ca_coords[i-lower, :] = r['CA'].get_coord()[:]
                            if 'CB' in r:
                                cb_coords[i-lower, :] = r['CB'].get_coord()[:]
                            if r.get_resname() == 'GLY':
                                cb_coords[i-lower, :] = ca_coords[i-lower, :]


                    sample_group = handle.create_group(domain_id)
                    # TODO write sequence to file to generate MSA
                    sequence_ds = sample_group.create_dataset('sequence', data=np.string_(strucseq))
                    ca_ds = sample_group.create_dataset('ca', data=ca_coords)
                    cb_ds = sample_group.create_dataset('cb', data=cb_coords)

        return


def _resolve_gaps(strucseq, seq):
    subseqs = re.findall(r'\w+|-+', ''.join(strucseq))
    resolved_subseqs = [s for s in subseqs if s[0] != '-']
    # NOTE use longest resolved subsequence as anchor into reference sequence
    longest_resolved = max(resolved_subseqs, key=len)
    longest_subseqs_index = subseqs.index(longest_resolved)
    longest_start_seq = seq.find(longest_resolved)
    longest_start_struc = ''.join(strucseq).find(longest_resolved)

    offset = longest_start_seq - longest_start_struc

    subseqs_starts = [offset + sum([len(s) for s in subseqs[:i]]) for i in range(len(subseqs))]

    # NOTE fix false indices from too long gaps before anchor
    for i in range(longest_subseqs_index-2, -1, -2):
        s = subseqs[i]
        while s != seq[subseqs_starts[i]:subseqs_starts[i]+len(s)] or subseqs_starts[i]<0:
            for j in range(i+1, -1, -1):
                subseqs_starts[j] += 1

    # NOTE fix false indices from too long gaps after anchor
    for i in range(longest_subseqs_index+2, len(subseqs), 2):
        s = subseqs[i]
        while s != seq[subseqs_starts[i]:subseqs_starts[i]+len(s)]:
            for j in range(i, len(subseqs)):
                subseqs_starts[j] -= 1

    # NOTE resolve gaps
    for i in range((longest_subseqs_index+1)%2, len(subseqs), 2):
        s = [x for x in subseqs[i]]
        for j in range(subseqs_starts[i+1]-subseqs_starts[i]):
            s[j] = seq[subseqs_starts[i]+j]
        subseqs[i] = ''.join(s)


    return ''.join(subseqs)
