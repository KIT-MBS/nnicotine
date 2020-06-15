import os
from collections import OrderedDict
import gzip
import re

import h5py
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from torchvision.datasets.utils import download_url

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
            d = version[5:]
            self.url = urlbase + "all-releases/archive/cath-b-{}{}{}-s35-all.gz"
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
        ID = self.h5pyfile['ids'][index].tostring().decode('utf-8')

        # TODO
        msa = None
        gapped_prim = None

        sample = OrderedDict(('primary', gapped_prim), ('msa', msa))
        target = OrderedDict(('cb', gapped_cb), ('ca', gapped_ca))

        return sample, target

    def __len__(self):
        return self.num_samples

    def download(self):
        if os.path.exists(self.versionroot):
            return
        download_url(self.url, self.versionroot, filename=self.cathfile)

    def preprocess(self, **kwargs):
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

        with h5py.File(self.h5pyfilename, 'w') as f:
            for domain in toy_domains:
                domain_id, version, cath_code, boundaries = domain
                pdb_id = domain_id[:4]
                print(pdb_id)
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
                chain_number = parser._mmcif_dict["_entity_poly.pdbx_strand_id"].index(chain_id)
                seq = parser._mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"][chain_number]
                # TODO write sequence to fasta?
                aas = []
                for i in range(lower, upper):
                    if i in c:
                        aas.append(protein_letters_3to1[c[i].resname.capitalize()])
                    else:
                        aas.append('-')
                print(''.join(aas))
                print(seq)
                for subseq in ''.join(aas).split('-'):
                    if subseq not in seq: raise RuntimeError("Part of the sequence found in the structure does not match the database sequence:\n {} \n{}".format(str(subseq), seq))

                # TODO write label to hdf5 file

        return
