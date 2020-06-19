import os
from collections import OrderedDict
import gzip
import re

import numpy as np
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
        if os.path.exists(self.versionroot):
            return
        download_url(self.url, self.versionroot, filename=self.cathfile)

    def preprocess(self, **kwargs):
        if os.path.exists(self.h5pyfilename):
            print("using existing file")
            return
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

        for mode, domains in [('toy', toy_domains)]: # TODO other modes
            h5pyfilename = os.path.join(self.root, "{}/{}.hdf5".format(self.version, mode))
            with h5py.File(h5pyfilename, 'w') as handle:
                # NOTE write IDs to file
                id_dset = handle.create_dataset('ids', (len(domains),), dtype='S7')

                for i, domain in enumerate(domains): # TODO tqdm
                    domain_id, version, cath_code, boundaries = domain
                    pdb_id = domain_id[:4]
                    print(domain_id)
                    print(pdb_id)
                    id_dset[i] = np.string_(domain_id)
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
                    print(strand_ids)
                    seq = parser._mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"][seq_idx]
                    aas = []
                    for i in range(lower, upper):
                        if i in c:
                            aas.append(protein_letters_3to1[c[i].get_resname().capitalize()])
                        else:
                            aas.append('-')
                    print("domain from structure: ", ''.join(aas))
                    print("full chain sequence  : ", seq)
                    if '-' in aas:
                        for subseq in ''.join(aas).split('-'):
                            if subseq not in seq: raise RuntimeError("Part of the sequence found in the structure does not match the database sequence:\n {} \n{}".format(str(subseq), seq))
                        first = ''.join(aas).split('-')[0]
                        start = seq.find(first)
                        for i, aa in enumerate(aas):
                            if aa == '-':
                                aas[i] = seq[start+i]
                        if ''.join(aas) not in seq: raise RuntimeError("Missing residue letters could not be reconstructed correctly:\n {} \n {}".format(''.join(aas), seq))
                        print("reconstructed from st: ", ''.join(aas))


                    ca_coords = np.full((l, 3), float('inf'), dtype=np.float32)
                    cb_coords = np.full((l, 3), float('inf'), dtype=np.float32)

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
                    sequence_ds = sample_group.create_dataset('sequence', data=np.string_(''.join(aas)))
                    ca_ds = sample_group.create_dataset('ca', data=ca_coords)
                    cb_ds = sample_group.create_dataset('cb', data=cb_coords)

        return
