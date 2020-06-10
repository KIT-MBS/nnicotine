from collections import OrderedDict
import gzip
import re

import torch
import hypy
from torchvision.datasets.utils import download_url
from .utils import cath_train_test_split

class CATHDerived(torch.utils.data.Dataset):
    """
    root: root directory of dataset
    mode: toy, train or val. there is no test set. A good test set would be a set of CASP targets released after the CATH version release
    version: either daily, a date in yyyymmdd format or valid CATH plus version (latest at time of writing is v4.2.0) or latest (the most recent CATH plus release)
    generate: whether to generate a dataset (does not automatically generate MSAs, only a list of sequences to generate MSAs from). Default is False.
    transform: transformation to apply to sample input
    target_transform: transformation to apply to sample target


    """

    urlbase = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/"


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
            # TODO the cath plus release files are formatted very differently
            raise NotImplementedError()
            # TODO
            self.url = urlbase + "all-releases/{}/".format()

        elif version[0] == 'v':
            raise NotImplementedError()
            self.url = urlbase + "all-releases/{}/".format(version.replace('.', '_')) # TODO put in the actual file
        else:
            y = version[0:4]
            m = version[4:6]
            d = version[5:]
            self.url = urlbase + "all-releases/archive/cath-b-{}{}{}-s35-all.gz"
        self.version = version
        self.versionroot = os.path.join(self.root, self.version)
        self.cathfile = os.join(self.versionroot, "cath-b-s35.gz")
        if generate:
            self.download()
            self.preprocess(**kwargs)
        self.h5pyfilename = os.path.join(root, "cath/{}/{}.hdf5".format(version, mode))
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
        download_url(self.url, self.versionroot, filename=cathfile)

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
                boundaries, chain_id = boundaries.split(':')[1]
                lower, upper = re.findall("-*\d+", boundaries)
                lower = int(lower)
                upper = int(upper[1:]) + 1
                l = upper - lower
                # TODO get sequence
                raise
                # TODO write sequence to fasta?
                # TODO write label to hdf5 file

        return
