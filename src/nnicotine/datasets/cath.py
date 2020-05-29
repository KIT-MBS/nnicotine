from collections import OrderedDict

import torch
import hypy

class CATHDerived(torch.utils.data.Dataset):
    """
    root: root directory of dataset
    mode: toy, train or val. there is no test set. A good test set would be a set of CASP targets released after the CATH version release
    version: either daily, a date in yyyymmdd format or valid CATH plus version (latest at time of writing is v4.2.0)
    generate: whether to generate a dataset (does not automatically generate MSAs, only a list of sequences to generate MSAs from). Default is False.
    transform: transformation to apply to sample input
    target_transform: transformation to apply to sample target


    """

    urlbase = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/"


    def __init__(self, root, mode='train', version='daily', generate=False, transform=None, target_transform=None, **kwargs):
        if mode != 'toy': raise NotImplementedError("That stuff comes later")
        self.root = root
        self.mode = mode
        if version == 'daily' or version == 'newest':
            # TODO figure out the date
            self.url = urlbase + "daily-release/newest/cath-b-s35-newest.gz"
            version = 'newest'
        elif version[0] == 'v':
            self.url = urlbase + "all-releases/{}/".format(version.replace('.', '_')) # TODO put in the actual file
        else:
            y = int(version[0:4])
            m = int(version[4:6])
            d = int(version[5:])
            self.url = urlbase + ""
        self.version = version
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
        return

    def preprocess(self):
        return
