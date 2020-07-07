import os
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
                    print(domain_id)
                    id_dset[i] = np.string_(domain_id)
                    if ',' in boundaries:
                        # NOTE if the domain consists of several fragments: pick only the longest one
                        # NOTE this domain definition doess not seem particularly sensible to me, but what do I know...
                        fragment_boundaries = boundaries.split(',')
                        # TODO put this in a list comprehension
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
                    print(strand_ids)
                    seq = parser._mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"][seq_idx]
                    seq = ''.join(seq.split())
                    residues = []
                    for i in range(lower, upper):
                        if i in c:
                            residues.append(protein_letters_3to1[c[i].get_resname().capitalize()])
                        else:
                            residues.append('-')
                    strucseq = [aa for aa in residues]

                    print("domain from structure: ", ''.join(strucseq))
                    print("full chain sequence  : ", seq)
                    if '-' in residues:
                        strucseq = _optimistic_resolve_gaps(strucseq, seq)
                        if ''.join(strucseq) not in seq:
                            strucseq = [aa for aa in residues]
                            strucseq = _length_mismatch_resolve_gaps(strucseq, seq)

                    if ''.join(strucseq) not in seq:
                        degapped_strucseq = ''.join([subseq for subseq in ''.join(strucseq).split('-') if subseq != ''])
                        if degapped_strucseq not in seq:
                            raise RuntimeError("Reconstructed sequence does not match the expected data base sequence:\n{}\n{}".format(''.join(strucseq), seq))
                        else:
                            print("Warning: Missing residues could not be filled in from sequence information. Keeping unresolved gaps.")


                    # if '-' in aas:
                    #     print("gaaaaaaps")
                    #     for subseq in ''.join(aas).split('-'):
                    #         if subseq not in seq:
                    #             raise RuntimeError("Part of the sequence found in the structure does not match the database sequence:\n{}\n{}".format(str(subseq), seq))
                    #     # first = ''.join(aas).split('-')[0]
                    #     # start = seq.find(first)
                    #     # for i, aa in enumerate(aas):
                    #     #     if aa == '-':
                    #     #         aas[i] = seq[start+i]

                    #     # TODO really short resolved subsequences are going to be a problem

                    #     # resolved_boundaries = [(seq.find(subseq), seq.find(subseq)+len(subseq)) for subseq in ''.join(aas).split('-') if subseq != '']

                    #     subseqs = [subseq for subseq in ''.join(aas).split('-') if subseq != '']
                    #     resolved = [seq.find]
                    #     for subseq in subeqs:
                    #         pass

                    #     print(resolved_boundaries)
                    #     for i in range(len(resolved_boundaries)-1):
                    #         unresolved_start = resolved_boundaries[i][1]
                    #         unresolved_end = resolved_boundaries[i+1][0]
                    #         for j in range(unresolved_start, unresolved_end):
                    #             aas[j-resolved_boundaries[0][0]] = seq[j]
                    #             print(j, seq[j])


                    #     # TODO print warning if there is still a gap in there
                    #     if ''.join(aas) not in seq:
                    #         print(l, lower, upper)
                    #         raise RuntimeError("Missing residue letters could not be reconstructed correctly:\n{}\n{}".format(''.join(''.join(aas).split('-')), seq))
                    #     print("reconstructed from st: ", ''.join(aas))

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

def _optimistic_resolve_gaps(strucseq, seq):
    subseqs = [subseq for subseq in ''.join(strucseq).split('-') if subseq != '']
    longest_subseq = max(subseqs, key=len)
    longest_start_seq = seq.find(longest_subseq)
    longest_start_struc = ''.join(strucseq).find(longest_subseq)

    strucseq_copy = [x for x in strucseq]

    for i in range(len(strucseq)):
        if strucseq[i] == '-':
            strucseq[i] = seq[longest_start_seq - longest_start_struc + i]
        else:
            if strucseq[i] != seq[longest_start_seq - longest_start_struc + i]:
                # raise RuntimeError("sequence mismatch:{}\n{}\nat index {}".format(''.join(strucseq), seq, i))
                return strucseq

    return strucseq

# NOTE when the length of the gap in the structure is longer than in the reference sequence
# TODO only works if the mismatch is in the last gap
def _length_mismatch_resolve_gaps(strucseq, seq):
    resolved_boundaries = [(seq.find(subseq), seq.find(subseq)+len(subseq)) for subseq in ''.join(strucseq).split('-') if subseq != '']

    offset = 0
    for i in range(len(resolved_boundaries)-1):
        unresolved_start = resolved_boundaries[i][1]
        unresolved_end = resolved_boundaries[i+1][0]
        assert unresolved_start <= unresolved_end
        for j in range(unresolved_start, unresolved_end):
            if strucseq[j-resolved_boundaries[0][0]] != '-':
                print(resolved_boundaries)
                print(i)
                print('index into ref sequence: ', j)
                print('index into str sequence: ', j-resolved_boundaries[0][0])
                print(unresolved_start, unresolved_end)
                print(''.join(seq[unresolved_start:unresolved_end]))
                print(strucseq[j-resolved_boundaries[0][0]])
                # print(strucseq[j-])
                print(seq[j])
                assert False
            strucseq[j-resolved_boundaries[0][0]] = seq[j]
        offset += strucseq[unresolved_end-resolved_boundaries[0][0]+offset:].count('-') - []
    return strucseq
