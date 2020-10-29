import os
import sys
from collections import OrderedDict
import gzip
import re

import numpy as np
import h5py
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio import AlignIO
from Bio import pairwise2
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from .utils import cath_train_test_split, get_pdb_filename, protein_letters_3to1

# NOTE silence warnings for the biopython parser
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)


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

        self.transform = transform
        self.target_transform = target_transform

        if generate:
            self.download()
            self.preprocess(**kwargs)
        self.h5pyfile = h5py.File(self.h5pyfilename, 'r')
        self.num_samples = len(self.h5pyfile['ids'])

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        domain_id = self.h5pyfile['ids'][index].tostring().decode('utf-8')

        msa_file = os.path.join(self.root, '{}/clustal/{}/{}.clu'.format(self.version, domain_id[1:3], domain_id))
        sequence = self.h5pyfile[domain_id]['sequence'][...].tostring().decode('utf-8')
        msa = AlignIO.read(open(msa_file), 'clustal')
        assert sequence == str(msa[0].seq).replace('-', '')

        sample = OrderedDict([('sequence', sequence), ('msa', msa)])
        target = OrderedDict([('ca', self.h5pyfile[domain_id]['ca'][...]), ('cb', self.h5pyfile[domain_id]['cb'][...])])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.num_samples

    def download(self):
        if os.path.exists(os.path.join(self.versionroot, self.cathfile)):
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

        for mode, domains in [('toy', toy_domains), ('train', train_domains), ('test', test_domains)]:
            h5pyfilename = os.path.join(self.root, "{}/{}.hdf5".format(self.version, mode))
            with h5py.File(h5pyfilename, 'w') as handle:
                id_dset = handle.create_dataset('ids', (len(domains),), dtype='S7')

                for i, domain in enumerate(tqdm(domains)): # NOTE decided to ignore obsolete files because they are even more of a pain than the normal ones
                # for i, domain in enumerate(["4hubI02 v4_2_0 1.10.10.250.4 71-156:I".split()]):
                # for i, domain in enumerate(["2f07B00 v4_2_0 1.10.357.10.117 303-193:B".split()]):
                # for i, domain in enumerate(["3frhA01 v4_2_0 1.10.8.10.2 -1-56:A".split(), "3bbzA00 v4_2_0 1.10.8.10.18 344-391:A".split(), "2zs0D00 v4_2_0 1.10.490.10.13 1-145:D".split()]):
                # for i, domain in enumerate(["3mlgA00 v4_2_0 1.20.120.1980.1 13-181:A".split()]):
                # for i, domain in enumerate(["6guxA00 putative 1.20.1070.10.1 8-247:A".split()]): # NOTE this can't even pe parsed because of duplicate atom names. i just modified the file. i'm so tired
                # for i, domain in enumerate(["1vs9S01 v4_2_0 2.30.30.30.32 1-72:S".split()]): # NOTE superfluously added insertion code to residue id. in some positions. in others they just put letters randomly at the end of the sid? wtf
                # for i, domain in enumerate(["4ck4A00 v4_2_0 2.40.128.20.7 3-162:A".split()]): # NOTE these guys didn't bother to put the mutated sequence anywhere, TODO i put it in as a quick fix. it would be better to have something scalable
                # for i, domain in enumerate(["6sxtA02 putative 2.80.10.50.32 338-501:A".split()]):
                # for i, domain in enumerate(["1br7003 v4_2_0 3.10.110.10.34 430-546:0".split()]): # NOTE this ones seems to be completely bonkers. it's been superseded 20 years ago... why is it still in a cath database?
                # for i, domain in enumerate(["4k5yA01 v4_2_0 1.20.1070.10.22 115-1002:A,224-368:A".split()]): # NOTE this one has a tag at the end with a weird numbering that is still part of the domain for some reason. i'll tread the upper limit as exclusive, even though i don't think it is (have not seen documentation on this)

                    domain_id, version, cath_code, boundaries = domain
                    pdb_id = domain_id[:4]
                    id_dset[i] = np.string_(domain_id)
                    if ',' in boundaries:
                        # NOTE if the domain consists of several fragments: pick only the longest one # TODO should be pick the one with the most resolved residues
                        fragment_boundaries = boundaries.split(',')
                        fragment_sizes = []
                        for b in fragment_boundaries:
                            b = b.split(':')[0]
                            lower, upper = re.findall("-*\d+", b)
                            lower = int(lower)
                            # NOTE tread upper limit as exclusive
                            upper = int(upper[1:])
                            fragment_sizes.append(upper-lower)
                        boundaries = fragment_boundaries[np.argmax(fragment_sizes)]
                    boundaries, chain_id = boundaries.split(':')
                    lower, upper = re.findall("-*\d+", boundaries)

                    lower = int(lower)
                    upper = int(upper[1:])
                    # NOTE if lower > upper assume they included a prefix with some arcane numbering scheme. set lower to 1 and hope nothing to idiiotic happens by sticking to upper. inspired by 2f07B00 303-193
                    if lower > upper: lower = 1
                    assert lower != upper
                    l = upper - lower

                    pdb_file = get_pdb_filename(pdb_id, pdb_root)

                    parser = MMCIFParser()
                    with gzip.open(pdb_file, 'rt') as f:
                        s = parser.get_structure(pdb_id, f)

                    c = s[0][chain_id]
                    # NOTE some cifs are formatted weirdly
                    strand_ids = parser._mmcif_dict["_entity_poly.pdbx_strand_id"]
                    seq_idx = [idx for idx, s in enumerate(strand_ids) if chain_id in s][0]

                    # refseq = _get_reference_sequence(parser._mmcif_dict, seq_idx)
                    refseq = parser._mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"][seq_idx]
                    refseq = ''.join(refseq.split())

                    try:
                        try:
                            # TODO read the _struct_ref_seq_dif entries to modify the reference sequence
                            seq, ca_coords, cb_coords = construct_sample(lower, upper, c, refseq)
                        except ValueError as e:
                            refseq =  parser._mmcif_dict["_struct_ref.pdbx_seq_one_letter_code"][seq_idx]
                            refseq = ''.join(refseq.split())
                            seq, ca_coords, cb_coords = construct_sample(lower, upper, c, refseq)
                    except (AssertionError, RuntimeError) as e:
                        print(domain_id)
                        raise(e)

                    # residues = []
                    # for i in range(lower, upper):
                    #     if i in c:
                    #         residues.append(protein_letters_3to1[c[i].get_resname().capitalize()])
                    #     else:
                    #         residues.append('-')
                    # strucseq = [aa for aa in residues]

                    # if '-' in residues:
                    #     strucseq = _resolve_gaps(strucseq, seq)

                    # if ''.join(strucseq) not in seq:
                    #     degapped_strucseq = ''.join([subseq for subseq in ''.join(strucseq).split('-') if subseq != ''])
                    #     if degapped_strucseq not in seq:
                    #         raise RuntimeError("Reconstructed sequence does not match the expected data base sequence:\n{}\nsequence from atom entries: {}\nsequence without unresolved gaps:{}\nreference sequence: {}".format(domain_id, ''.join(strucseq), degapped_strucseq, seq))
                    #     else:
                    #         print("Warning: Missing residues could not be filled in from sequence information. Keeping unresolved gaps.", file=sys.stderr)

                    # ca_coords = np.full((l, 3), float('nan'), dtype=np.float32)
                    # cb_coords = np.full((l, 3), float('nan'), dtype=np.float32)

                    # for i in range(lower, upper):
                    #     if i in c:
                    #         r = c[i]
                    #         if 'CA' in r:
                    #             ca_coords[i-lower, :] = r['CA'].get_coord()[:]
                    #         if 'CB' in r:
                    #             cb_coords[i-lower, :] = r['CB'].get_coord()[:]
                    #         if r.get_resname() == 'GLY':
                    #             cb_coords[i-lower, :] = ca_coords[i-lower, :]

                    sample_group = handle.create_group(domain_id)
                    sequence_ds = sample_group.create_dataset('sequence', data=np.string_(seq))
                    ca_ds = sample_group.create_dataset('ca', data=ca_coords)
                    cb_ds = sample_group.create_dataset('cb', data=cb_coords)

        return

def _divide_by_alignment(segment, sequence):
    assert '-' not in segment
    s, seq = pairwise2.align.globalms(segment, sequence, match=10, mismatch=-100, open=-5, extend=0., penalize_end_gaps=False)[0][0:2]
    # if '-' in seq: print("unable to align \n{} to \n{} without introducing a gap in to reference".format(s, seq))
    if '-' in seq: raise ValueError("unable to align \n{} to \n{} without introducing a gap in to reference".format(s, seq))
    return re.findall(r'\w+', s)


# TODO optimize a little
# NOTE this does not work exactly right if there is e.g. a long gap and than a single AA at the end
# NOTE this will probably underestimate the size of the gap
# TODO unsolved: in 4hub there is a gap in the structure that is not properly indexed. solution: align only the subsequence to the reference, find the gap, fix resolved subsequences, continue
def _align_subseqs(strucseq, seq):
    # subseqs = re.findall(r'\w+|-+', ''.join(strucseq))
    # resolved_subseqs = [s for s in subseqs if s[0] != '-']
    resolved_subseqs = re.findall(r'\w+', ''.join(strucseq))


    # NOTE check if all resolved subsequences  are contained in the reference sequence
    # NOTE if not assume a gap was not indexed correctly, find the gap by alignment

    l = len(resolved_subseqs)
    i= 0
    while i < l:
        s = resolved_subseqs[i]
        if s not in seq:
            new_subseqs = _divide_by_alignment(s, seq)
            resolved_subseqs[i:i+1] = new_subseqs
            l = len(resolved_subseqs)
        i+=1


    # NOTE use longest resolved subsequence as anchor into reference sequence
    longest_resolved = max(resolved_subseqs, key=len)
    # longest_subseqs_index = subseqs.index(longest_resolved)
    longest_resolved_index = resolved_subseqs.index(longest_resolved)

    # longest_start_seq = seq.find(longest_resolved)
    # NOTE inspired by 3mlgA00 which consists of two identical pieces concatenated
    longest_start_seq = sum(len(s) for s in resolved_subseqs[:longest_resolved_index]) + seq[sum(len(s) for s in resolved_subseqs[:longest_resolved_index]):].find(longest_resolved)
    assert longest_start_seq >= 0
    longest_start_struc = ''.join(strucseq).find(longest_resolved)

    offset = longest_start_seq - sum(len(s) for s in resolved_subseqs[:longest_resolved_index])

    # subseqs_starts = [offset + sum([len(s) for s in subseqs[:i]]) for i in range(len(subseqs))]
    resolved_subseqs_starts = [offset + sum([len(s) for s in resolved_subseqs[:i]]) for i in range(len(resolved_subseqs))]

    # NOTE fix false indices from too long gaps before anchor
    for i in range(longest_resolved_index-1, -1, -1):
        s = resolved_subseqs[i]
        while s != seq[resolved_subseqs_starts[i]:resolved_subseqs_starts[i]+len(s)]:
            for j in range(i, -1, -1):
                resolved_subseqs_starts[j] -= 1
                assert resolved_subseqs_starts[j] >= 0

    # # NOTE fix false indices from too long gaps after anchor
    for i in range(longest_resolved_index+1, len(resolved_subseqs)):
        s = resolved_subseqs[i]
        while s != seq[resolved_subseqs_starts[i]:resolved_subseqs_starts[i]+len(s)]:
            # assert s in seq[resolved_subseqs_starts[i]:]
            for j in range(i, len(resolved_subseqs)):
                resolved_subseqs_starts[j] += 1
                assert resolved_subseqs_starts[j] <= len(seq)
    assert all([x >= 0 for x in resolved_subseqs_starts])

    result = ['-']*len(seq)
    for i in range(len(resolved_subseqs)):
        for j in range(len(resolved_subseqs[i])):
            result[resolved_subseqs_starts[i]+j] = resolved_subseqs[i][j]


    # # NOTE resolve gaps
    # for i in range((longest_subseqs_index+1)%2, len(subseqs), 2):
    #     # s = [x for x in subseqs[i]]
    #     s = ['-' for x in range(subseqs_starts[i+1]-subseqs_starts[i])]
    #     for j in range(subseqs_starts[i+1]-subseqs_starts[i]):
    #         s[j] = seq[subseqs_starts[i]+j]
    #     subseqs[i] = ''.join(s)


    # return ''.join(subseqs)
    return ''.join(result), seq


def construct_sample(lower, upper, c, sequence):
    resolvedseq = ''.join([protein_letters_3to1[c[i].get_resname().capitalize()] for i in range(lower, upper) if i in c])

    chainseq = ''.join([protein_letters_3to1[c[i].get_resname().capitalize()]  if i in c else '-' for i in range(lower, upper)])

    # TODO remove subseqs shorter than 3 residues

    unresolved_prefix_length = 0
    for i in range(lower, upper):
        if i in c: break
        unresolved_prefix_length+=1
    unresolved_postfix_length = 0
    for i in range(upper-1, lower-1, -1):
        if i in c: break
        unresolved_postfix_length+=1

    ca_coords = []
    cb_coords = []
    for i in range(lower, upper):
        if i in c:
            r = c[i]
            if 'CA' in r:
                ca_coords.append(r['CA'].get_coord())
            else:
                ca_coords.append(np.full((3), float('nan'), dtype=np.float32))
            if 'CB' in r:
                cb_coords.append(r['CB'].get_coord())
            else:
                cb_coords.append(np.full((3), float('nan'), dtype=np.float32))

            if r.get_resname() == 'GLY':
                cb_coords[-1] = ca_coords[-1]
    assert len(resolvedseq) == len(ca_coords) == len(cb_coords)
    alignment = _align_subseqs(chainseq, sequence)

    aseq1, aseq2 = alignment

    aligned_gap_prefix_length = 0
    for i in range(len(aseq1)):
        if aseq1[i] != '-': break
        aligned_gap_prefix_length += 1
    aligned_gap_postfix_length = 0
    for i in range(len(aseq1)-1, -1, -1):
        if aseq1[i] != '-': break
        aligned_gap_postfix_length += 1

    out_seq = []
    out_ca_coords = []
    out_cb_coords = []
    j = 0
    for i in range(aligned_gap_prefix_length-unresolved_prefix_length, min(len(aseq1)-aligned_gap_postfix_length + unresolved_postfix_length, len(aseq1))): # NOTE the min check is required by 6sxtA02, where the reference sequence is apparently missing the last two residues. the last three residues are not resolved so we just ignore the last two
        if aseq1[i] == '-':
            out_seq.append(aseq2[i])
            out_ca_coords.append(np.array([float('nan')]*3))
            out_cb_coords.append(np.array([float('nan')]*3))
        else:
            out_seq.append(aseq1[i])
            assert aseq1[i] == aseq2[i]
            assert aseq1[i] == resolvedseq[j]
            out_ca_coords.append(ca_coords[j])
            out_cb_coords.append(cb_coords[j])
            j += 1

    out_seq = ''.join(out_seq)
    out_ca_coords = np.array(out_ca_coords)
    out_cb_coords = np.array(out_cb_coords)

    if out_seq not in sequence:
        raise(RuntimeError("Reconstructed sequence does not match the expected data base sequence:\n{}\nsequence from atom entries: {}\nreconstructed sequence:{}\nreference sequence: {}".format(domain_id, ''.join(strucseq), degapped_strucseq, seq)))

    return out_seq, out_ca_coords, out_cb_coords

def _get_reference_sequence(mmcif_dict, seq_idx):
    raise NotImplementedError()
    # NOTE this entire thing does not work as intended yet
    # NOTE the entity poly entry is missing mutations sometimes, while the struct ref one is missing residues (probably not resolved ones?) inspired by 2zs0D00

    struct_seq = None
    if "_struct_ref.pdbx_seq_one_letter_code" in mmcif_dict:
        struct_seq =  mmcif_dict["_struct_ref.pdbx_seq_one_letter_code"][seq_idx]
        struct_seq = ''.join(struct_seq.split())
        # TODO replace (MSE) with M and find a solution for similar non-canonical residues
    entity_seq = mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"][seq_idx]
    entity_seq = ''.join(entity_seq.split())
    print("\nstruct: ", ''.join(struct_seq.split()), 'AAAAAAAAAAAAAAAAAAAAAA')
    print("entity: ", ''.join(entity_seq.split()), 'BBBBBBBBBBBBBBBBBBBBBB')
    if struct_seq is None or struct_seq == '?' or struct_seq in entity_seq or '_struct_ref_seq_dif.align_id' in mmcif_dict: # NOTE last condition inspired by 3bbzA00

        refseq = entity_seq
    else:
        # NOTE assuming that if we can't use entity_seq as reference we can use struct_seq without modifications
        assert '(' not in struct_seq and ')' not in struct_seq
        assert '(' not in entity_seq and ')' not in entity_seq
        # from Bio.pairwise2 import format_alignment
        # alignments = pairwise2.align.globalms(struct_seq, entity_seq, match=10, mismatch=-100, open=-5, extend=0., penalize_end_gaps=False)

        # aligned_struct_seq, aligned_entity_seq =  alignments[0][0:2]
        # for a in alignments:
        #     print(format_alignment(*a))
        #     break
        refseq = struct_seq


    return refseq
