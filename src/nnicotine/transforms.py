import numpy as np
import torch
from torchvision import transforms as T
from Bio.Align import MultipleSeqAlignment
from Bio.Data.IUPACData import protein_letters

from pydca.meanfield_dca import meanfield_dca
from pydca.plmdca import plmdca


aas = "-" + protein_letters
default_categories = {aa : i for (i, aa) in enumerate(aas)}


def get_transforms(train=True, p=0.3, atom='cb'):
    sample_transforms = []
    target_transforms = []
    if train:
        sample_transforms.append(RandomDrop(p))
    sample_transforms.append(TrimToTarget())
    sample_transforms.append(ToCategorical())
    sample_transforms.append(RandomDrop(p))
    sample_transforms.append(ComputeCouplings())
    sample_transforms.append(ToTensor())
    sample_transforms = Compose(sample_transforms)

    target_transforms.append(ToTensor())
    target_transforms.append(ToDistance('cb'))
    target_transforms.append(ToBinnedDistance())

    # target_transforms.append(MaskShortRange())
    target_transforms = Compose(target_transforms)
    return sample_transforms, target_transforms


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomDrop():
    def __init__(self, p=0.3):
        self.p = p

    # TODO optimize
    def __call__(self, sample):
        msa = sample['msa'][0:1]
        sequences = [sample['msa'][i] for i in range(1, len(sample['msa']))]
        np.random.shuffle(sequences)

        msa.extend(MultipleSeqAlignment(sequences[:int(len(sequences)*(1.-self.p))], alphabet=msa._alphabet, annotations=msa.annotations, column_annotations=msa.column_annotations))
        sample['msa'] = msa

        # sample['msa'] = sample['msa'][0:1].extend(np.random.shuffle(sample['msa'][1:])[:int((1.-p)*(len(sample['msa'])-1))])

        return sample


class TrimToTarget():
    def __init__(self):
        return
    def __call__(self, sample):
        assert sample['sequence'] == str(sample['msa'][0].seq).replace('-', '')
        L = sample['msa'].get_alignment_length()
        gapped_target_sequence = sample['msa'][0].seq

        cols = [sample['msa'][:, i:i+1] for i in range(L) if gapped_target_sequence[i] != '-']
        trimmed = cols[0]
        for i in range(1, len(cols)):
            trimmed += cols[i]
        sample['msa'] = trimmed

        # NOTE using a 2d np array an transforming back to msa is probably better overall
        # indices = np.array([i for i,c in enumerate(gapped_target) if c != '-'])
        # trimmed = np.array([list(r) for r in sample['msa']], np.character)
        # sample['msa'][:,:L] = trimmed[:,:]
        # sample['msa'] = sample['msa'][:, :L]

        return sample


# TODO other dca params
# TODO PLMDCA
class ComputeCouplings():
    def __init__(self, num_threads=None):
        self.num_threads = num_threads
        return
    def __call__(self, sample):
        dca = meanfield_dca.MeanFieldDCA(sample['msa'], 'protein')
        # dca = plmdca.PlmDCA(sample['msa'], 'protein', num_threads=self.num_threads)

        fields = dca.compute_fields()

        freqs = dca.get_reg_single_site_freqs()
        pair_freqs = dca.get_reg_pair_site_freqs()
        corr_mat = dca.construct_corr_mat(freqs, pair_freqs)
        couplings = dca.compute_couplings(corr_mat)
        # apparently the gap state is not taken into account here
        l = dca.sequences_len
        nstates = dca.num_site_states -1
        couplings = couplings.reshape(l, nstates, l, nstates)
        couplings = np.swapaxes(couplings, 1, 3)
        couplings = couplings.reshape(l, l, -1)
        # couplings = np.ascontiguousarray(couplings)

        sample['fields'] = np.array([fields[i] for i in range(len(fields))], dtype=np.float32)
        del sample['msa']
        sample['couplings'] = couplings.astype(np.float32)

        return sample


class ToTensor():
    def __init__(self, keys=None):
        self.keys = keys
        return
    def __call__(self, sample):
        keys = self.keys
        if keys is None:
            keys = sample.keys() # TODO check for ordering

        for k in keys:
            sample[k] = torch.from_numpy(sample[k])

        return sample



class ToCategorical():
    def __init__(self, categories=default_categories):
        self.categories = categories

    def __call__(self, sample):
        s = sample['sequence']
        s = np.array([self.categories[l] for l in s])
        sample['sequence'] = s
        return sample


# NOTE probably not needed for distance prediction
class MaskShortRange():
    def __init__(self, mindist=12):
        return
    def __call__(self, sample):
        return sample


class ToBinnedDistance():
    """
    NOTE order is inverted, bin 0 is overflow, bin nbins-1 is underflow
    """
    def __init__(self, nbins=64, mindist=2., maxdist=22.):
        self.nbins = nbins
        self.mindist = mindist
        self.maxdist = maxdist

    def __call__(self, dist):
        labels = torch.zeros_like(dist, dtype=torch.long)
        upper = self.maxdist
        delta = (self.maxdist-self.mindist)/self.nbins
        for i in range(self.nbins-1):
            upper -= delta
            labels += dist < upper

        return labels


# TODO use squared distance instead?
class ToDistance():
    def __init__(self, atom='cb'):
        self.atom = atom
        return
    def __call__(self, sample):
        y = sample[self.atom]
        l, d = y.size()
        y1 = y.unsqueeze(1)
        y2 = y.unsqueeze(0)
        y1 = y1.expand(l, l, d)
        y2 = y2.expand(l, l, d)

        dist = y2 - y1
        dist = dist * dist
        dist = dist.sum(dim=-1)
        return dist.sqrt()
