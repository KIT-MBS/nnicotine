import numpy as np
from torchvision import transforms as T
from pydca.meanfield_dca import meanfield_dca
from pydca.plmdca import plmdca


def get_transforms(train=True, p=0.3):
    sample_transforms = []
    target_transforms = []
    if train:
        sample_transforms.append(RandomDrop(p))
    sample_transforms.append(TrimToTarget())
    sample_transforms.append(ComputeCouplings())
    sample_transforms.append(ComputeCouplings())
    sample_transforms.append(ToTensor)

    target_transforms.append(ToContact('cb', '3.0'))
    return sample_transforms, target_transforms


class RandomDrop():
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample):
        # TODO
        sample['msa'] = sample['msa'][0] + np.random.shuffle(sample['msa'][1:])[:int((1.-p)*(len(sample['msa'])-1))]

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

        # NOTE using a 2d np array an transforming back to msa is probably overall better
        # indices = np.array([i for i,c in enumerate(gapped_target) if c != '-'])
        # trimmed = np.array([list(r) for r in sample['msa']], np.character)
        # sample['msa'][:,:L] = trimmed[:,:]
        # sample['msa'] = sample['msa'][:, :L]

        return sample


# TODO other dca params
class ComputeCouplings():
    def __init__(self, num_threads=None):
        self.num_threads = num_threads
        return
    def __call__(self, sample):
        dca = plmdca.PlmDCA(sample['msa'], 'protein', num_threads=self.num_threads)
        freqs = dca.get_reg_single_site_freqs()
        pair_freqs = dca.get_ref_pair_site_freqs()
        corr_mat = dca.construct_corr_mat(freqs, pair_freqs)
        couplings = dca.compute_couplings(corr_mat)
        print(couplings.shape)
        # TODO
        raise

        del sample['msa']
        # TODO
        sample['biases'] = None
        sample['couplings'] = couplings

        return


class ToTensor():
    def __init__(self):
        return
    def __call__(self):
        return


class ToContact():
    def __init__(self, atom='cb', threshold=5.0):
        self.atom = atom
        self.threshold = threshold

    def __call__(self, coords):
        coords = coords[self.atom]

        return


class ToDist():
    def __init__(self):
        return
    def __call__(self):
        return
