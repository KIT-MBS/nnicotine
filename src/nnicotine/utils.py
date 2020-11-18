import numpy as np
from torch.utils.data import DataLoader
from .datasets import CATHDerived
from .transforms import get_transforms

def get_data_loaders(root, ds_version='20200708', batch_size=1, toy=False):
    """
    root: root folder for dataset
    ds_version: version string of dataset
    batch_size: training batch size TODO
    """
    collate_fn = None

    sample_transform, target_transform = get_transforms()

    train_ds_mode = 'train'
    test_ds_mode =  'test'
    if toy:
        train_ds_mode = 'toy'
        test_ds_mode = 'toy'

    train_ds = CATHDerived(root, mode=train_ds_mode, version=ds_version, transform=sample_transform, target_transform=target_transform)
    test_ds = CATHDerived(root, mode=test_ds_mode, version=ds_version, transform=sample_transform, target_transform=target_transform)

    if batch_size != 1:
        raise NotImplementedError()

    # TODO more workers, data parallel and stuff
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader

def _compute_sequence_identity(seqA, seqB):
    matches = [aa1 == aa2 for aa1, aa2 in zip(seqA, seqB) if (aa1!='-' or aa2!='-')]
    return sum(matches)/len(matches)


def compute_msa_weights(msa, threshold=.8):
    """
    msa (Bio.Align.MultipleSeqAlignment): alignment for which sequence frequency based weights are to be computed
    threshold (float): sequence identity threshold for reweighting

    NOTE that columns where both sequences have a gap will not be taken into account when computing identity
    """

    weights = np.zeros(len(msa))

    seq_identities = np.zeros((len(msa), len(msa)))


    for i in range(len(msa)):
        for j in range(i+1, len(msa)):
            seq_identities[i, j] = _compute_sequence_identity(msa[i], msa[j])
    seq_identities = seq_identities + np.diag(np.ones(len(msa)))

    ms = np.sum(seq_identities>threshold, 1)
    weights = 1./ms
    return weights
