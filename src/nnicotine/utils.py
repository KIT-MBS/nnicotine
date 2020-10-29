from .datasets import CATHDerived
from torch.utils.data import DataLoader
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
