import os
import glob

import numpy as np
import h5py
import torch

# NOTE adapted from github.com/biolib/openprotein

aa2index_dict = {x: i for (i,x) in enumerate("-ACDEFGHIKLMNPQRSTVWY")}
dssp2index_dict = {x: i for (i,x) in enumerate("LHBEGITS")}
mask2index_dict = {x: i for (i,x) in enumerate("-+")}

def read_entry(file_pointer):
    entry = {}

    while True:
        next_line = file_pointer.readline()
        if next_line == '[ID]\n':
            id_ = file_pointer.readline()[:-1]
            entry.update({'id': id_})
        elif next_line == '[PRIMARY]\n':
            primary = file_pointer.readline()[:-1]
            primary = np.string_(primary)
            entry.update({'primary': primary})
        elif next_line == '[EVOLUTIONARY]\n':
            evolutionary = []
            # 21 dimensions
            for _residue in range(21):
                evolutionary.append([float(step) for step in file_pointer.readline().split()])
            evolutionary = np.array(evolutionary)
            evolutionary = evolutionary.reshape(21, -1)
            np.transpose(evolutionary)

            entry.update({'evolutionary': evolutionary})
        elif next_line == '[SECONDARY]\n':
            secondary = np.string_(file_pointer.readline()[:-1])
            entry.update({'secondary': secondary})
        elif next_line == '[TERTIARY]\n':
            tertiary = []
            # 3 dimensions
            for _axis in range(3):
                tertiary.append([float(coord) for coord in file_pointer.readline().split()])
            tertiary = np.array(tertiary)*100.
            tertiary = tertiary.reshape(3, -1, 3)
            n_coord = tertiary[:,:,0]
            ca_coord = tertiary[:,:,1]
            c_coord = tertiary[:,:,2]
            n_coord = np.transpose(n_coord)
            ca_coord = np.transpose(ca_coord)
            c_coord = np.transpose(c_coord)
            entry.update({'n': n_coord})
            entry.update({'ca': ca_coord})
            entry.update({'c': c_coord})
        elif next_line == '[MASK]\n':
            mask = list([mask2index_dict[aa] for aa in file_pointer.readline()[:-1]])
            entry.update({'mask': mask})
        elif next_line == '\n':
            return entry
        elif next_line == '':
            return None


# NOTE this is not how one would usually organize an hdf5 file
# NOTE for now we will stick to one collection of datasets per sample instead of one dataset per features for all samples
def convert_proteinnet_file(in_filename, out_filename):

    file_pointer = open(in_filename, 'r')
    next_entry = read_entry(file_pointer)
    n = 0
    while next_entry is not None:
        next_entry = read_entry(file_pointer)
        n +=1

    with h5py.File(out_filename, 'w') as handle:

        fullnames_dset = handle.create_dataset('ids', (n,), dtype='S12')
        pdb_id_dset = handle.create_dataset('pdbids', (n,), dtype='S4')
        chain_id_dset = handle.create_dataset('chainids', (n,), dtype='S1')

        with open(in_filename, 'r') as file_pointer:
            for i in range(n):
                entry = read_entry(file_pointer)
                ID = entry['id']
                if len(ID.split('_')) == 3:
                    pdbcode, chain_number, chain_id = ID.split('_')
                elif len(ID.split('_')) == 2:
                    pdbcode, sid = ID.split('_')
                    # TODO what about a chain_id with more than one symbol?
                    chain_id = sid[5]

                assert len(chain_id) == 1

                fullnames_dset[i] = np.string_(entry['id'])
                pdb_id_dset[i] = np.string_(pdbcode)
                chain_id_dset[i] = np.string_(chain_id)

                sample_group = handle.create_group(pdbcode)

                l = len(entry['primary'])

                for key in entry:
                    if key == 'id': continue
                    ds = sample_group.create_dataset(key, data=entry[key])


# TODO
def collect_cbeta_coordinates(root):
    for ID in ids:
        if len(ID.split('_')) == 3:
            pdbcode, chain_number, chain_id = ID.split('_')
            pdbfilename = os.path.join("/dividede/", pdbcode[1:3]+'/'+pdbcode+".cif.gz")
            pdbfilename = os.path.join("/obsolete/", pdbcode[1:3]+'/'+pdbcode+".cif.gz")
            #
        elif len(ID.split('_')) == 2:
            pdbcode, sid = ID.split('_')

    return
