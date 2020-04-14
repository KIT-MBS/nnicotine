import os
import glob

import numpy as np
import h5py
import torch


# TODO since for CASP we need C_beta coordinates i have to reparse the pdb data again anyway so most of this is useless.

MAX_SEQUENCE_LENGTH = 2000
AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19, 'Y': 20}

# NOTE adapted from github.com/biolib/openprotein and github.com/pytorch/vision

def preprocess_proteinnet(root, filename, out_filename):

    print("Pre-processing {}...".format(filename))

    if not os.path.isfile(os.path.join(root, out_filename)):
        process_file(root, filename, out_filename)

    return

def process_file(root, filename, out_filename, device='cpu'):
    out_file = h5py.File(os.path.join(root, out_filename), 'w')

    current_buffer_size = 1
    current_buffer_allocation = 0
    dset1 = out_file.create_dataset('primary', (current_buffer_size, MAX_SEQUENCE_LENGTH),
                                maxshape=(None, MAX_SEQUENCE_LENGTH), dtype='int32')
    dset2 = out_file.create_dataset('tertiary', (current_buffer_size, MAX_SEQUENCE_LENGTH, 9),
                                maxshape=(None, MAX_SEQUENCE_LENGTH, 9), dtype='float')
    dset3 = out_file.create_dataset('mask', (current_buffer_size, MAX_SEQUENCE_LENGTH),
                                maxshape=(None, MAX_SEQUENCE_LENGTH),
                                dtype='uint8')

    input_file_pointer = open(os.path.join(root, filename), "r")

    while True:
        # while there's more proteins to process
        next_protein = read_protein_from_file(input_file_pointer)
        if next_protein is None:
            break

        sequence_length = len(next_protein['primary'])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            print("Dropping protein as length too long:", sequence_length)
            continue

        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dset1.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))
            dset2.resize((current_buffer_size, MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        # masking and padding here happens so that the stored dataset is of the same size.
        # TODO store more efficiently. since only single items are ever loaded, there should be no reason to pad.
        primary_padded[:sequence_length] = next_protein['primary']
        t_transposed = np.ravel(np.array(next_protein['tertiary']).T)
        t_reshaped = np.reshape(t_transposed, (sequence_length, 9)).T

        tertiary_padded[:, :sequence_length] = t_reshaped
        mask_padded[:sequence_length] = next_protein['mask']

        mask = torch.Tensor(mask_padded).type(dtype=torch.bool)

        prim = torch.masked_select(torch.Tensor(primary_padded)
                                   .type(dtype=torch.long), mask)
        pos = torch.masked_select(torch.Tensor(tertiary_padded), mask)\
                  .view(9, -1).transpose(0, 1).unsqueeze(1) / 100

        pos = pos.to(device)

        # NOTE ??? why do they compute the dihedral from the backbone coordinates to then compute the backbone coordinates from the dihedral?
        # NOTE even if done correctly, only introduces rotation+numerical error

        # angles, batch_sizes = calculate_dihedral_angles_over_minibatch(pos,
        #                                                                [len(prim)],
        #                                                                use_gpu=use_gpu)

        # tertiary, _ = get_backbone_positions_from_angular_prediction(angles,
        #                                                              batch_sizes,
        #                                                              use_gpu=use_gpu)
        # tertiary = tertiary.squeeze(1)

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 9))

        length_after_mask_removed = len(prim)

        primary_padded[:length_after_mask_removed] = prim.data.cpu().numpy()
        # tertiary_padded[:length_after_mask_removed, :] = tertiary.data.cpu().numpy()
        pos = pos.squeeze()
        tertiary_padded[:length_after_mask_removed, :] = pos.data.cpu().numpy()
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

        dset1[current_buffer_allocation] = primary_padded
        dset2[current_buffer_allocation] = tertiary_padded
        dset3[current_buffer_allocation] = mask_padded
        current_buffer_allocation += 1

    print("Wrote output to", current_buffer_allocation, "proteins to", out_filename)
    return

def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])

def read_protein_from_file(file_pointer):
    """The algorithm Defining Secondary Structure of Proteins (DSSP) uses information on e.g. the
    position of atoms and the hydrogen bonds of the molecule to determine the secondary structure
    (helices, sheets...).
    """
    dict_ = {}
    _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
    _mask_dict = {'-': 0, '+': 1}

    while True:
        next_line = file_pointer.readline()
        if next_line == '[ID]\n':
            id_ = file_pointer.readline()[:-1]
            dict_.update({'id': id_})
        elif next_line == '[PRIMARY]\n':
            primary = encode_primary_string(file_pointer.readline()[:-1])
            dict_.update({'primary': primary})
        elif next_line == '[EVOLUTIONARY]\n':
            evolutionary = []
            for _residue in range(21):
                evolutionary.append([float(step) for step in file_pointer.readline().split()])
            dict_.update({'evolutionary': evolutionary})
        elif next_line == '[SECONDARY]\n':
            secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
            dict_.update({'secondary': secondary})
        elif next_line == '[TERTIARY]\n':
            tertiary = []
            # 3 dimension
            for _axis in range(3):
                tertiary.append([float(coord) for coord in file_pointer.readline().split()])
            dict_.update({'tertiary': tertiary})
        elif next_line == '[MASK]\n':
            mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
            dict_.update({'mask': mask})
        elif next_line == '\n':
            return dict_
        elif next_line == '':
            return None
