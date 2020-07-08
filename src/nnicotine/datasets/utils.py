import os
import random

from Bio.Data.IUPACData import protein_letters_3to1 as IUPAC_letters

def cath_train_test_split(domains, max_test_family_size=200, min_test_samples=1800, random_state=None, **kwargs):
    superfamilies = {}
    for domain in domains:
        domain_id, version, cath_code, boundaries = domain
        superfamily = cath_code.split('.')[3]
        if superfamily not in superfamilies:
            superfamilies[superfamily] = 1
        else:
            superfamilies[superfamily] += 1

    possible_test_superfamilies = [x for x in superfamilies if superfamilies[x] < max_test_family_size]
    test_families = set()
    test_samples = 0

    random.seed(random_state)
    random.shuffle(possible_test_superfamilies)
    for f in possible_test_superfamilies:
        test_families.add(f)
        test_samples += superfamilies[f]
        if test_samples > min_test_samples: break

    train_domains = []
    test_domains = []

    for domain in domains:
        if domain[2].split('.')[3] in test_families:
            test_domains.append(domain)
        else:
            train_domains.append(domain)
    assert len(test_domains) > min_test_samples

    return train_domains, test_domains

def get_pdb_filename(pdb_code, pdb_root):
    pdb_filename = os.path.join(pdb_root, "divided/{}/{}.cif.gz".format(pdb_code[1:3], pdb_code))
    if not os.path.isfile(pdb_filename):
        pdb_filename = os.path.join(pdb_root, "obsolete/{}/{}.cif.gz".format(pdb_code[1:3], pdb_code))

    return pdb_filename


def generate_msa(dataset):
    return

protein_letters_3to1 = {key : IUPAC_letters[key] for key in IUPAC_letters}
protein_letters_3to1['Xaa'] = 'X'
protein_letters_3to1['Unk'] = 'X'
protein_letters_3to1['Sec'] = 'C' # selenocysteine gets mapped to cysteine
protein_letters_3to1['Pyl'] = 'K' # pyrrolysine gets mapped to lysine
protein_letters_3to1['Mse'] = 'M' # selenomethionine gets mapped to methionine
