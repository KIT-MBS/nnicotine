import os
import random

from Bio.Data.IUPACData import protein_letters_3to1_extended as IUPAC_letters

def cath_train_test_split(domains, max_test_family_size=200, min_test_samples=1800, random_state=None, **kwargs):

    current = 0
    obsolete = 0
    # NOTE count references to obsolete pdb files
    pdb_root = kwargs.get('pdb_root')

    domains = [d for d in domains if check_file_exists(d, pdb_root)]

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

def check_file_exists(domain, pdb_root):
    pdb_id = domain[0][:4]
    return os.path.isfile(os.path.join(pdb_root, "divided/{}/{}.cif.gz".format(pdb_id[1:3], pdb_id)))

def get_pdb_filename(pdb_code, pdb_root):
    pdb_filename = os.path.join(pdb_root, "divided/{}/{}.cif.gz".format(pdb_code[1:3], pdb_code))
    if not os.path.isfile(pdb_filename):
        pdb_filename = os.path.join(pdb_root, "obsolete/{}/{}.cif.gz".format(pdb_code[1:3], pdb_code))

    return pdb_filename


def generate_msa(dataset):
    return

protein_letters_3to1 = {key : IUPAC_letters[key] for key in IUPAC_letters}
# TODO use extended alphabet. if that's a problem with the alignment tool, use only canonical and map the rest to sensible canonical ones
protein_letters_3to1['Unk'] = 'X'
# protein_letters_3to1['Xaa'] = 'X'
# protein_letters_3to1['Sec'] = 'C' # selenocysteine gets mapped to cysteine
# protein_letters_3to1['Pyl'] = 'K' # pyrrolysine gets mapped to lysine
protein_letters_3to1['Mse'] = 'M' # selenomethionine gets mapped to methionine
# ambiguous
# protein_letters_3to1['Asx'] = '' aspartic acid or asparagine
# protein_letters_3to1['Glx'] = '' glutamic acid or glutamine
# protein_letters_3to1['Xle'] = '' leucine or isoleucine
# protein_letters_3to1['Xle'] = ''
