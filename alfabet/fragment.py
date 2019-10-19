import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from collections import Counter
import pandas as pd

import logging


def get_fragments(smiles):
    return pd.DataFrame(fragment_iterator(smiles))


def fragment_iterator(smiles):

    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.rdmolops.AddHs(mol)
    rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)

    for bond in mol.GetBonds():

        if bond.IsInRing():
            continue

        if bond.GetBondTypeAsDouble() > 1.9999:
            continue

        try:

            # Use RDkit to break the given bond
            mh = rdkit.Chem.RWMol(mol)
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            mh.RemoveBond(a1, a2)

            mh.GetAtomWithIdx(a1).SetNoImplicit(True)
            mh.GetAtomWithIdx(a2).SetNoImplicit(True)

            # Call SanitizeMol to update radicals
            rdkit.Chem.SanitizeMol(mh)

            # Convert the two molecules into a SMILES string
            fragmented_smiles = rdkit.Chem.MolToSmiles(mh)

            # Split fragment and canonicalize
            frag1, frag2 = sorted(fragmented_smiles.split('.'))
            frag1 = canonicalize_smiles(frag1)
            frag2 = canonicalize_smiles(frag2)

            # Stoichiometry check
            assert ((count_atom_types(frag1) + count_atom_types(frag2)) 
                    == count_atom_types(smiles)), "Error with {}; {}; {}".format(
                        frag1, frag2, smiles)
            
            # Check change in number of stereocenters
            delta_stereocenters = (
                (enumerate_stereocenters(frag1) 
                 + enumerate_stereocenters(frag2))
                - enumerate_stereocenters(smiles))

            yield pd.Series({
                'molecule': smiles,
                'bond_index': bond.GetIdx(),
                'bond_type': get_bond_type(bond),
                'fragment1': frag1,
                'fragment2': frag2,
                'delta_assigned_stereo': delta_stereocenters['assigned'],
                'delta_unassigned_stereo': delta_stereocenters['unassigned'],
            })

        except ValueError:
            logging.error('Fragmentation error with {}, bond {}'.format(
                smiles, bond.GetIdx()))
            continue


def count_atom_types(smiles):
    """ Return a dictionary of each atom type in the given fragment or molecule
    """
    mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=True)
    mol = rdkit.Chem.rdmolops.AddHs(mol)
    return Counter([atom.GetSymbol() for atom in mol.GetAtoms()])


def canonicalize_smiles(smiles):
    """ Return a consistent SMILES representation for the given molecule """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return rdkit.Chem.MolToSmiles(mol)


def enumerate_stereocenters(smiles):
    """ Returns a count of both assigned and unassigned stereocenters in the 
    given molecule """
    
    mol = rdkit.Chem.MolFromSmiles(smiles)
    stereocenters = rdkit.Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    assigned = len([center for center in stereocenters if center[1] != '?'])
    unassigned = len([center for center in stereocenters if center[1] == '?'])

    return pd.Series({'assigned': assigned, 'unassigned': unassigned})


def get_bond_type(bond):
    return "{}-{}".format(
        *tuple(sorted((bond.GetBeginAtom().GetSymbol(),
                       bond.GetEndAtom().GetSymbol()))))

