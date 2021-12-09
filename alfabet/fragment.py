import logging
from collections import Counter
from typing import Iterator

import pandas as pd
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem


def get_fragments(smiles: str) -> pd.DataFrame:
    return pd.DataFrame(fragment_iterator(smiles))


def fragment_iterator(smiles: str,
                      skip_warnings: bool = False
                      ) -> Iterator[pd.Series]:
    mol_stereo = count_stereocenters(smiles)
    if ((mol_stereo['atom_unassigned'] != 0) or
            (mol_stereo['bond_unassigned'] != 0)):
        logging.warning(f'Molecule {smiles} has undefined stereochemistry')
        if skip_warnings:
            return

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
                    == count_atom_types(smiles)), \
                "Error with {}; {}; {}".format(
                    frag1, frag2, smiles)

            # Check introduction of new stereocenters
            is_valid_stereo = (check_stereocenters(frag1) and
                               check_stereocenters(frag2))

            yield pd.Series({
                'molecule': smiles,
                'bond_index': bond.GetIdx(),
                'bond_type': get_bond_type(bond),
                'fragment1': frag1,
                'fragment2': frag2,
                'is_valid_stereo': is_valid_stereo
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


def canonicalize_smiles(smiles: str) -> str:
    """ Return a consistent SMILES representation for the given molecule """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return rdkit.Chem.MolToSmiles(mol)


def count_stereocenters(smiles: str) -> pd.Series:
    """ Returns a count of both assigned and unassigned stereocenters in the 
    given molecule """

    mol = rdkit.Chem.MolFromSmiles(smiles)
    rdkit.Chem.FindPotentialStereoBonds(mol)

    stereocenters = rdkit.Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    stereobonds = [bond for bond in mol.GetBonds() if bond.GetStereo() is not
                   rdkit.Chem.rdchem.BondStereo.STEREONONE]

    atom_assigned = len([center for center in stereocenters if center[1] != '?'])
    atom_unassigned = len([center for center in stereocenters if center[1] == '?'])

    bond_assigned = len([bond for bond in stereobonds if bond.GetStereo() is not
                         rdkit.Chem.rdchem.BondStereo.STEREOANY])
    bond_unassigned = len([bond for bond in stereobonds if bond.GetStereo() is
                           rdkit.Chem.rdchem.BondStereo.STEREOANY])

    return pd.Series({'atom_assigned': atom_assigned,
                      'atom_unassigned': atom_unassigned,
                      'bond_assigned': bond_assigned,
                      'bond_unassigned': bond_unassigned})


def check_stereocenters(smiles):
    """Check the given SMILES string to determine whether accurate
    enthalpies can be calculated with the given stereochem information
    """
    stereocenters = count_stereocenters(smiles)
    if stereocenters['bond_unassigned'] > 0:
        return False

    max_unassigned = 1 if stereocenters['atom_assigned'] == 0 else 1
    if stereocenters['atom_unassigned'] <= max_unassigned:
        return True
    else:
        return False


def get_bond_type(bond):
    return "{}-{}".format(
        *tuple(sorted((bond.GetBeginAtom().GetSymbol(),
                       bond.GetEndAtom().GetSymbol()))))
