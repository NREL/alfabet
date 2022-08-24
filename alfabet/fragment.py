import logging
from collections import Counter
from typing import Dict, Iterator, Type

import pandas as pd
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class Molecule:
    def __init__(self, mol: Type[rdkit.Chem.Mol] = None, smiles: str = None) -> None:
        assert (mol is not None) or (
            smiles is not None
        ), "mol or smiles must be provided"

        self._mol = mol
        self._smiles = smiles
        self._molH = None
        self._is_canon = False

    @property
    def mol(self) -> Type[rdkit.Chem.Mol]:
        if self._mol is None:
            self._mol = rdkit.Chem.MolFromSmiles(self._smiles)
        return self._mol

    @property
    def molH(self) -> Type[rdkit.Chem.Mol]:
        if self._molH is None:
            self._molH = rdkit.Chem.AddHs(self.mol)
        return self._molH

    @property
    def smiles(self) -> str:
        if (self._smiles is None) or not self._is_canon:
            self._smiles = rdkit.Chem.MolToSmiles(self.mol)
        return self._smiles


def get_fragments(
    input_molecule: Molecule, drop_duplicates: bool = False
) -> pd.DataFrame:
    df = pd.DataFrame(fragment_iterator(input_molecule))
    if drop_duplicates:
        df = df.drop_duplicates(["fragment1", "fragment2"]).reset_index(drop=True)
    return df


def fragment_iterator(
    input_molecule: str, skip_warnings: bool = False
) -> Iterator[Dict]:

    mol_stereo = count_stereocenters(input_molecule)
    if (mol_stereo["atom_unassigned"] != 0) or (mol_stereo["bond_unassigned"] != 0):
        logging.warning(
            f"Molecule {input_molecule.smiles} has undefined stereochemistry"
        )
        if skip_warnings:
            return

    rdkit.Chem.Kekulize(input_molecule.molH, clearAromaticFlags=True)

    for bond in input_molecule.molH.GetBonds():

        if bond.IsInRing():
            continue

        if bond.GetBondTypeAsDouble() > 1.9999:
            continue

        try:

            # Use RDkit to break the given bond
            mh = rdkit.Chem.RWMol(input_molecule.molH)
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
            frag1, frag2 = sorted(fragmented_smiles.split("."))
            frag1 = Molecule(smiles=frag1)
            frag2 = Molecule(smiles=frag2)

            # Stoichiometry check
            assert (
                count_atom_types(frag1) + count_atom_types(frag2)
            ) == count_atom_types(input_molecule), "Error with {}; {}; {}".format(
                frag1.smiles, frag2.smiles, input_molecule.smiles
            )

            # Check introduction of new stereocenters
            is_valid_stereo = check_stereocenters(frag1) and check_stereocenters(frag2)

            yield {
                "molecule": input_molecule.smiles,
                "bond_index": bond.GetIdx(),
                "bond_type": get_bond_type(bond),
                "fragment1": frag1.smiles,
                "fragment2": frag2.smiles,
                "is_valid_stereo": is_valid_stereo,
            }

        except ValueError:
            logging.error(
                "Fragmentation error with {}, bond {}".format(
                    input_molecule.smiles, bond.GetIdx()
                )
            )
            continue


def count_atom_types(molecule: Type[Molecule]):
    """Return a dictionary of each atom type in the given fragment or molecule"""
    return Counter([atom.GetSymbol() for atom in molecule.molH.GetAtoms()])


def count_stereocenters(molecule: Type[Molecule]) -> Dict:
    """Returns a count of both assigned and unassigned stereocenters in the
    given molecule"""

    rdkit.Chem.FindPotentialStereoBonds(molecule.mol)

    stereocenters = rdkit.Chem.FindMolChiralCenters(
        molecule.mol, includeUnassigned=True
    )
    stereobonds = [
        bond
        for bond in molecule.mol.GetBonds()
        if bond.GetStereo() is not rdkit.Chem.rdchem.BondStereo.STEREONONE
    ]

    atom_assigned = len([center for center in stereocenters if center[1] != "?"])
    atom_unassigned = len([center for center in stereocenters if center[1] == "?"])

    bond_assigned = len(
        [
            bond
            for bond in stereobonds
            if bond.GetStereo() is not rdkit.Chem.rdchem.BondStereo.STEREOANY
        ]
    )
    bond_unassigned = len(
        [
            bond
            for bond in stereobonds
            if bond.GetStereo() is rdkit.Chem.rdchem.BondStereo.STEREOANY
        ]
    )

    return {
        "atom_assigned": atom_assigned,
        "atom_unassigned": atom_unassigned,
        "bond_assigned": bond_assigned,
        "bond_unassigned": bond_unassigned,
    }


def check_stereocenters(molecule: Type[Molecule]):
    """Check the given SMILES string to determine whether accurate
    enthalpies can be calculated with the given stereochem information
    """
    stereocenters = count_stereocenters(molecule)
    if stereocenters["bond_unassigned"] > 0:
        return False

    max_unassigned = 1 if stereocenters["atom_assigned"] == 0 else 1
    if stereocenters["atom_unassigned"] <= max_unassigned:
        return True
    else:
        return False


def get_bond_type(bond):
    return "{}-{}".format(
        *tuple(sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())))
    )
