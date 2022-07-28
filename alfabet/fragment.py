#
#  Copyright (C) 2022
#   @@ All Rights Reserved @@
#
#  This file is part of the ALFABET program.
#  The contents are covered by the terms of the license
#  which is included in the file LICENSE, found at the root
#  of the source tree.
#

# Contributor: Ichiru Take
# This module is to support only the RDKit molecule processing.

from collections import defaultdict
from logging import warning
from typing import Dict, Union, List, Tuple

from rdkit.Chem import CanonSmiles, FindMolChiralCenters
from rdkit.Chem.rdchem import Mol, RWMol, BondType, BondStereo
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import (AddHs, Kekulize, SanitizeMol, AssignStereochemistry,
                                 FindPotentialStereoBonds, RemoveHs)


def sanitize(mol: Mol, *args, **kwargs) -> Mol:
    if mol is not None:
        try:
            _ = SanitizeMol(mol, *args, **kwargs)
        except (ValueError, RuntimeError):
            pass
    return mol


def smiles_to_mol(smiles: str, Hs: bool = True, *args, **kwargs) -> Mol:
    if isinstance(smiles, str):
        mol = MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}. Please check your SMILES notation.")
    else:
        raise TypeError("SMILES must be a valid python string.")
    try:
        if Hs:
            mol = AddHs(mol)
        _ = SanitizeMol(mol, *args, **kwargs)
    except (ValueError, RuntimeError):
        raise ValueError(f"Invalid molecule: {smiles}. Please check your SMILES notation.")
    return mol


def find_stereo_atoms_bonds(mol: Mol, legacy: bool = True) -> None:
    if legacy:
        AssignStereochemistry(mol, force=False, cleanIt=True, flagPossibleStereoCenters=True)
    FindPotentialStereoBonds(mol, cleanIt=True)


def count_stereocenters(mol: Mol, legacy: bool = True) -> Dict:
    """
        Returns a count of both assigned and unassigned stereo-centers in the
        given molecule.
    """
    find_stereo_atoms_bonds(mol, legacy=legacy)

    stereo_centers: List = FindMolChiralCenters(mol, includeUnassigned=True)

    stereo_bonds = [bond for bond in mol.GetBonds() if bond.GetStereo() is not BondStereo.STEREONONE]
    atom_assigned = len([center for center in stereo_centers if center[1] != "?"])
    atom_unassigned = len(stereo_centers) - atom_assigned

    bond_assigned = len([bond for bond in stereo_bonds if bond.GetStereo() is not BondStereo.STEREOANY])
    bond_unassigned = len(stereo_bonds) - bond_assigned

    return {"atom_assigned": atom_assigned, "atom_unassigned": atom_unassigned,
            "bond_assigned": bond_assigned, "bond_unassigned": bond_unassigned}


def check_stereocenters(mol: Mol, legacy: bool = True) -> bool:
    """
        Check the given SMILES string to determine whether accurate enthalpies
        can be calculated with the given stereo-chemistry information.
    """
    stereo_centers = count_stereocenters(mol, legacy=legacy)
    if stereo_centers["bond_unassigned"] > 0:
        return False

    max_unassigned = 1 if stereo_centers["atom_assigned"] == 0 else 1  # I don't get it here.
    return stereo_centers["atom_unassigned"] <= max_unassigned


class MolProcessor:
    # __slots__ = ('mol', )                     # For future Python (current minimal is 3.6 ?)

    def __init__(self, mol: Union[Mol, str] = None) -> None:
        if isinstance(mol, str):
            mol = smiles_to_mol(mol)
        self.mol: Mol = mol
        self._col = ['molecule', 'fragment1', 'fragment2', 'bond_index', 'bond_type', 'is_valid_stereo']

    # -------------------------------------------------------------------------
    def get_reactions(self, to_dict: bool, reverse_reaction: bool = False, single_bond_only: bool = True,
                      strict_radical: bool = True, zero_duplicate: bool = True, aro_ring: bool = False,
                      non_aro_ring: bool = False, aro_ring_attached: bool = True,
                      non_aro_ring_attached: bool = True, non_ring_attached: bool = True) -> Union[Dict, List]:

        """
            This method is to get all bonds inside molecule (in SMILES form) into a series of bonds.

            Arguments:
            ----------
                - to_dict (bool): If True, return a dictionary.
                - reverse_reaction (bool): If True and the bond is not a ring-membered or the directed bond,
                    an extra copy is also returned.
                - single_bond_only (bool): If True, only the single bond and the aromatic bond is returned.
                - zero_duplicate (bool): If True, all duplicated bonds are removed.
                - strict_radical (bool): If True, whenever enable arg::AroRing or arg::NonAroRing.
                    If the molecule contained cycles, only one radical (the second column) is returned
                    (The second value is always a string::NONE). If False, both radicals are identically
                    returned.
                - aro_ring (bool): Record the aromatic ring-membered bond.
                - non_aro_ring (bool): Record the non-aromatic ring-membered bond.
                - aro_ring_attached (bool): Record the aromatic ring-attached bond.
                - non_aro_ring_attached (bool):  Record the non-aromatic ring-attached bond.
                - non_ring_attached (bool): Record the non ring-attached bond.

            Returns:
            ----------
                - A list or a dictionary of reactions
        """
        if self.mol is None:
            raise ValueError("No molecule is stored.")

        mol = self.get_mol()
        smiles: str = self.to_smiles()
        reactions: List[List] = []

        MolStereo = count_stereocenters(self.mol, legacy=True)
        if MolStereo["atom_unassigned"] != 0 or MolStereo["bond_unassigned"] != 0:
            warning(f"Molecule {smiles} has undefined stereochemistry.")

        for bond in mol.GetBonds():
            start_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            if single_bond_only:
                # Equivalent as bond.GetBondType() not in (1, 12): <-- The enum code of the RDKit
                if bond.GetBondType() not in [BondType.SINGLE, BondType.AROMATIC]:
                    continue

            if not MolProcessor._eval_ring(bond, aro_ring=aro_ring,
                                           non_aro_ring=non_aro_ring,
                                           aro_ring_attached=aro_ring_attached,
                                           non_aro_ring_attached=non_aro_ring_attached,
                                           non_ring_attached=non_ring_attached):
                continue

            # You may want to check the isotope but I am not so confident on the code to validate it.
            # The latest RDKit (2022.3) said that the MolToSmiles Conversion ignore the isotope.
            # Can reviewer cross-check it later on ?

            # -------------------------------------------------------------------------
            new_mol, error = MolProcessor._break_bond(mol, start_atom, end_atom)

            # Convert the 2 molecules into a SMILES string
            bondType = MolProcessor.get_bond_type(bond)
            if not error:
                try:
                    smi_1A, smi_1B = sorted(MolToSmiles(new_mol).split('.'))
                    smi_2A, smi_2B = MolToSmiles(RemoveHs(MolFromSmiles(smi_1A)), isomericSmiles=True), \
                                     MolToSmiles(RemoveHs(MolFromSmiles(smi_1B)), isomericSmiles=True)

                    # RemoveHs(MolFromSmiles(Smi1A)) and RemoveHs(MolFromSmiles(smi_1B)) is fine I guessed ?
                    # Change the legacy=False if needed more detailed stereo-chemistry.
                    valid: bool = check_stereocenters(MolFromSmiles(smi_2A), legacy=True) and \
                                  check_stereocenters(MolFromSmiles(smi_2B), legacy=True)
                except ValueError:
                    # This is the ring-membered bond -> Only reproduce one radical only
                    smi_1A: str = MolToSmiles(new_mol).split('.')[0]
                    smi_2A: str = MolToSmiles(RemoveHs(MolFromSmiles(smi_1A)), isomericSmiles=True)
                    if strict_radical:
                        smi_1B, smi_2B = 'NONE', 'NONE'
                    else:
                        smi_1B, smi_2B = smi_1A, smi_2A

                    # RemoveHs(MolFromSmiles(Smi1A)) is fine I guessed ?
                    valid: bool = check_stereocenters(MolFromSmiles(smi_2A), legacy=True)
            else:
                smi_2A, smi_2B = 'NONE', 'NONE'
                valid: bool = False
                warning(f' Found invalid bond at index {bond.GetIdx()} of mol {smiles}.')

            reactions.append([smiles, smi_2A, smi_2B, bond.GetIdx(), bondType, valid])
            if reverse_reaction:
                if bond.GetBondType() not in (BondType.AROMATIC, BondType.DATIVE):  # Add more if necessary
                    reactions.append([smiles, smi_2B, smi_2A, bond.GetIdx(), bondType, valid])

        if len(reactions) == 0:
            warning(" No bond is broken or the molecule is not correct.")
            return reactions

        if zero_duplicate:
            reactions = MolProcessor._drop_duplicates(reactions, reverse_reaction=reverse_reaction)

        if to_dict:
            result = defaultdict(list)
            result[self._col[0]] = smiles
            for _, r1, r2, bIdx, bType, stereo in reactions:
                result[self._col[1]].append(r1)
                result[self._col[2]].append(r2)
                result[self._col[3]].append(bIdx)
                result[self._col[4]].append(bType)
                result[self._col[5]].append(stereo)
            return result

        return reactions

    # -------------------------------------------------------------------------
    @staticmethod
    def _eval_ring(bond, aro_ring: bool = False, non_aro_ring: bool = False,
                   aro_ring_attached: bool = True, non_aro_ring_attached: bool = True,
                   non_ring_attached: bool = True) -> bool:
        if bond.IsInRing():
            if not aro_ring and not non_aro_ring:
                return False
            if aro_ring and non_aro_ring:
                return True
            aromatic: bool = bond.GetIsAromatic()
            return (aro_ring and aromatic) or (non_aro_ring and not aromatic)

        if not non_aro_ring_attached and not aro_ring_attached and not non_ring_attached:
            return False
        if non_aro_ring_attached and aro_ring_attached and non_ring_attached:
            return True

        start_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        start_ring: bool = start_atom.IsInRing()
        end_ring: bool = end_atom.IsInRing()
        if non_ring_attached:
            if not start_ring and not end_ring:
                return True

        if aro_ring_attached and non_aro_ring_attached:
            if start_ring or end_ring:
                return True
        elif aro_ring_attached:
            if (start_ring and start_atom.GetIsAromatic()) or (end_ring and end_atom.GetIsAromatic()):
                return True
        elif non_aro_ring_attached:
            if (start_ring and not start_atom.GetIsAromatic()) or (end_ring and not end_atom.GetIsAromatic()):
                return True

        return False

    @staticmethod
    def _break_bond_by_idx(mol: Mol, index: int) -> Tuple[RWMol, bool]:
        bond = mol.GetBondWithIdx(index)
        return MolProcessor._break_bond(mol, bond.GetBeginAtom(), bond.GetEndAtom())

    @staticmethod
    def _break_bond(mol, start_atom, end_atom) -> Tuple[RWMol, bool]:
        # Kekulize the bond
        start_idx: int = start_atom.GetIdx()
        end_idx: int = end_atom.GetIdx()

        rw_mol: RWMol = RWMol(mol)
        rw_mol.RemoveBond(start_idx, end_idx)
        if start_atom.GetIsAromatic() or end_atom.GetIsAromatic():
            # Reduce unnecessary kekulization algorithm: O(atoms + bonds)
            # This applied on both case H-c1cccc1 and c1cccc1-c1ccccc1.
            # If the broken bond is not adjacent to any aromatic state, kekulization would not be called ?
            Kekulize(rw_mol, clearAromaticFlags=True)

        rw_mol.GetAtomWithIdx(start_idx).SetNoImplicit(True)
        rw_mol.GetAtomWithIdx(end_idx).SetNoImplicit(True)
        SanitizeError = False
        try:
            sanitize(rw_mol)
        except RuntimeError:
            SanitizeError = True
        return rw_mol, SanitizeError

    @staticmethod
    def get_bond_type(bond) -> str:
        return '{}-{}'.format(*sorted([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]))

    @staticmethod
    def _drop_duplicates(reactions: List[List], reverse_reaction: bool = False) -> List[List]:
        N: int = len(reactions)
        mask: List[bool] = [False] * N
        for i in range(0, N):
            if mask[i]:
                continue
            smi1, smi2 = reactions[i][1], reactions[i][2]
            if smi1 == 'NONE' or smi2 == 'NONE':
                continue
            for j in range(i + 1, N):
                if not mask[j]:
                    if smi1 == reactions[j][1] and smi2 == reactions[j][2]:
                        mask[j] = True
                    elif not reverse_reaction and smi1 == reactions[j][2] and smi2 == reactions[j][1]:
                        mask[j] = True
        return [reactions[idx] for idx in range(N) if not mask[idx]]

    # -------------------------------------------------------------------------
    def add_mol(self, mol: Union[Mol, str], Hs: bool = True, *args, **kwargs) -> None:
        """
            This method is to store a molecule.

            Arguments:
            ----------
                - mol (Mol): A valid RDKit molecule. If it is a string, it will be converted to a molecule.
                    and require extra testing for validity.
                - Hs (bool): If mol is a string, decide whether to add hydrogens.
                - *args, **kwargs (bool): Optional parameters to sanitize a molecule.
        """
        if self.mol is not None:
            warning(' A molecule is already stored, calling this method would overwrite it.')

        if isinstance(mol, str):
            mol = smiles_to_mol(mol, Hs, *args, **kwargs)
        self.mol = mol

    def clear_computed_props(self) -> None:
        if self.is_mol_stored(skip_warning=False):
            self.mol.ClearComputedProps(True)
        return None

    def get_mol(self) -> Mol:
        self.is_mol_stored(skip_warning=False)
        return self.mol

    def to_smiles(self, keepPossibleHs: bool = False) -> str:
        return MolToSmiles(self.mol) if keepPossibleHs else MolToSmiles(RemoveHs(self.mol))

    def canon_smiles(self, useChiral: bool) -> str:
        if self.mol is None:
            raise ValueError("No molecule is stored.")
        return CanonSmiles(self.mol, useChiral=useChiral)

    def clear_mol(self) -> None:
        self.mol = None

    def get_label(self) -> List[str]:
        return self._col

    def is_mol_stored(self, skip_warning: bool = False) -> bool:
        if not skip_warning and self.mol is None:
            warning(" No molecule is stored.")
        return self.mol is not None
