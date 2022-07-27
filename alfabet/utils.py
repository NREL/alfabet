#
#  Copyright (C) 2022
#   @@ All Rights Reserved @@
#
#  This file is part of the ALFABET program.
#  The contents are covered by the terms of the license
#  which is included in the file LICENSE, found at the root
#  of the source tree.
#

# Outsize contributor: Ichiru Take
# This module is to support only the RDKit molecule processing.

from collections import defaultdict
from logging import warning
from typing import Dict, Optional, Union, List, Tuple

from rdkit.Chem import CanonSmiles, FindMolChiralCenters
from rdkit.Chem.rdchem import Mol, RWMol, BondType, BondStereo
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import (AddHs, Kekulize, RemoveHs, SanitizeMol, AssignStereochemistry, 
                                 FindPotentialStereoBonds, RemoveHs)

def Sanitize(mol: Mol, *args, **kwargs) -> Mol:
    if mol is not None:
        try:
            _ = SanitizeMol(mol, *args, **kwargs)
        except (ValueError, RuntimeError):
            pass
    return mol


def SmilesToMol(smiles: str, Hs: bool = True, *args, **kwargs) -> Mol:
    if isinstance(smiles, str):
        mol = MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES: {}. Please check your SMILES notation.".format(smiles))
    else:
        raise TypeError("SMILES must be a valid python string.")
    try:
        if Hs:
            mol = AddHs(mol)
        _ = SanitizeMol(mol, *args, **kwargs)
    except (ValueError, RuntimeError):
        raise ValueError("Invalid molecule: {}. Please check your SMILES notation.".format(smiles))
    return mol               


def FindStereoAtomsBonds(mol: Mol, NeedAssignStereochemistry: bool = True) -> None:
    if NeedAssignStereochemistry:
        AssignStereochemistry(mol, force=False, cleanIt=True, flagPossibleStereoCenters=True)
    FindPotentialStereoBonds(mol, cleanIt=True)

def StereoCenters(mol: Mol, legacy: bool = True) -> Dict:
    FindStereoAtomsBonds(mol, NeedAssignStereochemistry=legacy)
    
    stereocenters: List = FindMolChiralCenters(mol, includeUnassigned=True)
    
    stereobonds = [bond for bond in mol.GetBonds() if bond.GetStereo() is not BondStereo.STEREONONE]
    atom_assigned = len([center for center in stereocenters if center[1] != "?"])
    atom_unassigned = len(stereocenters) - atom_assigned
    
    bond_assigned = len([bond for bond in stereobonds if bond.GetStereo() is not BondStereo.STEREOANY])
    bond_unassigned = len(stereobonds) - bond_assigned
    
    return {"atom_assigned": atom_assigned, "atom_unassigned": atom_unassigned,
            "bond_assigned": bond_assigned, "bond_unassigned": bond_unassigned}
    
def CountStereoCenters(mol: Mol, legacy: bool = True) -> bool:
    """ 
        Check the given SMILES string to determine whether accurate enthalpies 
        can be calculated with the given stereochem information.
    """
    stereocenters = StereoCenters(mol, legacy=legacy)
    if stereocenters["bond_unassigned"] > 0:
        return False

    max_unassigned = 1 if stereocenters["atom_assigned"] == 0 else 1        # I don't get it here.
    return stereocenters["atom_unassigned"] <= max_unassigned


class MolProcessor:
    # __slots__ = ('mol', )                     # For future Python (current minimal is 3.6 ?)          
    
    def __init__(self, mol: Union[Mol, str]) -> None:
        if isinstance(mol, str):
            mol = SmilesToMol(mol)
        self.mol: Mol = mol
        self._col = ['molecule', 'fragment1', 'fragment2', 'bond_index', 'bond_type', 'is_valid_stereo']
    
    # -------------------------------------------------------------------------
    def GetReactions(self, ToDict: bool, ReverseReaction: bool = False, SingleBondOnly: bool = True, 
                     StrictRadical: bool = True, ZeroDuplicate: bool = True, AroRing: bool = False, 
                     NonAroRing: bool = False, AroRingAttached: bool = True, 
                     NonAroRingAttached: bool = True, NonRingAttached: bool = True) -> Union[Dict, List]:
        
        """
            This method is to get all bonds inside molecule (in SMILES form) into a series of bonds.
            Arguments:
            ----------
                - ToDict (bool): If True, return a dictionary.
                - ReverseReaction (bool): If True and the bond is not a ring-membered or the directed bond,
                    an extra copy is also returned.
                - SingleBondOnly (bool): If True, only the single bond and the aromatic bond is returned.
                - ZeroDuplicate (bool): If True, all duplicated bonds are removed.
                - StrictRadical (bool): If True, whenever enable arg::AroRing or arg::NonAroRing.
                    If the molecule contained cycles, only one radical (the second column) is returned 
                    (The second value is always a string::NONE). If False, both radicals are identically 
                    returned.
                - AromaticRing (bool): Record the aromatic ring-membered bond.
                - NonAromaticRing (bool): Record the non-aromatic ring-membered bond.
                - AromaticRingAttached (bool): Record the aromatic ring-attached bond.
                - NonAromaticRingAttached (bool):  Record the non-aromatic ring-attached bond.
                - NonRingAttached (bool): Record the non ring-attached bond.
            
            Returns:
            ----------
                - A list or a dictionary of reactions 
        """
        if self.mol is None:
            raise ValueError("No molecule is stored.")
        
        mol = self.GetMol()
        smiles: str = self.ToSmiles()
        reactions: List[List] = []
        
        for bond in mol.GetBonds():
            StartAtom = bond.GetBeginAtom()
            EndAtom = bond.GetEndAtom()
            
            if SingleBondOnly:
                # Equivalent as bond.GetBondType() not in (1, 12): <-- The enum code of the RDKit
                if bond.GetBondType() not in [BondType.SINGLE, BondType.AROMATIC]:
                    continue
            
            if not MolProcessor._TestBondWithRingType_(bond, AroRing, NonAroRing, AroRingAttached, 
                                                       NonAroRingAttached, NonRingAttached):
                continue
                
            # You may want to check the isotope but I am not so confident on the code to validate it.
            # The latest RDKit (2022.3) said that the MolToSmiles Conversion ignore the isotope.
            # Can reviewer cross-check it later on ?
            
            # -------------------------------------------------------------------------
            new_mol, error = MolProcessor._BreakBond_(mol, StartAtom, EndAtom)
            
            # Convert the 2 molecules into a SMILES string
            bondType = '{}-{}'.format(*sorted([StartAtom.GetSymbol(), EndAtom.GetSymbol()]))
            if not error:
                try:
                    Smi1A, Smi1B = sorted(MolToSmiles(new_mol).split('.'))
                    Smi2A, Smi2B = MolToSmiles(RemoveHs(MolFromSmiles(Smi1A)), isomericSmiles=True), \
                                   MolToSmiles(RemoveHs(MolFromSmiles(Smi1B)), isomericSmiles=True)
                    
                    # RemoveHs(MolFromSmiles(Smi1A)) and RemoveHs(MolFromSmiles(Smi1B)) is fine I guessed ?
                    # Change the legacy=False if needed more detailed stereo-chemistry.
                    valid: bool = CountStereoCenters(MolFromSmiles(Smi2A), legacy=True) and \
                                  CountStereoCenters(MolFromSmiles(Smi2B), legacy=True)
                except ValueError:
                    # This is the ring-membered bond -> Only reproduce one radical only
                    Smi1A: str = MolToSmiles(new_mol).split('.')[0]
                    Smi2A: str = MolToSmiles(RemoveHs(MolFromSmiles(Smi1A)), isomericSmiles=True)
                    if StrictRadical:
                        Smi1B, Smi2B = 'NONE', 'NONE'
                    else:
                        Smi1B, Smi2B = Smi1A, Smi2A
                        
                    # RemoveHs(MolFromSmiles(Smi1A)) is fine I guessed ?
                    valid: bool = CountStereoCenters(MolFromSmiles(Smi2A), legacy=True)
            else:
                Smi2A, Smi2B = 'NONE', 'NONE'
                valid: bool = False
                warning(f' Found invalid bond at index {bond.GetIdx()} of mol {smiles}.')

            reactions.append([smiles, Smi2A, Smi2B, bond.GetIdx(), bondType, valid])
            if ReverseReaction:
                if bond.GetBondType() not in (BondType.AROMATIC, BondType.DATIVE):      # Add more if necessary
                    reactions.append([smiles, Smi2B, Smi2A, bond.GetIdx(), bondType, valid])

        if len(reactions) == 0:
            raise ValueError("No bond is broken or the molecule is not correct.")
        
        if ZeroDuplicate:
            N: int = len(reactions)
            mask: List[bool] = [False] * N
            for row in range(0, N):
                if mask[row]:
                    continue
                smi1, smi2 = reactions[row][3], reactions[row][4]
                if smi1 == 'NONE' or smi2 == 'NONE':
                    continue
                for next_row in range(row + 1, N):
                    if not mask[next_row]:
                        if smi1 == reactions[next_row][3] and smi2 == reactions[next_row][4]:
                            mask[next_row] = True
                        elif not ReverseReaction and (smi1 == reactions[next_row][4] and smi2 == reactions[next_row][3]):
                            mask[next_row] = True
            temp = [reactions[idx] for idx in range(N) if not mask[idx]]
            reactions = temp
        
        if ToDict:
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
    def _TestBondWithRingType_(bond, AroRing: bool = False, NonAroRing: bool = False,
                               NonAroRingAttached: bool = True, AroRingAttached: bool = True,
                               NonRingAttached: bool = True) -> bool:
        if bond.IsInRing():
            if not AroRing and not NonAroRing:
                return False
            if AroRing and NonAroRing:
                return True
            aromatic: bool = bond.GetIsAromatic()
            return (AroRing and aromatic) or (NonAroRing and not aromatic)

        if not NonAroRingAttached and not AroRingAttached and not NonRingAttached:
            return False
        if NonAroRingAttached and AroRingAttached and NonRingAttached:
            return True

        StartAtom = bond.GetBeginAtom()
        EndAtom = bond.GetEndAtom()

        StartRing: bool = StartAtom.IsInRing()
        EndRing: bool = EndAtom.IsInRing()
        if NonRingAttached:
            if not StartRing and not EndRing:
                return True

        if AroRingAttached and NonAroRingAttached:
            if StartRing or EndRing:
                return True
        elif AroRingAttached:
            if (StartRing and StartAtom.GetIsAromatic()) or (EndRing and EndAtom.GetIsAromatic()):
                return True
        elif NonAroRingAttached:
            if (StartRing and not StartAtom.GetIsAromatic()) or (EndRing and not EndAtom.GetIsAromatic()):
                return True

        return False
    
    @staticmethod
    def _BreakBondWithIdx_(mol: Mol, index: int) -> Tuple[RWMol, bool]:
        bond = mol.GetBondWithIdx(index)
        return MolProcessor._BreakBond_(mol, bond.GetBeginAtom(), bond.GetEndAtom())

    @staticmethod
    def _BreakBond_(mol, StartAtom, EndAtom) -> Tuple[RWMol, bool]:
        StartIdx: int = StartAtom.GetIdx()
        EndIdx: int = EndAtom.GetIdx()

        rw_mol: RWMol = RWMol(mol)
        rw_mol.RemoveBond(StartIdx, EndIdx)
        rw_mol.GetAtomWithIdx(StartIdx).SetNoImplicit(True)
        rw_mol.GetAtomWithIdx(EndIdx).SetNoImplicit(True)
        SanitizeError = False
        try:
            Sanitize(rw_mol)
        except RuntimeError:
            try:
                rw_mol: RWMol = RWMol(mol)
                Kekulize(rw_mol, clearAromaticFlags=True)
                rw_mol.RemoveBond(StartIdx, EndIdx)
                rw_mol.GetAtomWithIdx(StartIdx).SetNoImplicit(True)
                rw_mol.GetAtomWithIdx(EndIdx).SetNoImplicit(True)
                Sanitize(rw_mol)
            except RuntimeError:
                SanitizeError = True
        return rw_mol, SanitizeError
    
    # -------------------------------------------------------------------------
    def AddMol(self, mol: Union[Mol, str], Hs: bool = True, *args, **kwargs) -> None:
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
            mol = SmilesToMol(mol, Hs=Hs, *args, **kwargs)          
        self.mol = mol
    
    def ClearComputedProps(self) -> None:
        if self.IsMolStored(skip_warning=False):
            self.mol.ClearComputedProps(True)
        return None
    
    def GetMol(self) -> Mol:
        self.IsMolStored(skip_warning=False)
        return self.mol
    
    def ToSmiles(self, keepPossibleHs: bool = False) -> str:
        return MolToSmiles(self.mol) if keepPossibleHs else MolToSmiles(RemoveHs(self.mol))
    
    def CanonSmiles(self, useChiral: bool) -> str:
        if self.mol is None:
            raise ValueError("No molecule is stored.")
        return CanonSmiles(self.mol, useChiral=useChiral)
    
    def ClearMol(self) -> None:
        self.mol = None
    
    def GetLabel(self) -> List[str]:
        return self._col

    def IsMolStored(self, skip_warning: bool = False) -> bool:
        if not skip_warning and self.mol is None:
            warning(" No molecule is stored.")
        return self.mol is not None
    
    # -------------------------------------------------------------------------
    