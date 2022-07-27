try:
    import flask
    from flask import Markup
except ImportError:
    flask = None

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs



def draw_bde(smiles: str, bond_index: int, figwidth: int = 200):
    mol = MolFromSmiles(smiles)
    if not isinstance(bond_index, int):
        bond_index = int(bond_index)

    if mol.GetNumAtoms() > 20:
        figwidth = 300
    if mol.GetNumAtoms() > 40:
        figwidth = 400

    if bond_index >= mol.GetNumBonds():
        molH = AddHs(mol)
        if bond_index >= molH.GetNumBonds():
            raise RuntimeError(
                f"Fewer than {bond_index} bonds in {smiles}: "
                f"{molH.GetNumBonds()} total bonds."
            )
        
        bond = molH.GetBondWithIdx(bond_index)
        mol = Chem.AddHs(mol, onlyOnAtoms=[bond.GetBeginAtomIdx()])
        bond_index = mol.GetNumBonds() - 1

    if not mol.GetNumConformers():
        Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(figwidth, figwidth)
    drawer.drawOptions().fixedBondLength = 30
    drawer.drawOptions().highlightBondWidthMultiplier = 20

    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=[bond_index])
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return Markup(svg) if flask else svg

def draw_mol_outlier(smiles, missing_atoms, missing_bonds, figsize=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if not isinstance(bond_index, int):
        bond_index = int(bond_index)
    
    missing_bonds_adjusted = []
    for bond_index in missing_bonds:
        if bond_index >= mol.GetNumBonds():
            molH = Chem.AddHs(mol)
            bond = molH.GetBondWithIdx(bond_index)
            mol = AddHs(mol, onlyOnAtoms=[bond.GetBeginAtomIdx()])
            bond_index = mol.GetNumBonds() - 1
        missing_bonds_adjusted.append(bond_index)

    if not mol.GetNumConformers():
        # Is mol.Compute2DCoords() is a classmethod alias of rdDepictor.Compute2DCoords() ?
        Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)
    drawer.SetFontSize(0.6)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=[int(index) for index in missing_atoms],
        highlightBonds=missing_bonds_adjusted,
    )

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return Markup(svg) if flask else svg



def draw_mol(smiles, figsize=(300, 300)):
    mol = MolFromSmiles(smiles)
    Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)
    drawer.SetFontSize(0.6)
    drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return Markup(svg) if flask else svg
