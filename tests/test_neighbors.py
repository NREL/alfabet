import rdkit.Chem

from alfabet.neighbors import find_neighbor_bonds


def test_find_neighbor_bonds():
    neighbor_df = find_neighbor_bonds('CC', 0)
    assert neighbor_df.distance.min() < 1E-3  # bond should be in the database

    for _, row in neighbor_df.iterrows():
        mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(row.molecule))
        bond = mol.GetBondWithIdx(row.bond_index)
        assert bond.GetEndAtom().GetSymbol() == 'C'
        assert bond.GetBeginAtom().GetSymbol() == 'C'
        assert bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE