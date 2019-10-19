import numpy as np

from rdkit.Chem import MolFromSmiles, AddHs
from nfp.preprocessing import SmilesPreprocessor
from nfp.preprocessing import GraphSequence
from nfp.preprocessing import get_ring_size
    
class SmilesBondIndexPreprocessor(SmilesPreprocessor):
    def construct_feature_matrices(self, smiles):
        """ construct a molecule from the given smiles string and return atom
        and bond classes.
        Returns
        dict with entries
        'n_atom' : number of atoms in the molecule
        'n_bond' : number of bonds in the molecule 
        'atom' : (n_atom,) length list of atom classes
        'bond' : (n_bond,) list of bond classes
        'connectivity' : (n_bond, 2) array of source atom, target atom pairs.
        """

        mol = MolFromSmiles(smiles)
        if self.explicit_hs:
            mol = AddHs(mol)

        n_atom = len(mol.GetAtoms())
        n_bond = 2 * len(mol.GetBonds())

        # If its an isolated atom, add a self-link
        if n_bond == 0:
            n_bond = 1
        
        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        bond_indices = np.zeros(n_bond, dtype='int')        
        connectivity = np.zeros((n_bond, 2), dtype='int')

        bond_index = 0

        atom_seq = mol.GetAtoms()
        atoms = [atom_seq[i] for i in range(n_atom)]

        for n, atom in enumerate(atoms):

            # Atom Classes
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))

            start_index = atom.GetIdx()

            for bond in atom.GetBonds():
                # Is the bond pointing at the target atom
                rev = bond.GetBeginAtomIdx() != start_index

                # Bond Classes
                bond_feature_matrix[bond_index] = self.bond_tokenizer(
                    self.bond_features(bond, flipped=rev))
                
                bond_indices[bond_index] = bond.GetIdx()

                # Connectivity
                if not rev:  # Original direction
                    connectivity[bond_index, 0] = bond.GetBeginAtomIdx()
                    connectivity[bond_index, 1] = bond.GetEndAtomIdx()

                else:  # Reversed
                    connectivity[bond_index, 0] = bond.GetEndAtomIdx()
                    connectivity[bond_index, 1] = bond.GetBeginAtomIdx()

                bond_index += 1


        return {
            'n_atom': n_atom,
            'n_bond': n_bond,
            'bond_indices': bond_indices,
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'connectivity': connectivity,
        }

    
class ConcatGraphSequence(GraphSequence):
    
    def process_data(self, batch_data):
        return batch_data
    
    def index_y(self, indices):
        """Here we want to stack self._y for each minibatch, such that
        output target is n_bonds in length"""
        
        return self._concat([self._y[i] for i in indices])
    
    def on_epoch_end(self):
        if self.shuffle:
            
            np.random.seed(self.seed)
            self.seed += 1
            
            indices = np.arange(0, len(self._inputs))
            
            np.random.shuffle(indices)
            self._inputs = [self._inputs[i] for i in indices]
            if self._y is not None:
                self._y = [self._y[i] for i in indices]
                
                
def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_features(bond, flipped=False):
   
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()
