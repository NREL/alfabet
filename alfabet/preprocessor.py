import nfp
import numpy as np
from nfp.preprocessing.features import get_ring_size
from pooch import retrieve

from alfabet import MODEL_CONFIG


def atom_featurizer(atom):
    """ 
        Return a merged string representation of atom features. This would be useful 
        to store the simple representation of an atom into a word dictionary for
        the Embedding layer in Keras/Tensorflow.
    """

    return str(
        (
            atom.GetSymbol(),
            atom.GetNumRadicalElectrons(),
            atom.GetFormalCharge(),
            atom.GetChiralTag(),
            atom.GetIsAromatic(),
            get_ring_size(atom, max_size=6),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
    )


def bond_featurizer(bond, flipped: bool = False) -> str:
    """ 
        Return a merged string representation of bond features. This would be useful 
        to store the simple representation of an bond into a word dictionary for
        the Embedding layer in Keras/Tensorflow.
    """
    if not flipped:
        atoms = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
    else:
        atoms = [bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()]
    atoms = f'{atoms[0]}-{atoms[1]}'

    btype = str((bond.GetBondType(), bond.GetIsConjugated()))
    ring = f"R{get_ring_size(bond, max_size=6)}" if bond.IsInRing() else ""

    return " ".join([atoms, btype, ring]).strip()


preprocessor = nfp.SmilesBondIndexPreprocessor(
    atom_features=atom_featurizer,
    bond_features=bond_featurizer,
    explicit_hs=True,
    # Would it better to be 'uint32' for better memory usage on x32 CPU ?
    output_dtype="int64",
)

preprocessor.from_json(
    retrieve(
        MODEL_CONFIG['base_url'] + MODEL_CONFIG['preprocessor_name'],
        known_hash="412d15ca4d0e8b5030e9b497f566566922818ff355b8ee677a91dd23696878ac",
    )
)


def get_features(smiles: str, pad: bool = False, **kwargs) -> dict:
    """
        Run the pre-processor on the given SMILES string.
        
        Arguments:
        ----------
            - smiles (bool): The input molecule.
            - pad (bool): Optional. Whether to left-pad the input with zeros in preparation 
                for tf-serving's padding behavior. Defaults to False.

        Returns:
        ----------
            - A numpy array inputs with atom, bond, connectivity, and bond_indicies.
    """
    features = preprocessor(smiles, train=False, **kwargs)
    if not pad:
        return features

    # We have to offset the connectivity array by 1 since we're adding a phantom atom
    # at position 0
    features["connectivity"] += 1

    def pad_value(val):
        return np.pad(val, [(1, 0)] + [(0, 0)] * (val.ndim - 1))

    return {key: pad_value(val) for key, val in features.items()}
