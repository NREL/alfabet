import nfp
import numpy as np
from nfp.preprocessing.features import get_ring_size
from pooch import retrieve

from alfabet import _model_files_baseurl


def atom_featurizer(atom):
    """Return an integer hash representing the atom type"""

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


def bond_featurizer(bond, flipped=False):
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
        )
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()))
        )

    btype = str((bond.GetBondType(), bond.GetIsConjugated()))
    ring = "R{}".format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ""

    return " ".join([atoms, btype, ring]).strip()


preprocessor = nfp.SmilesBondIndexPreprocessor(
    atom_features=atom_featurizer,
    bond_features=bond_featurizer,
    explicit_hs=True,
    output_dtype="int64",
)

preprocessor.from_json(
    retrieve(
        _model_files_baseurl + "preprocessor.json",
        known_hash="412d15ca4d0e8b5030e9b497f566566922818ff355b8ee677a91dd23696878ac",
    )
)


def get_features(smiles: str, pad: bool = False, **kwargs) -> dict:
    """Run the preprocessor on the given SMILES string

    Args:
        smiles (str): the input molecule
        pad (bool, optional): whether to left-pad the inputs with zeros in preparation
            for tf-serving's padding behavior. Defaults to False.

    Returns:
        dict: numpy array inputs with atom, bond, connectivity, and bond_indicies.
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
