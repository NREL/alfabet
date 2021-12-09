import nfp
from nfp.preprocessing.features import get_ring_size
from pooch import retrieve

from alfabet import _model_files_baseurl


def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond, flipped=False):
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))

    btype = str((bond.GetBondType(),
                 bond.GetIsConjugated()))
    ring = 'R{}'.format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''

    return " ".join([atoms, btype, ring]).strip()


preprocessor = nfp.SmilesBondIndexPreprocessor(
    atom_features=atom_featurizer,
    bond_features=bond_featurizer,
    explicit_hs=True,
    output_dtype='int64'
)

preprocessor.from_json(retrieve(
    _model_files_baseurl + 'preprocessor.json',
    known_hash='412d15ca4d0e8b5030e9b497f566566922818ff355b8ee677a91dd23696878ac'))


def get_features(smiles: str) -> dict:
    return preprocessor(smiles, train=False)