import os
import warnings

import nfp
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import RDLogger

from alfabet.drawing import draw_bde
from alfabet.fragment import fragment_iterator

RDLogger.DisableLog('rdApp.*')

currdir = os.path.dirname(os.path.abspath(__file__))


def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        nfp.get_ring_size(atom, max_size=6),
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

    btype = str(bond.GetBondType())
    ring = 'R{}'.format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''

    return " ".join([atoms, btype, ring]).strip()


preprocessor = nfp.SmilesPreprocessor(
    atom_features=atom_featurizer, bond_features=bond_featurizer)

preprocessor.from_json(os.path.join(currdir, 'model_files/preprocessor.json'))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model = tf.keras.models.load_model(
        os.path.join(currdir, 'model_files/best_model.hdf5'),
        custom_objects=nfp.custom_objects,
        compile=False)

bde_dft = pd.read_csv(os.path.join(
    currdir, 'model_files/20201012_bonds_for_neighbors.csv.gz'))


def check_input(smiles):
    """ Check the given SMILES to ensure it's present in the model's
    preprocessor dictionary.

    Returns:
    (is_outlier, missing_atom, missing_bond)

    """

    iinput = preprocessor.construct_feature_matrices(smiles, train=False)

    missing_bond = np.array(
        list(set(iinput['bond_indices'][iinput['bond'] == 1])))
    missing_atom = np.arange(iinput['n_atom'])[iinput['atom'] == 1]

    is_outlier = (missing_bond.size != 0) | (missing_atom.size != 0)

    return is_outlier, missing_atom, missing_bond


def predict_bdes(smiles, draw=False):
    # Break bonds and get corresponding bond indexes where predictions are
    # valid
    frag_df = pd.DataFrame(fragment_iterator(smiles))

    ds = tf.data.Dataset.from_generator(
        lambda: (preprocessor.construct_feature_matrices(item, train=False)
                 for item in (smiles,)),
        output_types=preprocessor.output_types,
        output_shapes=preprocessor.output_shapes).batch(batch_size=1)

    bde_pred, bdfe_pred = model.predict(ds)

    # Reindex predictions to fragment dataframe
    frag_df['bde_pred'] = pd.Series(bde_pred.squeeze()) \
        .reindex(frag_df.bond_index).reset_index(drop=True)
    frag_df['bdfe_pred'] = pd.Series(bdfe_pred.squeeze()) \
        .reindex(frag_df.bond_index).reset_index(drop=True)

    # Add DFT calculated bdes
    frag_df = frag_df.merge(bde_dft[['molecule', 'bond_index', 'bde', 'bdfe', 'set']],
                            on=['molecule', 'bond_index'], how='left')

    # Drop duplicate entries and sort from weakest to strongest
    frag_df = frag_df.sort_values('bde_pred').drop_duplicates(
        ['fragment1', 'fragment2']).reset_index(drop=True)

    # Draw SVGs
    if draw:
        frag_df['svg'] = frag_df.apply(
            lambda x: draw_bde(x.molecule, x.bond_index), 1)

    frag_df['has_dft_bde'] = frag_df.bde.notna()

    return frag_df
