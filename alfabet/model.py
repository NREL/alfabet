import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from alfabet.fragment import get_fragments
from joblib import Parallel, delayed

with warnings.catch_warnings():
    # Put these imports inside a catch warnings context to supress a number of
    # TF 2.0 warnings and numpy version mismatch warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from keras.models import load_model
    from alfabet.preprocessor_utils import ConcatGraphSequence
    from nfp import custom_layers

currdir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(currdir, 'model_files/preprocessor.p'), 'rb') as f:
    from alfabet import preprocessor_utils
    sys.modules['preprocessor_utils'] = preprocessor_utils
    preprocessor = pickle.load(f)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # These supress the exploding gradient warnings

    model = load_model(
        os.path.join(currdir, 'model_files/best_model.hdf5'),
        custom_objects=custom_layers)
    model._make_predict_function()

def check_valid(iinput):
    """ Check the given SMILES to ensure it's present in the model's
    preprocessor dictionary.

    Returns:
    (is_outlier, missing_atom, missing_bond)

    """

    missing_bond = np.array(
        list(set(iinput['bond_indices'][iinput['bond'] == 1])))
    missing_atom = np.arange(iinput['n_atom'])[iinput['atom'] == 1]
    is_outlier = bool((missing_bond.size != 0) | (missing_atom.size != 0))

    return not is_outlier


def inputs_to_dataframe(smiles, inputs):
    molecule = np.repeat(np.array(smiles), np.stack([iinput['n_bond'] for iinput in inputs]))
    bond_index = np.concatenate([iinput['bond_indices'] for iinput in inputs])
    input_df = pd.DataFrame(np.vstack([molecule, bond_index]).T,
                            columns=['molecule', 'bond_index'])
    input_df['bond_index'] = input_df.bond_index.astype('int64')

    return input_df


def predict(smiles_list, batch_size=128, drop_duplicates=True, verbose=True,
            n_jobs=-1):
    """Predict the BDEs of each bond in a list of molecules.

    Parameters
    ----------
    smiles_list : list
        List of SMILES strings for each molecule
    batch_size : int, optional
        Batch size of molecules (when running on a GPU)
    drop_duplicates : bool, optional
        Whether to drop duplicate bonds (those with the same resulting radicals)
    verbose : bool, optional
        Whether to show a progress bar
    n_jobs : int, optional
        jobs parameter for joblib. Default to use all cores.

    Returns
    -------
    pd.DataFrame
    dataframe of prediction results with columns:

        molecule - SMILES of parent
        bond_index - integer corresponding to given bond (of mol with explicit
                     H's)
        bond_type - elements of start and end atom types
        fragment1 - SMILES of one radical product
        fragment2 - SMILES of second radical product
        delta_assigned_stereo - # of assigned stereocenters created or destroyed
        delta_unassigned_stereo - # of unassigned stereocenters changed
        bde_pred - predicted BDE (in kcal/mol)
        is_valid - whether the starting molecule is present in the model's
                   domain of validity
    """

    # Process the smiles list into graph representations
    inputs = preprocessor.predict(smiles_list, verbose=verbose)

    # Predict the inputs with the neural network
    pred = model.predict_generator(
        ConcatGraphSequence(inputs, batch_size=batch_size, shuffle=False),
        verbose=int(verbose))

    bde_df = inputs_to_dataframe(smiles_list, inputs)
    bde_df['bde_pred'] = pred
    bde_df = bde_df.groupby(['molecule',
                             'bond_index']).bde_pred.mean().reset_index()

    # Check mols for preprocessor class presence in training data
    valid_mols = pd.Series([check_valid(iinput) for iinput in inputs],
                           dtype=bool, index=smiles_list, name='is_valid')
    bde_df = bde_df.merge(valid_mols, left_on='molecule', right_index=True,
                          how='left')

    # Seperately fragment the molecules to find their valid bonds
    frag_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(get_fragments)(smiles) for smiles in smiles_list)
    frag_df = pd.concat(frag_results)

    pred_df = frag_df.merge(bde_df, on=['molecule', 'bond_index'],
                            how='left')

    if drop_duplicates:
        pred_df = pred_df.drop_duplicates([
            'fragment1', 'fragment2']).reset_index()

    return pred_df.drop('index', 1)
