import os
from itertools import chain
    
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import nfp

from alfabet.fragment import fragment_iterator
from alfabet.preprocess_inputs import atom_featurizer, bond_featurizer

currdir = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(currdir, 'model_files/best_model.hdf5'),
    custom_objects=nfp.custom_objects)


# Load the preprocessor from the saved json configuration
preprocessor = nfp.SmilesPreprocessor(atom_features=atom_featurizer,
                                      bond_features=bond_featurizer)
preprocessor.from_json(os.path.join(currdir, 'model_files/preprocessor.json'))


def check_valid(iinput):
    """ Check the given SMILES to ensure it's present in the model's
    preprocessor dictionary.

    Returns:
    (is_outlier, missing_atom, missing_bond)

    """

    missing_bond = np.array(
        list(set(iinput['bond_indices'][np.array(iinput['bond'] == 1)].numpy().tolist())))
    missing_atom = np.arange(iinput['n_atom'])[np.array(iinput['atom'] == 1).squeeze()]
    is_outlier = bool((missing_bond.size != 0) | (missing_atom.size != 0))

    return not is_outlier


def predict(smiles_list, drop_duplicates=True, verbose=True):
    """Predict the BDEs of each bond in a list of molecules.

    Parameters
    ----------
    smiles_list : list
        List of SMILES strings for each molecule
    drop_duplicates : bool, optional
        Whether to drop duplicate bonds (those with the same resulting radicals)
    verbose : bool, optional
        Whether to show a progress bar

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

    frag_df = pd.DataFrame(chain(*(fragment_iterator(smiles)
                                   for smiles in smiles_list)))

    def prediction_generator(smiles_iterator):
    
        dataset = tf.data.Dataset.from_generator(
            lambda: (preprocessor.construct_feature_matrices(item, train=False)
                     for item in smiles_iterator),
            output_types=preprocessor.output_types,
            output_shapes=preprocessor.output_shapes).batch(1)
        
        for molecule, inputs in tqdm(zip(smiles_iterator, dataset), 
                                     disable=not verbose):
            out = model.predict_on_batch(inputs)
            df = pd.DataFrame(out[0, :inputs['n_bond'][0], 0], columns=['BDE'])
            df['molecule'] = molecule
            df.index.name = 'bond_index'
            df.reset_index(inplace=True)
            
            df['is_valid'] = check_valid(inputs)
            
            yield df

    bde_df = pd.concat(prediction_generator(smiles_list))

    pred_df = frag_df.merge(bde_df, on=['molecule', 'bond_index'],
                            how='left')

    if drop_duplicates:
        pred_df = pred_df.drop_duplicates([
            'fragment1', 'fragment2']).reset_index(drop=True)

    return pred_df
