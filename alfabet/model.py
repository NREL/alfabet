import pandas as pd

from alfabet.fragment import get_fragments
from alfabet.prediction import predict_bdes, validate_inputs
from alfabet.preprocessor import get_features


def predict(smiles_list, drop_duplicates=True):
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

    inputs = {smiles: get_features(smiles) for smiles in smiles_list}
    fragments = {smiles: get_fragments(smiles) for smiles in smiles_list}

    pred_df = pd.concat([
        predict_bdes(fragments[smiles],
                     inputs[smiles],
                     drop_duplicates=drop_duplicates)
        for smiles in smiles_list])

    is_valid = pd.Series({smiles: not validate_inputs(input_)[0] for smiles, input_
                          in inputs.items()}, name='is_valid')

    pred_df = pred_df.merge(is_valid, left_on='molecule', right_index=True)

    return pred_df.sort_values(['molecule', 'bond_index'])
