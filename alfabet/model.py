from typing import List

import pandas as pd
from nfp.frameworks import tf
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs

from alfabet.fragment import MolProcessor
from alfabet.prediction import bde_dft, model, validate_inputs
from alfabet.preprocessor import get_features, preprocessor


def NumBonds(smiles: str) -> int:
    return AddHs(MolFromSmiles(smiles)).GetNumBonds()


def GetMaxBonds(smiles_list: List[str]) -> int:
    return max(NumBonds(smiles) for smiles in smiles_list)


def predict(smiles_list: List[str], drop_duplicates: bool = True,
            batch_size: int = 1, verbose: bool = False):
    """
        Predict the BDEs / BDFEs of each bond in a list of molecules.

        Parameters:
        ----------
            - smiles_list (list) : A list of SMILES strings for each molecule
            - drop_duplicates (bool, optional): Whether to drop duplicate bonds or reactions
                within a same molecule. Default to True.
            - batch_size (int, optional): The size of each batch to passed through model. Varied
                depending on the Batch Normalization layer. Default to 1
            - verbose (bool, optional): Whether to display some information for checking.
                Default to False.

        Returns:
        -------
        A pandas DataFrame of prediction results with columns:
        - molecule: The valid SMILES (in arg::smiles_list).
        - fragment1: The SMILES of one radical product.
        - fragment2: The SMILES of one radical product.
        - bond_index: A integer corresponding to given bond (of mol with explicit Hs).
        - bond_type: The elements name of start and end atom types (sorted by name).
        - is_valid_stereo: Determine whether the molecule is valid stoichiometry
            (equal atom types).
        - bde_pred: The predicted BDE (in kcal/mol).
        - bdfe_pred: The predicted BDFE (in kcal/mol).
        - is_valid: Whether the starting molecule is present in the model's
                   domain of validity.
        - bde: The reference BDE if found in the training set (in kcal/mol).
            Result to NaN if not found.
        - bdfe: The reference BDE if found in the training set (in kcal/mol).
            Result to NaN if not found.
        - set: Whether the molecule has been seen at least once in the training dataset.
            Result to NaN if not found.

    """

    # [1]: Predict BDEs / BDFEs for every bond on all possible SMILES
    MaxNumBondsFound: int = GetMaxBonds(smiles_list)
    function = lambda: (
        get_features(smiles, max_num_edges=2 * MaxNumBondsFound)
        for smiles in smiles_list
    )

    input_dataset = tf.data.Dataset.from_generator(
        function, output_signature=preprocessor.output_signature
    ).cache()

    batched_dataset = input_dataset.padded_batch(batch_size=batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    if verbose:
        model.summary()

    bdes, bdfes = model.predict(batched_dataset)

    # [2]: Calculating all valid reactions / fragments of all SMILES found
    RDMolProcessor: MolProcessor = MolProcessor(None)
    reactions = []
    for smiles in smiles_list:
        RDMolProcessor.add_mol(mol=smiles, Hs=True)
        temp_reaction = RDMolProcessor.get_reactions(to_dict=False, reverse_reaction=False, single_bond_only=True,
                                                     strict_radical=True, zero_duplicate=drop_duplicates,
                                                     aro_ring=False, non_aro_ring=False, aro_ring_attached=True,
                                                     non_aro_ring_attached=True, non_ring_attached=True)
        reactions.extend(temp_reaction)
        RDMolProcessor.clear_mol()

    label: List[str] = RDMolProcessor.get_label()
    t_label = [label[0], label[3]]  # 'molecule', 'bond_index'
    base_df = pd.DataFrame(reactions, index=None, columns=label)

    # Re-worked on all BDEs/BDFEs and do reindex to match with the bde_df
    bde_df = (
        pd.DataFrame(bdes.squeeze(axis=-1), index=smiles_list)
        .T.unstack().reindex(base_df[t_label])
    )

    bdfe_df = (
        pd.DataFrame(bdfes.squeeze(axis=-1), index=smiles_list)
        .T.unstack().reindex(base_df[t_label])
    )

    base_df["bde_pred"] = bde_df.values
    base_df["bdfe_pred"] = bdfe_df.values

    is_valid = pd.Series(
        {
            smiles: not validate_inputs(input_)[0]
            for smiles, input_ in zip(smiles_list, input_dataset)
        },
        name="is_valid",
    )

    base_df = base_df.merge(is_valid, left_on="molecule", right_index=True)
    base_df = base_df.merge(bde_dft[["molecule", "bond_index", "bde", "bdfe", "set"]],
                            on=t_label, how="left")
    # print(base_df)
    if verbose:
        print(base_df)
    return base_df
