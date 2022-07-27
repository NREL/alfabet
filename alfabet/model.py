from typing import List
import pandas as pd
from nfp.frameworks import tf

from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs

from alfabet.utils import MolProcessor
from alfabet.prediction import bde_dft, model, validate_inputs
from alfabet.preprocessor import get_features, preprocessor

def NumBonds(smiles: str) -> int:
    return AddHs(MolFromSmiles(smiles)).GetNumBonds()


def GetMaxBonds(smiles_list: List[str]) -> int:
    return max(NumBonds(smiles) for smiles in smiles_list)


def predict(smiles_list: List[str], drop_duplicates: bool = True, 
            batch_size: int = 1):
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

    # [1]: Predict BDEs / BDFEs for every bond on all possible SMILES
    MaxNumBondsFound: int = GetMaxBonds(smiles_list)
    function = lambda : (
        get_features(smiles, max_num_edges=2 * MaxNumBondsFound) 
        for smiles in smiles_list
    )
        
    input_dataset = tf.data.Dataset.from_generator(
        function, output_signature=preprocessor.output_signature
        ).cache()

    batched_dataset = input_dataset.padded_batch(batch_size=batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    bdes, bdfes = model.predict(batched_dataset)

    # [2]: Calculating all valid reactions / fragments of all SMILES found
    RDMolProcessor: MolProcessor = MolProcessor()
    reactions = []
    for smiles in smiles_list:
        RDMolProcessor.AddMol(mol=smiles, Hs=True)
        temp_reaction = RDMolProcessor.GetReactions(
            ToDict=False, ReverseReaction=False, SingleBondOnly=True, StrictRadical=True, 
            ZeroDuplicate=drop_duplicates, AroRing=False, NonAroRing=False, 
            AroRingAttached=True, NonAroRingAttached=True, NonRingAttached=True        
        )
        reactions.extend(temp_reaction)
        RDMolProcessor.ClearMol()
    
    label: List[str] = RDMolProcessor.GetLabel()
    t_label = [label[0], label[3]]              # 'molecule', 'bond_index' 
    base_df = pd.DataFrame(reactions, index='mol', columns=label)
    
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

    """
    pred_df = pd.concat((get_fragments(smiles) for smiles in smiles_list))
    if drop_duplicates:
        pred_df = pred_df.drop_duplicates(["fragment1", "fragment2"]).reset_index(
            drop=True
        )
    
    bde_df = (
        pd.DataFrame(bdes.squeeze(axis=-1), index=smiles_list)
        .T.unstack()
        .reindex(pred_df[["molecule", "bond_index"]])
    )
    bdfe_df = (
        pd.DataFrame(bdfes.squeeze(axis=-1), index=smiles_list)
        .T.unstack()
        .reindex(pred_df[["molecule", "bond_index"]])
    )

    pred_df["bde_pred"] = bde_df.values
    pred_df["bdfe_pred"] = bdfe_df.values
    """
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
    return base_df
