import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pooch
from nfp.frameworks import tf
from pooch import retrieve
from rdkit import RDLogger

from alfabet import _model_files_baseurl

RDLogger.DisableLog("rdApp.*")

model_files = retrieve(
    _model_files_baseurl + "model.tar.gz",
    known_hash="sha256:f1c2b9436f2d18c76b45d95140e6"
    "a08c096250bd5f3e2b412492ca27ab38ad0c",
    processor=pooch.Untar(extract_dir="model"),
)

model = tf.keras.models.load_model(os.path.dirname(model_files[0]))

bde_dft = pd.read_csv(
    retrieve(
        _model_files_baseurl + "bonds_for_neighbors.csv.gz",
        known_hash="sha256:d4fb825c42d790d4b2b4bd5dc2d"
        "87c844932e2da82992a31d7521ce51395adb1",
    )
)


def validate_inputs(inputs: dict) -> Tuple[bool, np.array, np.array]:
    """Check the given SMILES to ensure it's present in the model's
    preprocessor dictionary.

    Returns:
    (is_outlier, missing_atom, missing_bond)

    """
    inputs = {key: np.asarray(val) for key, val in inputs.items()}

    missing_bond = np.array(list(set(inputs["bond_indices"][inputs["bond"] == 1])))
    missing_atom = np.arange(len(inputs["atom"]))[inputs["atom"] == 1]

    is_outlier = (missing_bond.size != 0) | (missing_atom.size != 0)

    return is_outlier, missing_atom, missing_bond


def tf_model_forward(inputs: dict) -> Tuple[List[float], List[float]]:
    """Mimimcs a call to tf-serving by padding the input arrays to resemble a single
    batch and passing them through the loaded tensorflow model

    Args:
        inputs (dict): input arrays for a single molecule

    Returns:
        Tuple[List[int], List[int]]: flattened output arrays for predicted bde and bdfe
    """

    def expand(val):
        return tf.constant(np.expand_dims(val, 0))

    bde_pred, bdfe_pred = model({key: expand(val) for key, val in inputs.items()})
    return bde_pred.numpy().squeeze().tolist(), bdfe_pred.numpy().squeeze().tolist()


def format_predictions_into_dataframe(
    bde_pred: List[float],
    bdfe_pred: List[float],
    frag_df: pd.DataFrame,
    drop_duplicates: bool = False,
) -> pd.DataFrame:

    # Reindex predictions to fragment dataframe
    frag_df["bde_pred"] = (
        pd.Series(bde_pred).reindex(frag_df.bond_index).reset_index(drop=True)
    )
    frag_df["bdfe_pred"] = (
        pd.Series(bdfe_pred).reindex(frag_df.bond_index).reset_index(drop=True)
    )

    # Add DFT calculated bdes
    frag_df = frag_df.merge(
        bde_dft[["molecule", "bond_index", "bde", "bdfe", "set"]],
        on=["molecule", "bond_index"],
        how="left",
    )

    # Drop duplicate entries and sort from weakest to strongest
    frag_df = frag_df.sort_values("bde_pred").reset_index(drop=True)

    if drop_duplicates:
        frag_df = frag_df.drop_duplicates(["fragment1", "fragment2"]).reset_index(
            drop=True
        )

    frag_df["has_dft_bde"] = frag_df.bde.notna()

    return frag_df
