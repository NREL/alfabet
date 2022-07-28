from rdkit import RDLogger

from . import _version

# I think this is disable by default on RDKit version 2022.03.x and later ?
# See here: https://github.com/rdkit/rdkit/releases/tag/Release_2022_03_1
RDLogger.DisableLog("rdApp.*")

__version__ = _version.get_versions()["version"]

_model_tag = "v0.1.1"  # Tag on https://github.com/pstjohn/alfabet-models/
__base_url: str = f"https://github.com/pstjohn/alfabet-models/releases/download/{_model_tag}/"

# Cast to key-value to better value management
MODEL_CONFIG = {
    'tag': _model_tag,
    'base_url': __base_url,
    'preprocessor_name': 'preprocessor.json',
    'bde_dft_dataset_name': 'bonds_for_neighbors.csv.gz',
    'model_name': 'model.tar.gz',
    'neighbor_pipeline_name': 'bond_embedding_nbrs.p.z',
}
