from rdkit import RDLogger

from . import _version

RDLogger.DisableLog("rdApp.*")


__version__ = _version.get_versions()["version"]

_model_tag = "v0.1.1"  # Tag on https://github.com/pstjohn/alfabet-models/
_model_files_baseurl = (
    f"https://github.com/pstjohn/alfabet-models/releases/download/{_model_tag}/"
)
