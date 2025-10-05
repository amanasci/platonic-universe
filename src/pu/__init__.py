import os
import logging
from types import SimpleNamespace
from typing import Optional, Dict, Any


# Public helpers for programmatic use (wrappers around your CLI handlers)
from .experiments import run_experiment#, get_specformer_embeddings 
from .metrics import run_mknn_comparison as _mknn

_log = logging.getLogger(__name__)
PU_CACHE_DIR: Optional[str] = None

def setup_cache_dir(path: str) -> None:
    """
    Set a directory for caches used by the package and external libs (HuggingFace, XDG).
    Creates the dir if it does not exist and sets HF_HOME / XDG_CACHE_HOME env vars.
    """
    global PU_CACHE_DIR
    os.makedirs(path, exist_ok=True)
    os.environ.setdefault("HF_HOME", path)
    os.environ.setdefault("XDG_CACHE_HOME", path)
    PU_CACHE_DIR = path
    _log.info("Cache dir set to %s", path)

def compare_models_mknn(parquet_file: str) -> Dict[str, Any]:
    """
    Wrapper around the mknn comparison function to return results as a dictionary.
    Accepts the path to a parquet file produced by `run_experiment`.
    """
    return _mknn(parquet_file)

__all__ = ["setup_cache_dir", "compare_models_mknn", "run_experiment"] # "get_specformer_embeddings"]
