from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

class ModelAdapter(ABC):
    """
    Minimal adapter interface for models used by experiments.
    Implementations should handle model loading, provide a preprocessor callable,
    and convert a batch from the DataLoader into embeddings.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        self.model_name = model_name
        self.size = size
        self.alias = alias

    @abstractmethod
    def load(self) -> None:
        "Load model and any required resources (to cuda if needed)."
        raise NotImplementedError

    @abstractmethod
    def get_preprocessor(self, modes: Iterable[str]):
        """
        Return a callable that can be passed to `datasets.Dataset.map`.
        Should accept a single example dict and return a dict of tensors/arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Given a batch from the DataLoader and the mode name (e.g., 'hsc' or 'jwst'),
        return a torch.Tensor (or array-like) with embeddings for that batch.
        """
        raise NotImplementedError
