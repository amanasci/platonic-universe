from abc import ABC, abstractmethod
from typing import Iterable, Callable
from datasets import Dataset

class DatasetAdapter(ABC):
    """Minimal dataset adapter interface.

    Implementations should encapsulate all dataset-specific loading and column
    transformations so experiments only need to call `prepare(processor, modes, filterfun)`.
    """

    def __init__(self, hf_ds: str, comp_mode: str):
        self.hf_ds = hf_ds
        self.comp_mode = comp_mode

    @abstractmethod
    def load(self) -> None:
        """Load any external resources required by this adapter (if any)."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self, processor: Callable, modes: Iterable[str], filterfun: Callable):
        """Return a preprocessed `datasets.Dataset` ready for iteration.

        The adapter should:
        - load/concatenate/rename columns as needed for this dataset
        - call .filter(filterfun) when streaming is used (the caller provides filterfun)
        - call .map(processor)
        - remove any raw image columns that are not required for downstream code

        Args:
            processor: callable returned by model_adapter.get_preprocessor(modes)
            modes: sequence of mode names used by experiments (e.g. ['hsc', 'jwst'])
            filterfun: callable used to filter streaming datasets

        Returns:
            datasets.Dataset
        """
        raise NotImplementedError
