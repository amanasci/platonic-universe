from typing import Type, Dict, Any, Callable

_REGISTRY: Dict[str, Type] = {}


def register_adapter(alias: str, adapter_cls: Type) -> None:
    """
    Register a model adapter class under a short alias.

    Adapter classes should implement the minimal methods used by experiments:
      - load(model_name: str, size: str) -> None
      - embed_batch(batch: Dict[str, Any]) -> Any (tensor/ndarray)
      - get_preprocessor() -> callable
    """
    _REGISTRY[alias] = adapter_cls


def get_adapter(alias: str) -> Type:
    """
    Retrieve a registered adapter class (not an instance). Raises KeyError if not found.
    """
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(
            f"Model adapter for alias '{alias}' not found. "
            f"Available adapters: {sorted(list(_REGISTRY.keys()))}"
        ) from exc


def list_adapters() -> list:
    """Return a sorted list of registered adapter aliases."""
    return sorted(list(_REGISTRY.keys()))
