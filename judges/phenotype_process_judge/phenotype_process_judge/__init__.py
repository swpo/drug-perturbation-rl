"""Process-coherence judge audit environment."""

from typing import Any


def load_environment(*args: Any, **kwargs: Any):
    """Import the Verifiers integration lazily for contract-only tooling."""
    from .env import load_environment as _load_environment

    return _load_environment(*args, **kwargs)


__all__ = ["load_environment"]
