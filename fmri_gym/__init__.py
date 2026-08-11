"""fmri-gym: one fMRI experiment framework for any Gymnasium-compatible game.

The experiment loop is engine-agnostic; each game engine is a small EnvAdapter.
"""

from .session import Session, Clock
from .display import Display
from .logging import Logger

__all__ = ["Session", "Clock", "Display", "Logger", "get_adapter"]


def get_adapter(backend, **kwargs):
    """Lazily construct an adapter by backend name.

    Lazy imports keep optional deps (ale_py, stable_retro) from being required
    unless a curriculum actually uses that backend.
    """
    if backend == "ale":
        from .adapters.ale import ALEAdapter
        return ALEAdapter(**kwargs)
    if backend == "retro":
        from .adapters.retro import RetroAdapter
        return RetroAdapter(**kwargs)
    if backend == "gym":
        from .adapters.default import DefaultAdapter
        return DefaultAdapter(**kwargs)
    raise ValueError(f"unknown backend: {backend!r}")
