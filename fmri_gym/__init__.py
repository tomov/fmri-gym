"""fmri-gym: one fMRI experiment framework for any Gymnasium-compatible game.

The experiment loop is engine-agnostic; each game engine is a small EnvAdapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .session import Session, Clock
from .display import Display
from .logging import Logger

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter

__all__ = ["Session", "Clock", "Display", "Logger", "get_adapter"]


def get_adapter(backend: str, **kwargs: Any) -> EnvAdapter:
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
    if backend == "vgdl":
        from .adapters.vgdl import VGDLAdapter
        return VGDLAdapter(**kwargs)
    if backend == "crafter":
        from .adapters.crafter import CrafterAdapter
        return CrafterAdapter(**kwargs)
    if backend == "minihack":
        from .adapters.minihack import MiniHackAdapter
        return MiniHackAdapter(**kwargs)
    if backend in ("supertuxkart","stk"):
        from .adapters.supertuxkart import SuperTuxKartAdapter
        return SuperTuxKartAdapter(**kwargs)
    if backend == "rushhour":
        from .adapters.rushhour import RushHourAdapter
        return RushHourAdapter(**kwargs)
    if backend == "baba":
        from .adapters.baba import BabaAdapter
        return BabaAdapter(**kwargs)
    if backend == "overcooked":
        from .adapters.overcooked import OvercookedAdapter
        return OvercookedAdapter(**kwargs)
    if backend == "vizdoom":
        from .adapters.vizdoom import VizDoomAdapter
        return VizDoomAdapter(**kwargs)
    if backend == "nethack":
        from .adapters.nethack import NetHackAdapter
        return NetHackAdapter(**kwargs)
    if backend in ("aigamestore", "p5"):
        from .adapters.aigamestore import AIGameStoreAdapter
        return AIGameStoreAdapter(**kwargs)
    raise ValueError(f"unknown backend: {backend!r}")
