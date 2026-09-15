"""Engine adapters. Construct one per game block via :func:`get_adapter`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import EnvAdapter


def get_adapter(backend: str, spec: dict) -> EnvAdapter:
    """Construct the :class:`~.base.EnvAdapter` wrapper for one game block.

    A fresh instance is built for every game block (it wraps the underlying
    engine env), so per-block state never leaks between blocks. Lazy imports
    keep optional deps (ale_py, stable_retro, ...) from being required unless a
    curriculum actually uses that backend.

    :param backend: backend id from the game phase (``"ale"``, ``"retro"``, ...).
    :param spec: the game-phase config dict, passed to the env's constructor.
    :return: a backend-specific :class:`~.base.EnvAdapter` wrapping a live env.
    :raises ValueError: if ``backend`` is unknown.
    """
    if backend == "ale":
        from .ale import ALEAdapter
        return ALEAdapter(spec)
    if backend == "retro":
        from .retro import RetroAdapter
        return RetroAdapter(spec)
    if backend == "gym":
        from .default import DefaultAdapter
        return DefaultAdapter(spec)
    if backend == "vgdl":
        from .vgdl import VGDLAdapter
        return VGDLAdapter(spec)
    if backend == "crafter":
        from .crafter import CrafterAdapter
        return CrafterAdapter(spec)
    if backend == "minihack":
        from .minihack import MiniHackAdapter
        return MiniHackAdapter(spec)
    if backend in ("supertuxkart", "stk"):
        from .supertuxkart import SuperTuxKartAdapter
        return SuperTuxKartAdapter(spec)
    if backend in ("stk_gym", "stk-gym"):
        from .stk_gym import STKGymAdapter
        return STKGymAdapter(spec)
    if backend == "rushhour":
        from .rushhour import RushHourAdapter
        return RushHourAdapter(spec)
    if backend == "baba":
        from .baba import BabaAdapter
        return BabaAdapter(spec)
    if backend == "overcooked":
        from .overcooked import OvercookedAdapter
        return OvercookedAdapter(spec)
    if backend == "vizdoom":
        from .vizdoom import VizDoomAdapter
        return VizDoomAdapter(spec)
    if backend == "nethack":
        from .nethack import NetHackAdapter
        return NetHackAdapter(spec)
    if backend in ("aigamestore", "p5"):
        from .aigamestore import AIGameStoreAdapter
        return AIGameStoreAdapter(spec)
    raise ValueError(f"unknown backend: {backend!r}")
