"""One :class:`~.base.SymbolicReader` per backend; :func:`get_reader` picks it."""

from __future__ import annotations

from .base import Observation, SymbolicObject, SymbolicReader

__all__ = ["Observation", "SymbolicObject", "SymbolicReader", "get_reader"]


def get_reader(backend: str) -> SymbolicReader:
    """A fresh reader for a block's backend; the generic one for a backend with none.

    :param backend: the block's ``backend`` (an adapter's ``name``).
    """
    if backend in ("vizdoom", "coom"):
        from .doom import DoomReader
        reader: SymbolicReader = DoomReader()
    elif backend in ("minihack", "nethack"):
        from .games import NLEReader
        reader = NLEReader()
    elif backend == "crafter":
        from .games import CrafterReader
        reader = CrafterReader()
    elif backend == "rushhour":
        from .games import RushHourReader
        reader = RushHourReader()
    elif backend == "baba":
        from .games import BabaReader
        reader = BabaReader()
    else:
        reader = SymbolicReader()
    reader.reset()
    return reader
