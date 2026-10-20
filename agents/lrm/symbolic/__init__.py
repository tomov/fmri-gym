"""One :class:`~.base.SymbolicReader` per backend; :func:`get_reader` picks it."""

from __future__ import annotations

from .base import NOOP, Observation, SymbolicObject, SymbolicReader

__all__ = ["NOOP", "Observation", "SymbolicObject", "SymbolicReader", "get_reader"]


def get_reader(backend: str) -> SymbolicReader:
    """The reader for a block's backend, or the generic one if it has none.

    Which one was used is written in the transcript's header, so a transcript
    read through the generic reader says so.

    :param backend: the block's ``backend``.
    :return: a reader.
    """
    if backend == "vizdoom":
        from .vizdoom import VizDoomReader
        return VizDoomReader()
    return SymbolicReader()
