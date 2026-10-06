"""Shared keyboard helpers for the display loop and adapters.

What a game hears is the **rig keys** (:data:`fmri_gym.rig.CONTROLS`): a typed
key that stands for one (the keyboard's :data:`fmri_gym.rig.TYPED_KEYS`, and the
rig file's own) is reported as that rig key, and the controller
(:mod:`fmri_gym.pad`, registered here) presses them by name. Any other key is
no game key.

Typed keys are named as pygame names them (:func:`pygame.key.name`: ``"1"``,
``"space"``, ``"left shift"``, ``"[1]"`` for keypad 1), in any case, and are
compared by keycode: two spellings of one key (``"[1]"``, ``"keypad 1"``) are the
same key, and a name pygame does not know is refused (:func:`keycode`).

The pad also feeds events in: read the queue with :func:`get_events`, never
``pygame.event.get()``, and a pad press arrives as its rig key.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pygame

from . import rig

#: Devices other than the keyboard that can hold a key down (:mod:`fmri_gym.pad`
#: registers the gamepad here). Kept as a list of callables so this module stays
#: the one place that answers "what is held", whatever is plugged in.
_held_sources: list[Callable[[], frozenset[str]]] = []
#: Keycode -> rig key, read from the rig file on first use (:func:`typed_keys`).
_typed: dict[int, str] | None = None


def keycode(name: str) -> int:
    """The keycode of a typed key, named as pygame names it, in any case.

    :raises ValueError: a name pygame does not know.
    """
    # SDL reads names from a static table; pygame warns before pygame.init() all
    # the same, and a config is checked before the window opens.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pygame.key.key_code(name)
        except ValueError:
            raise ValueError(f"{name!r} is not a key name (pygame's names: \"1\", \"space\", "
                             "\"left shift\", \"[1]\" for keypad 1, ...)") from None


def is_key(name: str) -> bool:
    """Whether ``name`` is a typed key's name (:func:`keycode` takes it)."""
    try:
        keycode(name)
    except ValueError:
        return False
    return True


def typed_keys() -> dict[int, str]:
    """Keycode -> the rig key it stands for, on this machine.

    :raises ValueError: a rig file whose ``keys`` or ``controls`` are wrong.
    """
    global _typed
    if _typed is None:
        site = rig.read() or {}
        problems = rig.key_problems(site, is_key)
        own = site.get("keys", {}) if not problems else {}
        codes = [keycode(name) for name in own]
        problems += [f'"keys": {name!r} is a key named twice' for name, code in zip(own, codes)
                     if codes.count(code) > 1]
        if problems:
            raise ValueError(f"{rig.current()}: " + "; ".join(problems))
        _typed = {**{keycode(n): r for n, r in rig.TYPED_KEYS.items()},
                  **dict(zip(codes, own.values()))}
    return _typed


def reread_rig() -> None:
    """Forget the rig file's keys, so the next read takes the file as it is now."""
    global _typed
    _typed = None


def register_held_source(source: Callable[[], frozenset[str]]) -> None:
    """Add a device whose held keys count as held, alongside the keyboard's.

    :param source: called each frame; returns the rig keys it is holding down.
    """
    _held_sources.append(source)


#: Devices whose input reaches the event queue only once something moves it
#: there (:mod:`fmri_gym.pad` registers its ``pump``). :func:`get_events` runs
#: them first, so no loop has to remember to.
_event_sources: list[Callable[[], None]] = []


def register_event_source(source: Callable[[], None]) -> None:
    """Add a device that posts its input to the event queue when called.

    :param source: called before every :func:`get_events`; posts the keypresses
        it has waiting.
    """
    if source not in _event_sources:
        _event_sources.append(source)


def get_events(*types: int) -> list[pygame.event.Event]:
    """Take events off the queue, the registered devices' included.

    Use this, not ``pygame.event.get()``, so a pad press is on the queue before
    it is read. Returns a list, not a generator: the devices must be pumped now,
    not at the caller's first iteration.

    :param types: only take these event types (all of them when none given).
    :return: the events, in order.
    """
    for source in _event_sources:
        source()
    return pygame.event.get(types) if types else pygame.event.get()


def held_key_names() -> frozenset[str]:
    """Return the rig keys held now (``"LEFT"``, ``"A"``, ...), from every device.

    :return: frozenset of the held rig keys.
    """
    pressed = pygame.key.get_pressed()
    names = {name for code, name in typed_keys().items() if pressed[code]}
    for source in _held_sources:
        names |= source()
    return frozenset(names)


def key_name(code: int) -> str:
    """The name pygame gives a keycode (``"space"``, ``"1"``); what a game hears is
    :func:`event_name`."""
    return pygame.key.name(code)


def event_name(event: Any) -> str | None:
    """The rig key of a ``KEYDOWN`` / ``KEYUP``, or ``None`` if it is not a game key.

    A controller press (``pad=True``) carries its rig key as ``name``; a typed
    key is read through :func:`typed_keys`.

    :param event: a pygame key event.
    """
    if getattr(event, "pad", False):
        return event.name
    return typed_keys().get(event.key)
