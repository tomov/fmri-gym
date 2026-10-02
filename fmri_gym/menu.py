"""The in-game menu: hold a key to pause and choose rewind / reset / forfeit / resume.

A subject can get stuck (a Baba Is You puzzle with the rules pushed into a
corner) or want out of a level, and the button box has few buttons, so the
menu is opened by HOLDING one key for several seconds -- a key that may well
do something in the game meanwhile -- and then choosing with two more keys and
confirming with a fourth, which is very hard to do by accident. It is not
meant to be used lightly, so it is unavailable for the first ``after`` seconds
of a block. Optional and default-off: a game phase opts in with a ``"menu"``
dict, and the loop in :mod:`fmri_gym.run` never knows which game it is for.

Phase field::

    "menu": {"key": "X", "hold": 5.0, "after": 15.0,
             "options": ["reset", "forfeit", "resume"],
             "move": ["UP", "DOWN"], "confirm": "RETURN"}

``key`` is required; the rest default to the values shown. ``reset`` starts
the episode over (a new ``reset`` of the env with the same seed: the very
instance the subject gave up on), ``forfeit`` ends the block and moves on to
the next phase, ``resume`` goes on where the game paused. Options are shown in
the order listed.

``rewind`` is the fifth option and the one not in the defaults: it puts the
world back a few frames and plays on (:mod:`fmri_gym.rewind`), and how far back
is a phase field rather than a menu one, so a menu that offers it without that
field would pause the game and do nothing. A phase lists it explicitly or does
not have it, and ``validate_config`` refuses the half of either.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import pygame

from . import pad
from .keys import held_key_names, key_name

if TYPE_CHECKING:
    from .display import Display
    from .logging import Logger

#: What a menu offers unless the phase says otherwise.
DEFAULT_OPTIONS = ("reset", "forfeit", "resume")
#: Every option a phase may list. ``rewind`` is not a default: it does nothing
#: without the phase field that says how far back to go (:mod:`fmri_gym.rewind`).
OPTIONS = ("rewind", *DEFAULT_OPTIONS)
_LABELS = {"rewind": "Go back a few frames", "reset": "Restart the level",
           "forfeit": "Give up this level", "resume": "Resume"}
_DEFAULTS = {"hold": 5.0, "after": 15.0, "options": list(DEFAULT_OPTIONS),
             "move": ["UP", "DOWN"], "confirm": "RETURN"}


def options_of(phase: dict) -> list[str]:
    """The options a game phase's pause menu offers.

    For the cross-field checks in :func:`fmri_gym.config.validate_config`: an
    option and the phase field behind it are written in two places and have to
    agree.

    :param phase: a game-phase config.
    :return: the options in force, empty when the phase has no menu (or one
        :func:`menu_problems` is about to refuse).
    """
    spec = phase.get("menu")
    if not isinstance(spec, dict):
        return []
    options = spec.get("options", _DEFAULTS["options"])
    return [o for o in options if isinstance(o, str)] if isinstance(options, list) else []


def menu_problems(spec: object) -> list[str]:
    """What is wrong with a phase's ``menu`` dict, for :func:`config.validate_config`.

    :param spec: the phase's ``"menu"`` value.
    :return: human-readable problems, empty when the menu is usable.
    """
    if not isinstance(spec, dict):
        return [f"menu: expected a dict with at least 'key', got {spec!r}"]
    out = []
    if not isinstance(spec.get("key"), str):
        out.append("menu.key: the key to hold is required, e.g. \"X\"")
    for field in ("hold", "after"):
        value = spec.get(field, _DEFAULTS[field])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            out.append(f"menu.{field}: expected seconds (>= 0), got {value!r}")
    options = spec.get("options", _DEFAULTS["options"])
    if not isinstance(options, list) or not options or set(options) - set(OPTIONS):
        out.append(f"menu.options: expected a non-empty list from {list(OPTIONS)}, got {options!r}")
    move = spec.get("move", _DEFAULTS["move"])
    if not (isinstance(move, list) and len(move) == 2 and all(isinstance(k, str) for k in move)):
        out.append(f"menu.move: expected two key names [previous, next], got {move!r}")
    if not isinstance(spec.get("confirm", _DEFAULTS["confirm"]), str):
        out.append(f"menu.confirm: expected a key name, got {spec.get('confirm')!r}")
    unknown = set(spec) - {"key"} - set(_DEFAULTS)
    if unknown:
        out.append(f"menu: unknown fields {sorted(unknown)}")
    return out


class Menu:
    """One game block's menu: the hold that arms it, and the pop-up itself.

    :param spec: the phase's ``"menu"`` dict (already validated).
    :param block_start: ``perf_counter`` at which the block began.
    :param held: returns the names of the keys held now (a test can fake it).
    :param labels: what to call an option instead of its default label, for
        the ones whose wording belongs to a phase field rather than to the
        menu (``rewind``: how far back, in the units it was asked for).
    """

    def __init__(self, spec: dict, block_start: float,
                 held: Callable[[], frozenset[str]] = held_key_names,
                 labels: dict[str, str] | None = None) -> None:
        cfg = {**_DEFAULTS, **spec}
        self.labels = {**_LABELS, **(labels or {})}
        self.key = cfg["key"].upper()
        self.hold = float(cfg["hold"])
        self.after = float(cfg["after"])
        self.available_from = block_start + self.after
        self.options = list(cfg["options"])
        self.prev_key, self.next_key = (k.upper() for k in cfg["move"])
        self.confirm_key = cfg["confirm"].upper()
        self._held = held
        self._held_since: float | None = None
        self.pending = False        # the hold completed; the pop-up is due
        self.events: list[dict] = []  # one per pop-up: when, and what was chosen

    def describe(self) -> dict:
        """The settings in force and every pop-up so far, for the manifest."""
        return {"key": self.key, "hold": self.hold, "after": self.after,
                "options": self.options, "move": [self.prev_key, self.next_key],
                "confirm": self.confirm_key, "events": self.events}

    def armed(self) -> bool:
        """Poll the hold; ``True`` (once, and :attr:`pending`) when it completes.

        Call often while the game runs. The hold only counts once the menu is
        available, and starts over on every release.
        """
        now = time.perf_counter()
        if self.key not in self._held() or now < self.available_from:
            self._held_since = None
            return False
        if self._held_since is None:
            self._held_since = now
        if now - self._held_since < self.hold or self.pending:
            return False
        self.pending = True
        return True

    def run(self, display: Display, run_time: Callable[[], float], logger: "Logger") -> str:
        """Show the pop-up until an option is confirmed, and return it.

        :param display: the display to draw on.
        :param run_time: the run clock, for the log.
        :param logger: the run's logger; the menu's own presses are
            logged like any other.
        :return: the chosen option, or ``"quit"`` on window close / ESC.
        """
        self.pending, self._held_since = False, None
        opened = run_time()
        choice = 0
        while True:
            self._draw(display, choice)
            pad.pump()          # a pad press arrives as the key it stands for
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type not in (pygame.KEYDOWN, pygame.KEYUP):
                    continue
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                name = key_name(event.key)
                if name is None:
                    continue
                logger.log(type="input_event", run_time=run_time(), name=name,
                             down=event.type == pygame.KEYDOWN)
                if event.type != pygame.KEYDOWN:
                    continue
                if name == self.prev_key:
                    choice = (choice - 1) % len(self.options)
                elif name == self.next_key:
                    choice = (choice + 1) % len(self.options)
                elif name == self.confirm_key:
                    chosen = self.options[choice]
                    self.events.append({"opened": opened, "closed": run_time(), "choice": chosen})
                    return chosen
            display.idle(time.perf_counter() + 0.005, poll=0.005)

    def _draw(self, display: Display, choice: int) -> None:
        """Render the option list with ``choice`` marked."""
        rows = [f"{'>' if i == choice else ' '}  {self.labels[o]}"
                for i, o in enumerate(self.options)]
        display.draw_text("Paused\n\n" + "\n".join(rows) + "\n\n"
                          f"({self.prev_key}/{self.next_key} to choose, {self.confirm_key} to select)",
                          align="left")
