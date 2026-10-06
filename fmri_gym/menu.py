"""The in-game menu: hold a key to pause and choose reset / forfeit / resume.

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
             "move": ["UP", "DOWN"], "confirm": "A"}

``key`` is required; the rest default to the values shown. Keys are rig keys
(:data:`fmri_gym.rig.CONTROLS`), so the menu works on every device. ``reset`` starts
the episode over (a new ``reset`` of the env with the same seed: the very
instance the subject gave up on), ``forfeit`` ends the block and moves on to
the next phase, ``resume`` goes on where the game paused. Options are shown in
the order listed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import pygame

from .keys import event_name, get_events, held_key_names
from .rig import CONTROLS

if TYPE_CHECKING:
    from .display import Display
    from .logging import Logger

OPTIONS = ("reset", "forfeit", "resume")
_LABELS = {"reset": "Restart the level", "forfeit": "Give up this level", "resume": "Resume"}
_DEFAULTS = {"hold": 5.0, "after": 15.0, "options": list(OPTIONS),
             "move": ["UP", "DOWN"], "confirm": "A"}


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
    elif spec["key"] not in CONTROLS:
        out.append(f"menu.key: {spec['key']!r} is not a rig key ({', '.join(CONTROLS)})")
    for field in ("hold", "after"):
        value = spec.get(field, _DEFAULTS[field])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            out.append(f"menu.{field}: expected seconds (>= 0), got {value!r}")
    options = spec.get("options", _DEFAULTS["options"])
    if not isinstance(options, list) or not options or set(options) - set(OPTIONS):
        out.append(f"menu.options: expected a non-empty list from {list(OPTIONS)}, got {options!r}")
    move = spec.get("move", _DEFAULTS["move"])
    if not (isinstance(move, list) and len(move) == 2 and all(k in CONTROLS for k in move)):
        out.append(f"menu.move: expected two rig keys [previous, next], got {move!r}")
    if spec.get("confirm", _DEFAULTS["confirm"]) not in CONTROLS:
        out.append(f"menu.confirm: expected a rig key ({', '.join(CONTROLS)}), "
                   f"got {spec.get('confirm')!r}")
    unknown = set(spec) - {"key"} - set(_DEFAULTS)
    if unknown:
        out.append(f"menu: unknown fields {sorted(unknown)}")
    return out


class Menu:
    """One game block's menu: the hold that arms it, and the pop-up itself.

    :param spec: the phase's ``"menu"`` dict (already validated).
    :param block_start: ``perf_counter`` at which the block began.
    :param held: returns the names of the keys held now (a test can fake it).
    """

    def __init__(self, spec: dict, block_start: float,
                 held: Callable[[], frozenset[str]] = held_key_names) -> None:
        cfg = {**_DEFAULTS, **spec}
        self.key = cfg["key"]
        self.hold = float(cfg["hold"])
        self.after = float(cfg["after"])
        self.available_from = block_start + self.after
        self.options = list(cfg["options"])
        self.prev_key, self.next_key = cfg["move"]
        self.confirm_key = cfg["confirm"]
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
            for event in get_events():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type not in (pygame.KEYDOWN, pygame.KEYUP):
                    continue
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                name = event_name(event)
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
        rows = [f"{'>' if i == choice else ' '}  {_LABELS[o]}" for i, o in enumerate(self.options)]
        display.draw_text("Paused\n\n" + "\n".join(rows) + "\n\n"
                          f"({self.prev_key}/{self.next_key} to choose, {self.confirm_key} to select)",
                          align="left")
