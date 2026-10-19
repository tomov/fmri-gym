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
             "move": ["UP", "DOWN"], "confirm": "RETURN"}

``key`` is required; the rest default to the values shown. ``reset`` starts
the episode over (a new ``reset`` of the env with the same seed: the very
instance the subject gave up on), ``forfeit`` ends the block and moves on to
the next phase, ``resume`` goes on where the game paused. Options are shown in
the order listed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import pygame

from .keys import get_events, held_key_names, key_name

if TYPE_CHECKING:
    from .display import Display
    from .logging import Logger

OPTIONS = ("reset", "forfeit", "resume")
_LABELS = {"reset": "Restart the level", "forfeit": "Give up this level", "resume": "Resume"}
_DEFAULTS = {"hold": 5.0, "after": 15.0, "options": list(OPTIONS),
             "move": ["UP", "DOWN"], "confirm": "RETURN"}


@dataclass(frozen=True, slots=True)
class MenuSettings:
    """A phase's ``menu`` field with its defaults filled in and its key names upper-cased."""

    #: the key held to open the menu
    key: str
    #: seconds it has to be held
    hold: float
    #: seconds into the block before the menu is available
    after: float
    #: the options offered, in the order shown
    options: tuple[str, ...]
    #: the keys that move the choice: to the previous option, to the next
    move: tuple[str, str]
    #: the key that confirms the choice
    confirm: str

    @classmethod
    def from_dict(cls, menu_spec: dict) -> MenuSettings:
        """Read a phase's ``menu`` dict (already validated by :func:`menu_problems`)."""
        filled = {**_DEFAULTS, **menu_spec}
        previous_key, next_key = filled["move"]
        return cls(key=filled["key"].upper(), hold=float(filled["hold"]),
                   after=float(filled["after"]), options=tuple(filled["options"]),
                   move=(previous_key.upper(), next_key.upper()),
                   confirm=filled["confirm"].upper())


def menu_problems(menu_spec: object) -> list[str]:
    """What is wrong with a phase's ``menu`` dict, for :func:`curriculum.validate_curriculum`.

    :param menu_spec: the phase's ``"menu"`` value.
    :return: human-readable problems, empty when the menu is usable.
    """
    if not isinstance(menu_spec, dict):
        return [f"menu: expected a dict with at least 'key', got {menu_spec!r}"]
    problems = []
    if not isinstance(menu_spec.get("key"), str):
        problems.append("menu.key: the key to hold is required, e.g. \"X\"")
    for field_name in ("hold", "after"):
        seconds = menu_spec.get(field_name, _DEFAULTS[field_name])
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
            problems.append(f"menu.{field_name}: expected seconds (>= 0), got {seconds!r}")
    options = menu_spec.get("options", _DEFAULTS["options"])
    if not isinstance(options, list) or not options or set(options) - set(OPTIONS):
        problems.append(f"menu.options: expected a non-empty list from {list(OPTIONS)}, "
                        f"got {options!r}")
    move = menu_spec.get("move", _DEFAULTS["move"])
    if not (isinstance(move, list) and len(move) == 2
            and all(isinstance(key, str) for key in move)):
        problems.append(f"menu.move: expected two key names [previous, next], got {move!r}")
    if not isinstance(menu_spec.get("confirm", _DEFAULTS["confirm"]), str):
        problems.append(f"menu.confirm: expected a key name, got {menu_spec.get('confirm')!r}")
    unknown = set(menu_spec) - {"key"} - set(_DEFAULTS)
    if unknown:
        problems.append(f"menu: unknown fields {sorted(unknown)}")
    return problems


class Menu:
    """One game block's menu: the hold that arms it, and the pop-up itself.

    :param menu_spec: the phase's ``"menu"`` dict (already validated).
    :param block_start_perf: ``perf_counter`` at which the block began.
    :param held: returns the names of the keys held now (a test can fake it).
    """

    def __init__(self, menu_spec: dict, block_start_perf: float,
                 held: Callable[[], frozenset[str]] = held_key_names) -> None:
        self.settings = MenuSettings.from_dict(menu_spec)
        self.available_from_perf = block_start_perf + self.settings.after
        self._held = held
        self._held_since_perf: float | None = None
        self.pending = False        # the hold completed; the pop-up is due
        self.events: list[dict] = []  # one per pop-up: when, and what was chosen

    def describe(self) -> dict:
        """The settings in force and every pop-up so far, for the manifest."""
        return {**asdict(self.settings), "events": self.events}

    def armed(self) -> bool:
        """Poll the hold; ``True`` (once, and :attr:`pending`) when it completes.

        Call often while the game runs. The hold only counts once the menu is
        available, and starts over on every release.
        """
        now_perf = time.perf_counter()
        if self.settings.key not in self._held() or now_perf < self.available_from_perf:
            self._held_since_perf = None
            return False
        if self._held_since_perf is None:
            self._held_since_perf = now_perf
        if now_perf - self._held_since_perf < self.settings.hold or self.pending:
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
        self.pending, self._held_since_perf = False, None
        settings = self.settings
        opened_run_time = run_time()
        choice_index = 0
        while True:
            self._draw(display, choice_index)
            for event in get_events():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type not in (pygame.KEYDOWN, pygame.KEYUP):
                    continue
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                key = key_name(event.key)
                if key is None:
                    continue
                logger.log(type="input_event", run_time=run_time(), name=key,
                           down=event.type == pygame.KEYDOWN)
                if event.type != pygame.KEYDOWN:
                    continue
                if key == settings.move[0]:
                    choice_index = (choice_index - 1) % len(settings.options)
                elif key == settings.move[1]:
                    choice_index = (choice_index + 1) % len(settings.options)
                elif key == settings.confirm:
                    chosen = settings.options[choice_index]
                    self.events.append({"opened": opened_run_time, "closed": run_time(),
                                        "choice": chosen})
                    return chosen
            display.idle(time.perf_counter() + 0.005, poll=0.005)

    def _draw(self, display: Display, choice_index: int) -> None:
        """Render the option list with the option at ``choice_index`` marked."""
        settings = self.settings
        rows = [f"{'>' if index == choice_index else ' '}  {_LABELS[option]}"
                for index, option in enumerate(settings.options)]
        display.draw_text("Paused\n\n" + "\n".join(rows) + "\n\n"
                          f"({settings.move[0]}/{settings.move[1]} to choose, "
                          f"{settings.confirm} to select)",
                          align="left")
