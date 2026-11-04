"""The in-game menu: hold a control to pause and choose reset / forfeit / resume.

A subject can get stuck (a Baba Is You puzzle with the rules pushed into a
corner) or want out of a level, and the button box has few buttons, so the
menu is opened by HOLDING one control for several seconds -- one that may well
do something in the game meanwhile -- and then choosing with two more controls
and confirming with a fourth, which is very hard to do by accident. It is not
meant to be used lightly, so it is unavailable for the first ``after`` seconds
of a block. Optional and default-off: a game phase opts in with a ``"menu"``
dict, and the loop in :mod:`fmri_gym.run` never knows which game it is for.

Phase field::

    "menu": {"control": "X", "hold": 5.0, "after": 15.0,
             "options": ["reset", "forfeit", "resume"],
             "move": ["UP", "DOWN"], "confirm": "RETURN"}

``control`` is required; the rest default to the values shown. Each is a
control name (:mod:`fmri_gym.inputs`), as a game phase's ``controls`` are. ``reset`` starts
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

from .inputs import ESCAPE, QUIT, Edge, InputEvent, is_control_of_any_input_device

if TYPE_CHECKING:
    from .display import Display
    from .logging import Logger

OPTIONS = ("reset", "forfeit", "resume")
_LABELS = {"reset": "Restart the level", "forfeit": "Give up this level", "resume": "Resume"}
_DEFAULTS = {"hold": 5.0, "after": 15.0, "options": list(OPTIONS),
             "move": ["UP", "DOWN"], "confirm": "RETURN"}


@dataclass(frozen=True, slots=True)
class MenuSettings:
    """A phase's ``menu`` field with its defaults filled in and its control names upper-cased."""

    #: the control held to open the menu
    control: str
    #: seconds it has to be held
    hold: float
    #: seconds into the block before the menu is available
    after: float
    #: the options offered, in the order shown
    options: tuple[str, ...]
    #: the controls that move the choice: to the previous option, to the next
    move: tuple[str, str]
    #: the control that confirms the choice
    confirm: str

    @classmethod
    def from_dict(cls, spec: dict) -> MenuSettings:
        """Read a phase's ``menu`` dict (already validated by :func:`menu_problems`)."""
        cfg = {**_DEFAULTS, **spec}
        previous, following = cfg["move"]
        return cls(control=cfg["control"].upper(), hold=float(cfg["hold"]),
                   after=float(cfg["after"]), options=tuple(cfg["options"]),
                   move=(previous.upper(), following.upper()),
                   confirm=cfg["confirm"].upper())


def menu_problems(spec: object) -> list[str]:
    """What is wrong with a phase's ``menu`` dict, for :func:`curriculum.validate_curriculum`.

    :param spec: the phase's ``"menu"`` value.
    :return: human-readable problems, empty when the menu is usable.
    """
    if not isinstance(spec, dict):
        return [f"menu: expected a dict with at least 'control', got {spec!r}"]
    out = []
    if not isinstance(spec.get("control"), str):
        out.append("menu.control: the control to hold is required, e.g. \"X\"")
    for field, names in (("control", [spec.get("control")]), ("move", spec.get("move", [])),
                         ("confirm", [spec.get("confirm", _DEFAULTS["confirm"])])):
        wrong = [n for n in (names if isinstance(names, list) else []) if isinstance(n, str)
                 and (not is_control_of_any_input_device(n.upper()) or n.upper() == ESCAPE)]
        if wrong:
            out.append(f"menu.{field}: {wrong} is no control name (ESCAPE quits the run): "
                       "the keyboard's (fmri_gym/inputs/keyboard.py) or a device's "
                       "(fmri_gym/inputs/<device>.py)")
    for field in ("hold", "after"):
        value = spec.get(field, _DEFAULTS[field])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            out.append(f"menu.{field}: expected seconds (>= 0), got {value!r}")
    options = spec.get("options", _DEFAULTS["options"])
    if not isinstance(options, list) or not options or set(options) - set(OPTIONS):
        out.append(f"menu.options: expected a non-empty list from {list(OPTIONS)}, got {options!r}")
    move = spec.get("move", _DEFAULTS["move"])
    if not (isinstance(move, list) and len(move) == 2 and all(isinstance(k, str) for k in move)):
        out.append(f"menu.move: expected two control names [previous, next], got {move!r}")
    if not isinstance(spec.get("confirm", _DEFAULTS["confirm"]), str):
        out.append(f"menu.confirm: expected a control name, got {spec.get('confirm')!r}")
    unknown = set(spec) - {"control"} - set(_DEFAULTS)
    if unknown:
        out.append(f"menu: unknown fields {sorted(unknown)}")
    return out


class Menu:
    """One game block's menu: the hold that arms it, and the pop-up itself.

    :param spec: the phase's ``"menu"`` dict (already validated).
    :param block_start: ``perf_counter`` at which the block began.
    :param held: returns the names of the controls held now (:meth:`InputDevice.held`).
    """

    def __init__(self, spec: dict, block_start: float,
                 held: Callable[[], frozenset[str]]) -> None:
        self.settings = MenuSettings.from_dict(spec)
        self.available_from = block_start + self.settings.after
        self._held = held
        self._held_since: float | None = None
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
        now = time.perf_counter()
        if self.settings.control not in self._held() or now < self.available_from:
            self._held_since = None
            return False
        if self._held_since is None:
            self._held_since = now
        if now - self._held_since < self.settings.hold or self.pending:
            return False
        self.pending = True
        return True

    def run(self, display: Display, run_time: Callable[[], float], logger: Logger,
            read_input_events: Callable[[], list[InputEvent]]) -> str:
        """Show the pop-up until an option is confirmed, and return it.

        :param display: the display to draw on.
        :param run_time: the run clock, for the log.
        :param logger: the run's logger; the menu's own presses are
            logged like any other.
        :param read_input_events: the edges since it was last called (the run's).
        :return: the chosen option, or ``"quit"`` on window close / ESC.
        """
        self.pending, self._held_since = False, None
        s = self.settings
        opened = run_time()
        choice = 0
        while True:
            self._draw(display, choice)
            for name, edge in read_input_events():
                down = edge is Edge.DOWN
                if name in (QUIT, ESCAPE):
                    if down:
                        return "quit"
                    continue
                logger.log(type="input_event", run_time=run_time(), name=name, down=down)
                if not down:
                    continue
                if name == s.move[0]:
                    choice = (choice - 1) % len(s.options)
                elif name == s.move[1]:
                    choice = (choice + 1) % len(s.options)
                elif name == s.confirm:
                    chosen = s.options[choice]
                    self.events.append({"opened": opened, "closed": run_time(), "choice": chosen})
                    return chosen
            display.idle(time.perf_counter() + 0.005, poll=0.005)

    def _draw(self, display: Display, choice: int) -> None:
        """Render the option list with ``choice`` marked."""
        s = self.settings
        rows = [f"{'>' if i == choice else ' '}  {_LABELS[o]}" for i, o in enumerate(s.options)]
        hint = f"({s.move[0]}/{s.move[1]} to choose, {s.confirm} to select)"
        display.draw_text("Paused\n\n" + "\n".join(rows) + "\n\n" + hint, align="left")
