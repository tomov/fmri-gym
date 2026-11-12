"""The pop-ups a held control opens over a game: the pause menu and the hint.

A subject can get stuck (a Baba Is You puzzle with the rules pushed into a
corner) or want out of a level, and the button box has few buttons, so the
menu is opened by HOLDING one control for several seconds -- one that may well
do something in the game meanwhile -- and then choosing with two more controls
and confirming with a fourth, which is very hard to do by accident. It is not
meant to be used lightly, so it is unavailable for the first ``after`` seconds
of a block. The hint is opened the same way and shows its text, the game paused,
for as long as its control stays held. Both are optional and default-off: a game
phase opts in with a ``"menu"`` or a ``"hint"`` dict, and the loop in
:mod:`humans.run` never knows which game they are for.

Phase fields::

    "menu": {"control": "X", "hold": 5.0, "after": 15.0,
             "options": ["reset", "forfeit", "resume"],
             "move": ["UP", "DOWN"], "confirm": "RETURN",
             "text": ["arrows up / down   choose", "return   select"]}
    "hint": {"control": "Y", "hold": 3.0, "after": 0.0,
             "text": ["![controls](images/controls/<backend>/<name>.png)", "", "..."]}

A menu's ``control`` is required; the rest default to the values shown, except
``text``: without it, the line under the options names ``move`` and ``confirm``.
A hint's ``control`` and ``text`` are required. Each control is a control name
(:mod:`humans.inputs`), as a game phase's ``controls`` are; a ``text`` is shown
as a message's is, image lines included. ``reset`` starts the episode over (a
new ``reset`` of the env with the same seed: the very instance the subject gave
up on), ``forfeit`` ends the block and moves on to the next phase, ``resume``
goes on where the game paused. Options are shown in the order listed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING

from .config.phases import HintSettings, MenuSettings
from .inputs import QUIT, ControlNames, Edge, InputEvent

if TYPE_CHECKING:
    from .outputs.display import Display
    from .save.logging import Logger

_LABELS = {"reset": "Restart the level", "forfeit": "Give up this level", "resume": "Resume"}


class PopUp(ABC):
    """A pop-up opened by holding one control: the hold that arms it, and its log.

    :param settings: its settings; ``control``, ``hold`` and ``after`` are read here.
    :param block_start: ``perf_counter`` at which the block began.
    :param held: returns the names of the controls held now (:meth:`InputDevice.held`).
    """

    def __init__(self, settings: MenuSettings | HintSettings, block_start: float,
                 held: Callable[[], frozenset[str]]) -> None:
        self.settings = settings
        self.available_from = block_start + settings.after
        self._held = held
        self._held_since: float | None = None
        self.pending = False        # the hold completed; the pop-up is due
        self.events: list[dict] = []  # one per pop-up: when, and how it closed

    def describe(self) -> dict:
        """The settings in force and every pop-up so far, for the manifest."""
        return {**asdict(self.settings), "events": self.events}

    def armed(self) -> bool:
        """Poll the hold; ``True`` (once, and :attr:`pending`) when it completes.

        Call often while the game runs. The hold only counts once the pop-up is
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

    @abstractmethod
    def run(self, display: Display, run_time: Callable[[], float], logger: Logger,
            read_input_events: Callable[[], list[InputEvent]]) -> str:
        """Show the pop-up until it closes, and say how it closed.

        :param display: the display to draw on.
        :param run_time: the run clock, for the log.
        :param logger: the run's logger; the presses meanwhile are logged like any other.
        :param read_input_events: the edges since it was last called (the run's).
        :return: ``"resume"``, another of the menu's options, or ``"quit"`` on
            window close / ESC.
        """


class Menu(PopUp):
    """One game block's pause menu.

    :param spec: the phase's ``"menu"`` dict (already validated).
    :param block_start: ``perf_counter`` at which the block began.
    :param held: returns the names of the controls held now (:meth:`InputDevice.held`).
    """

    def __init__(self, spec: dict, block_start: float,
                 held: Callable[[], frozenset[str]]) -> None:
        super().__init__(MenuSettings.from_dict(spec), block_start, held)

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
        self._draw(display, choice)
        while True:
            for name, edge in read_input_events():
                down = edge is Edge.DOWN
                if name in (QUIT, ControlNames.K_ESCAPE):
                    if down:
                        return "quit"
                    continue
                logger.log(type="input_event", run_time=run_time(), name=name, down=down)
                if not down:
                    continue
                if name in s.move:
                    choice = (choice + (1 if name == s.move[1] else -1)) % len(s.options)
                    self._draw(display, choice)
                elif name == s.confirm:
                    chosen = s.options[choice]
                    self.events.append({"opened": opened, "closed": run_time(), "choice": chosen})
                    return chosen
            display.idle(time.perf_counter() + 0.005, poll=0.005)

    def _draw(self, display: Display, choice: int) -> None:
        """Render the option list with ``choice`` marked, and how to choose under it."""
        s = self.settings
        rows = "\n".join(f"{'>' if i == choice else ' '}  {_LABELS[o]}"
                         for i, o in enumerate(s.options))
        how_to_choose = s.text
        if how_to_choose is None:
            how_to_choose = f"({s.move[0]}/{s.move[1]} to choose, {s.confirm} to select)"
        display.draw_text(f"Paused\n\n{rows}\n\n{how_to_choose}", align="left")


class Hint(PopUp):
    """One game block's hint: its text, shown while its control stays held.

    :param spec: the phase's ``"hint"`` dict (already validated).
    :param block_start: ``perf_counter`` at which the block began.
    :param held: returns the names of the controls held now (:meth:`InputDevice.held`).
    """

    def __init__(self, spec: dict, block_start: float,
                 held: Callable[[], frozenset[str]]) -> None:
        super().__init__(HintSettings.from_dict(spec), block_start, held)

    def run(self, display: Display, run_time: Callable[[], float], logger: Logger,
            read_input_events: Callable[[], list[InputEvent]]) -> str:
        """Show the text until the control is let go.

        :param display: the display to draw on.
        :param run_time: the run clock, for the log.
        :param logger: the run's logger; the presses meanwhile are logged like any other.
        :param read_input_events: the edges since it was last called (the run's).
        :return: ``"resume"``, or ``"quit"`` on window close / ESC.
        """
        self.pending, self._held_since = False, None
        opened = run_time()
        display.draw_text(self.settings.text, align="left")
        while self.settings.control in self._held():
            for name, edge in read_input_events():
                if name in (QUIT, ControlNames.K_ESCAPE):
                    if edge is Edge.DOWN:
                        return "quit"
                    continue
                logger.log(type="input_event", run_time=run_time(), name=name,
                           down=edge is Edge.DOWN)
            display.idle(time.perf_counter() + 0.005, poll=0.005)
        self.events.append({"opened": opened, "closed": run_time()})
        return "resume"
