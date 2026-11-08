"""Measuring a controller's calibration: the procedure, and the fit that turns it into one.

A stick axis rests off centre, travels further one way than the other, and moves
its centre between sessions, so each is measured on the rig it is used on.
:func:`calibrate` measures every axis, waiting on the controller at every step,
never on a clock, and writes the controller's calibration as a new file
(:func:`fmri_gym.inputs.calibration.write`), the one a run then reads.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

import pygame

from .inputs.calibration import AxisCalibration, ControllerCalibration, write
from .inputs.control_names import ControlNames
from .inputs.controller import AXIS_CONTROLS, FACE_BUTTONS, Controller, short_name

#: The procedure's colours: the face buttons in the colour they are printed in,
#: so a cue can be matched by colour without reading it.
SURFACE, PANEL, GRID = (18, 21, 28), (24, 28, 37), (35, 41, 54)
INK, INK2, INK3 = (231, 234, 240), (154, 164, 178), (91, 100, 114)
LIVE, WARN = (63, 185, 80), (210, 153, 34)
FACE: dict[str, tuple[int, int, int]] = {
    ControlNames.CONTROLLER_BUTTON_Y: (226, 192, 68),
    ControlNames.CONTROLLER_BUTTON_B: (230, 86, 86),
    ControlNames.CONTROLLER_BUTTON_X: (79, 143, 247),
    ControlNames.CONTROLLER_BUTTON_A: (63, 185, 80),
}


def cue_order(buttons: tuple[str, ...]) -> list[str]:
    """``buttons`` in the order the procedure asks for them: the face buttons first, round
    the face the way they sit under a thumb; the rest follow."""
    def rank(name: str) -> int:
        return FACE_BUTTONS.index(name) if name in FACE_BUTTONS else len(FACE_BUTTONS)
    return sorted(buttons, key=rank)


def arrow_of(control: str) -> str:
    """The arrow (UP, DOWN, LEFT, RIGHT) of one of :data:`AXIS_CONTROLS`' controls."""
    return control.rsplit("_", 1)[1]


#: Each control is asked for this many times in a row, so one slip is caught.
TWICE = 2
#: An axis is at rest once each has stayed within this, raw, for this long.
STILL_S, STILL_RANGE = 1.0, 0.02
#: An axis has been pushed to its stops when each side has reached this fraction of
#: the travel seen that way, with no further reach since; and has travelled this far, raw.
STOP_REACH, STOP_MIN = 0.9, 0.3
#: A cued push counts past this fraction of its travel; the next waits for every axis under
#: the second.
PUSH, RELEASE = 0.5, 0.2
#: The thresholds a calibration writes, in fractions of travel; raised above ten times the dither.
CAL_ON, CAL_OFF = 0.5, 0.3


class CalibrationResult(NamedTuple):
    """The calibration written, and how the controls answered."""

    path: Path
    calibration: ControllerCalibration
    confirmed: int          # controls asked for and given, each twice in a row
    corrections: int        # wrong controls given, named and asked again

    def record(self) -> dict[str, Any]:
        """For the rig measurement's report."""
        return {"measured": True, "confirmed": self.confirmed, "corrections": self.corrections,
                "config": str(self.path), "samples": str(self.path.with_suffix(".csv")),
                "axes": {name: axis.record() for name, axis in self.calibration.axes.items()}}


# ------------------------------------------------------------ the procedure
def draw_arrow(surface: pygame.Surface, direction: str, at: tuple[int, int], size: int,
               colour) -> None:
    """Draw a direction (UP, DOWN, LEFT, RIGHT) as the arrow it is."""
    dx, dy = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}[direction]
    px, py = -dy, dx                                    # across the shaft
    cx, cy = at
    tip = (cx + dx * size, cy + dy * size)
    base = (cx - dx * size * 0.4, cy - dy * size * 0.4)
    neck = (cx + dx * size * 0.15, cy + dy * size * 0.15)
    pygame.draw.polygon(surface, colour, [
        tip,
        (neck[0] + px * size * 0.55, neck[1] + py * size * 0.55),
        (neck[0] + px * size * 0.2, neck[1] + py * size * 0.2),
        (base[0] + px * size * 0.2, base[1] + py * size * 0.2),
        (base[0] - px * size * 0.2, base[1] - py * size * 0.2),
        (neck[0] - px * size * 0.2, neck[1] - py * size * 0.2),
        (neck[0] - px * size * 0.55, neck[1] - py * size * 0.55),
    ])


def draw_button(surface: pygame.Surface, font: pygame.font.Font, rect: pygame.Rect, label: str,
                hover: bool, enabled: bool = True, on: bool = False) -> None:
    """A button: its label in a box, lit when on, brighter under the mouse, dim when disabled."""
    fill = LIVE if on else (GRID if hover and enabled else PANEL)
    pygame.draw.rect(surface, fill, rect, border_radius=5)
    pygame.draw.rect(surface, INK3 if enabled else GRID, rect, 1, border_radius=5)
    text = font.render(label, True, SURFACE if on else (INK if enabled else INK3))
    surface.blit(text, text.get_rect(center=rect.center))


class _Quit(Exception):
    """ESC, Cancel, or the window closed: the procedure stops, and writes nothing."""


class _FrameClock:
    """Paces the procedure at 60 frames a second, and tells the time."""

    def __init__(self) -> None:
        self._clock = pygame.time.Clock()

    def tick(self) -> float:
        """Wait for the next frame. Returns the time now, in seconds."""
        self._clock.tick(60)
        return time.perf_counter()


class _Task:
    """The procedure's screen and its recording: one row per frame.

    It draws on ``screen`` and shows it with ``present``. Given ``to_canvas`` (a
    window position -> ``screen``'s), it draws a Cancel button that clicks stop it.
    """

    def __init__(self, controller: Controller, screen: pygame.Surface,
                 present: Callable[[], None],
                 to_canvas: Callable[[tuple[int, int]], tuple[int, int]] | None) -> None:
        self.controller, self.screen, self.present = controller, screen, present
        self.to_canvas = to_canvas
        w, self.h = screen.get_size()
        self.mx, scale = w // 2, min(2.0, max(0.8, self.h / 600))
        font = "Menlo,Monaco,monospace"
        self.big, self.mid, self.sml = (pygame.font.SysFont(font, int(n * scale))
                                        for n in (62, 25, 15))
        self.cancel = None if to_canvas is None else pygame.Rect(w - 150, self.h - 60, 130, 40)
        self.clock = _FrameClock()
        self.buttons = cue_order(controller.BUTTON_NAMES)
        self.samples: list[dict] = []
        self.t0 = self.now = self.clock.tick()

    def frame(self, phase: str, cue: str) -> tuple[dict[str, float], set[str]]:
        """Wait for the next frame, hand the controller its events and record its reading.

        :return: each axis's raw reading, by its name; and the buttons down.
        :raises _Quit: on ESC, a click on Cancel, or the window closing (which is
            left on the queue for the window's owner, when it has a Cancel).
        """
        self.now = self.clock.tick()
        events = pygame.event.get()
        self.controller.handle(events, self.now)
        for e in events:
            if e.type == pygame.QUIT:
                if self.cancel:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                raise _Quit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                raise _Quit
            if e.type == pygame.MOUSEBUTTONDOWN and self.cancel \
                    and self.cancel.collidepoint(self.to_canvas(e.pos)):
                raise _Quit
        ax = dict(self.controller.axis_values)
        bt = set(self.controller.held()) & set(self.buttons)
        row = {"t": round(self.now - self.t0, 4), "phase": phase, "cue": cue}
        row.update({name: round(v, 5) for name, v in ax.items()})
        row.update({f"b_{c}": int(c in bt) for c in self.buttons})
        self.samples.append(row)
        return ax, bt

    def draw(self, lines: list[str], colour=INK, sub: str = "",
             arrows: tuple[tuple[str, Any], ...] = ()) -> None:
        """One screen: arrows side by side (a direction and its colour each), a word or two,
        a line under."""
        self.screen.fill(SURFACE)
        for k, (direction, arrow_colour) in enumerate(arrows):
            at = (int(self.mx + (k - (len(arrows) - 1) / 2) * 200), 235)
            draw_arrow(self.screen, direction, at, 74, arrow_colour)
        top = 300 if arrows else 235
        for k, line in enumerate(lines):
            font, ink = (self.big, colour) if k == 0 else (self.mid, INK2)
            surf = font.render(line, True, ink)
            self.screen.blit(surf, surf.get_rect(center=(self.mx, top + k * 66)))
        s = self.sml.render(sub or ("Cancel or ESC stops; nothing is written" if self.cancel
                                    else "ESC stops; nothing is written"), True, INK2)
        self.screen.blit(s, s.get_rect(center=(self.mx, self.h - 40)))
        if self.cancel:
            draw_button(self.screen, self.sml, self.cancel, "Cancel", hover=False)
        self.present()


def _stick_words(controller: Controller, axis_name: str) -> str:
    """The stick an axis is on, as a cue says it: ``the stick``, or ``the left stick`` on a
    pad with two."""
    sticks = {short_name(name)[:-1] for name in controller.AXIS_NAMES}
    stick = short_name(axis_name)[:-1]
    return f"the {stick.lower()} stick" if len(sticks) > 1 else "the stick"


def _hold_still(task: _Task, phase: str, words: list[str]) -> dict[str, float]:
    """Wait until every axis has been still for :data:`STILL_S`. Returns where each rests."""
    window: deque = deque()
    while True:
        task.draw(words, INK2)
        ax, _ = task.frame(phase, "")
        window.append((task.now, ax))
        while task.now - window[0][0] > STILL_S:
            window.popleft()
        if task.now - window[0][0] < 0.95 * STILL_S:
            continue
        spans = [max(a[c] for _, a in window) - min(a[c] for _, a in window) for c in ax]
        if max(spans, default=0.0) <= STILL_RANGE:
            return {c: statistics.median(a[c] for _, a in window) for c in ax}


def _push_to_stops(task: _Task, axis_name: str, rest: float) -> tuple[float, float]:
    """Wait until the axis has been pushed to its stop each way.

    :return: its (min, max), raw.
    """
    negative, positive = AXIS_CONTROLS[axis_name]
    lo, hi, reached = rest, rest, set()
    words = [f"push {_stick_words(task.controller, axis_name)}",
             f"{arrow_of(negative).lower()} and {arrow_of(positive).lower()}"]
    while True:
        arrows = tuple((arrow_of(c), LIVE if c in reached else GRID) for c in (negative, positive))
        task.draw(words, sub="push it to its stop each way, until both arrows are lit",
                  arrows=arrows)
        ax, _ = task.frame("stops", axis_name)
        value = ax[axis_name]
        if value < lo - 0.01 or value > hi + 0.01:
            reached.clear()                     # further than before: to each stop again
        lo, hi = min(lo, value), max(hi, value)
        pushed = _fraction(value, rest, lo, hi)
        if pushed <= -STOP_REACH:
            reached.add(negative)
        if pushed >= STOP_REACH:
            reached.add(positive)
        if len(reached) == 2 and min(hi - rest, rest - lo) >= STOP_MIN:
            return lo, hi


def _fraction(value: float, rest: float, lo: float, hi: float) -> float:
    """A reading as a fraction of the travel seen on its side of rest, -1..1."""
    span = (hi - rest) if value >= rest else (rest - lo)
    return 0.0 if span <= 1e-6 else max(-1.0, min(1.0, (value - rest) / span))


def _ask(task: _Task, cue: str, words: str, colour, arrow: str | None, answer, idle) -> int:
    """Ask for ``cue`` until it comes :data:`TWICE` in a row. Returns the wrong answers.

    :param answer: the reading -> the control it answers, or None.
    :param idle: the reading -> whether the pad is let go of, which a next answer waits for.
    """
    wrong, row, said = 0, 0, ""
    arrows = ((arrow, colour),) if arrow else ()
    while row < TWICE:
        task.draw([words, f"{row + 1} of {TWICE}"], colour, said, arrows)
        while not idle(*task.frame("idle", cue)):
            pass
        got = None
        while got is None:
            task.draw([words, f"{row + 1} of {TWICE}"], colour, said, arrows)
            got = answer(*task.frame("ask", cue))
        if got == cue:
            row, said = row + 1, ""
        else:
            wrong, row, said = wrong + 1, 0, f"that was {got}: again, {TWICE} in a row"
    return wrong


def _ask_axis_controls(task: _Task, rest: dict[str, float],
                       travel: dict[str, tuple[float, float]]) -> int:
    """Each control of each axis, asked for twice in a row. Returns the wrong answers.

    The answer is the side pushed furthest of every axis, so a push that moves the
    wrong axis further than the one asked for is named.
    """
    def pushes(ax: dict[str, float]) -> dict[str, float]:
        out = {}
        for name, (negative, positive) in ((n, AXIS_CONTROLS[n]) for n in rest):
            pushed = _fraction(ax[name], rest[name], *travel[name])
            out[negative], out[positive] = -pushed, pushed
        return out

    def answer(ax: dict[str, float], _bt: set[str]) -> str | None:
        best = max(pushes(ax).items(), key=lambda kv: kv[1])
        return best[0] if best[1] >= PUSH else None

    def idle(ax: dict[str, float], _bt: set[str]) -> bool:
        return all(abs(v) < RELEASE for v in pushes(ax).values())

    return sum(_ask(task, control, f"push {_stick_words(task.controller, name)}", INK,
                    arrow_of(control), answer, idle)
               for name in rest for control in AXIS_CONTROLS[name])


def _ask_buttons(task: _Task) -> int:
    """Each button the pad has, asked for twice in a row. Returns the wrong answers."""
    return sum(_ask(task, name, f"press  {short_name(name)}", FACE.get(name, INK), None,
                    lambda _ax, bt: min(bt) if bt else None, lambda _ax, bt: not bt)
               for name in task.buttons)


def _measure(task: _Task) -> tuple[int, int]:
    """Every axis, then each of its controls, then each button. Returns (confirmed, wrong)."""
    cold = _hold_still(task, "cold", ["hands off"])
    travel = {name: _push_to_stops(task, name, cold[name]) for name in task.controller.AXIS_NAMES}
    rest = _hold_still(task, "rest", ["let go", "hands off"])
    wrong = _ask_axis_controls(task, rest, travel) + _ask_buttons(task)
    task.draw(["done"], LIVE)
    return len(task.controller.axis_controls()) + len(task.buttons), wrong


def calibrate(controller: Controller, screen: pygame.Surface, present: Callable[[], None],
              to_canvas: Callable[[tuple[int, int]], tuple[int, int]] | None = None,
              ) -> CalibrationResult | None:
    """Measure every axis of ``controller`` on ``screen``, and write its new calibration.

    Hands off until every axis is still; each axis pushed to its stop each way;
    let go until still. Then each control of each axis, and each button, asked
    for until it comes twice in a row, a wrong one named and asked again. The
    controller goes on reading through the calibration it had; the caller
    applies the new one (:meth:`Controller.set_calibration`).

    :param present: shows ``screen``.
    :param to_canvas: a window position -> ``screen``'s; given, the screen has a Cancel button.
    :return: what was written; None when ESC, Cancel or the window closing stopped
        it, with nothing written.
    """
    task = _Task(controller, screen, present, to_canvas)
    try:
        confirmed, wrong = _measure(task)
    except _Quit:
        return None
    made = datetime.now(timezone.utc).replace(microsecond=0)
    calibration = fit(task.samples, controller.AXIS_NAMES, controller.device_name, made)
    return CalibrationResult(write(calibration, task.samples), calibration, confirmed, wrong)


# ------------------------------------------------------------ the fit
def _still_at(rows: list[dict], column: str) -> tuple[float, float]:
    """Where ``column`` rested over the last :data:`STILL_S` of ``rows``, and its dither."""
    if not rows:
        return 0.0, 0.0
    vals = [r[column] for r in rows if r["t"] >= rows[-1]["t"] - STILL_S]
    return statistics.median(vals), (max(vals) - min(vals)) / 2


def fit(rows: list[dict], axis_names: tuple[str, ...], device_name: str,
        made: datetime) -> ControllerCalibration:
    """A controller's calibration from the frames that measured it: each axis's rest, stops,
    drift and thresholds.

    An axis presses at :data:`CAL_ON` of its travel and releases under :data:`CAL_OFF`,
    raised to ten times its dither at rest for an axis that dithers.

    :param rows: the frames of a calibration (raw readings), as :func:`calibrate` records them.
    :param made: when the measurement was made.
    """
    axes = {}
    for name in axis_names:
        cold, _ = _still_at([r for r in rows if r["phase"] == "cold"], name)
        rest, noise = _still_at([r for r in rows if r["phase"] == "rest"], name)
        lo, hi = min(r[name] for r in rows), max(r[name] for r in rows)
        dither = noise / max(min(hi - rest, rest - lo), 1e-3)
        on = round(min(max(CAL_ON, 10 * dither), 0.65), 3)
        off = CAL_OFF if on == CAL_ON else round(0.6 * on, 3)
        axes[name] = AxisCalibration(round(rest, 5), round(lo, 5), round(hi, 5),
                                     round(abs(rest - cold), 5), on, off)
    return ControllerCalibration(device_name, None, made, axes)
