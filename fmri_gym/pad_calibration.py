"""Measuring a stick's calibration: the procedure, and the fit that turns it into one.

A stick rests off centre, travels further one way than the other, and moves its
centre between sessions, so each is measured on the rig it is used on.
:func:`calibrate` measures it, waiting on the controller at every step, never on
a clock, and writes each stick's calibration as a new file
(:func:`fmri_gym.inputs.calibration.write`), the one a run then reads.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

import pygame

from .inputs.calibration import (DOWN, LEFT, RIGHT, UNCALIBRATED, UP,
                                 AxisCalibration, StickCalibration, write)
from .inputs.controller import (BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y, FACE_BUTTONS, Controller,
                                Stick, short_name)

#: The procedure's colours: the face buttons in the colour they are printed in,
#: so a cue can be matched by colour without reading it.
SURFACE, PANEL, GRID = (18, 21, 28), (24, 28, 37), (35, 41, 54)
INK, INK2, INK3 = (231, 234, 240), (154, 164, 178), (91, 100, 114)
LIVE, WARN = (63, 185, 80), (210, 153, 34)
FACE: dict[str, tuple[int, int, int]] = {
    BUTTON_Y: (226, 192, 68), BUTTON_B: (230, 86, 86), BUTTON_X: (79, 143, 247),
    BUTTON_A: (63, 185, 80),
}
#: Each direction of a stick as (its axis, 0 for x and 1 for y; the sign a push gives it).
PUSHES: dict[str, tuple[int, int]] = {UP: (1, -1), DOWN: (1, 1), LEFT: (0, -1), RIGHT: (0, 1)}
#: The order the procedure asks for the directions.
CUE_DIRECTIONS = (DOWN, LEFT, UP, RIGHT)


def cue_order(buttons: tuple[str, ...]) -> list[str]:
    """``buttons`` in the order the procedure asks for them: the face buttons first, round
    the face the way they sit under a thumb; the rest follow."""
    def rank(name: str) -> int:
        return FACE_BUTTONS.index(name) if name in FACE_BUTTONS else len(FACE_BUTTONS)
    return sorted(buttons, key=rank)


def columns_of(stick: Stick) -> tuple[str, str]:
    """The recording's columns for a stick's raw x and y."""
    return f"{stick.name}_x", f"{stick.name}_y"
#: Each control is asked for this many times in a row, so one slip is caught.
TWICE = 2
#: A stick is at rest once each axis has stayed within this, raw, for this long.
STILL_S, STILL_RANGE = 1.0, 0.02
#: A roll is done when each of this many directions round the edge has been
#: pushed this far, as a fraction of the travel seen that way, with no axis
#: reaching further since; and each side has travelled at least this far, raw.
ROLL_BINS, ROLL_REACH, ROLL_MIN = 12, 0.9, 0.3
#: A cued push counts past this fraction of its travel; the next waits for the stick under
#: the second.
PUSH, RELEASE = 0.5, 0.2
#: The thresholds a calibration writes, in fractions of travel; raised above ten times the dither.
CAL_ON, CAL_OFF = 0.5, 0.3


class CalibrationResult(NamedTuple):
    """The calibrations written, by stick, and how the controls answered."""

    #: each stick measured: its new file, and what it holds
    written: dict[str, tuple[Path, StickCalibration]]
    confirmed: int          # controls asked for and given, each twice in a row
    corrections: int        # wrong controls given, named and asked again

    def record(self) -> dict[str, Any]:
        """For the rig check's report."""
        return {"measured": True, "confirmed": self.confirmed, "corrections": self.corrections,
                "sticks": {name: {"config": str(path), "samples": str(path.with_suffix(".csv")),
                                  **cal.record()} for name, (path, cal) in self.written.items()}}


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

        :return: each stick's raw x and y, by :func:`columns_of`; and the buttons down.
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
        ax = {col: v for stick in self.controller.sticks
              for col, v in zip(columns_of(stick), (stick.x, stick.y))}
        bt = set(self.controller.held()) & set(self.buttons)
        row = {"t": round(self.now - self.t0, 4), "phase": phase, "cue": cue}
        row.update({col: round(v, 5) for col, v in ax.items()})
        row.update({f"b_{c}": int(c in bt) for c in self.buttons})
        self.samples.append(row)
        return ax, bt

    def draw(self, lines: list[str], colour=INK, sub: str = "", arrow: str | None = None,
             ring: tuple[int, set[int]] | None = None) -> None:
        """One screen: a word or two, an arrow or the roll's ring of directions, a line under."""
        self.screen.fill(SURFACE)
        if arrow:
            draw_arrow(self.screen, arrow, (self.mx, 235), 74, colour)
        if ring:
            n, lit = ring
            for b in range(n):
                angle = (b + 0.5) / n * 2 * math.pi - math.pi
                at = (self.mx + math.cos(angle) * 110, 235 + math.sin(angle) * 110)
                pygame.draw.circle(self.screen, LIVE if b in lit else GRID, at, 12)
        top = 360 if ring else 300 if arrow else 235
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


def _hold_still(task: _Task, columns: tuple[str, str], phase: str, cue: str,
                words: list[str]) -> dict[str, float]:
    """Wait until the stick has been still for :data:`STILL_S`. Returns where it rests."""
    window: deque = deque()
    while True:
        task.draw(words, INK2)
        ax, _ = task.frame(phase, cue)
        window.append((task.now, ax))
        while task.now - window[0][0] > STILL_S:
            window.popleft()
        if task.now - window[0][0] < 0.95 * STILL_S:
            continue
        spans = [max(a[c] for _, a in window) - min(a[c] for _, a in window) for c in columns]
        if max(spans) <= STILL_RANGE:
            return {c: statistics.median(a[c] for _, a in window) for c in columns}


def _roll(task: _Task, columns: tuple[str, str], cue: str, rest: dict[str, float],
          which: str) -> dict[str, tuple[float, float]]:
    """Wait until the stick has been rolled to its stop all the way round.

    :return: each axis's (min, max).
    """
    lo, hi, lit = dict(rest), dict(rest), set()
    while True:
        task.draw([f"roll {which}", "round its edge"], ring=(ROLL_BINS, lit),
                  sub="push it to its stop, all the way round, until every dot is lit")
        ax, _ = task.frame("roll", cue)
        for c in columns:
            if ax[c] < lo[c] - 0.01 or ax[c] > hi[c] + 0.01:
                lit.clear()                     # further than before: round again at it
            lo[c], hi[c] = min(lo[c], ax[c]), max(hi[c], ax[c])
        n = [_fraction(ax[c], rest[c], lo[c], hi[c]) for c in columns]
        if math.hypot(*n) >= ROLL_REACH:
            turn = (math.atan2(n[1], n[0]) + math.pi) / (2 * math.pi)
            lit.add(int(turn * ROLL_BINS) % ROLL_BINS)
        if len(lit) == ROLL_BINS and all(min(hi[c] - rest[c], rest[c] - lo[c]) >= ROLL_MIN
                                         for c in columns):
            return {c: (lo[c], hi[c]) for c in columns}


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
    while row < TWICE:
        task.draw([words, f"{row + 1} of {TWICE}"], colour, said, arrow)
        while not idle(*task.frame("idle", cue)):
            pass
        got = None
        while got is None:
            task.draw([words, f"{row + 1} of {TWICE}"], colour, said, arrow)
            got = answer(*task.frame("ask", cue))
        if got == cue:
            row, said = row + 1, ""
        else:
            wrong, row, said = wrong + 1, 0, f"that was {got}: again, {TWICE} in a row"
    return wrong


def _calibrate_stick(task: _Task, stick: Stick, which: str) -> int:
    """Rest, travel and directions of one stick. Returns the wrong answers."""
    columns = columns_of(stick)
    cold = _hold_still(task, columns, "cold", stick.name, ["hands off"])
    travel = _roll(task, columns, stick.name, cold, which)
    rest = _hold_still(task, columns, "rest", stick.name, ["let go", "hands off"])

    def pushes(ax: dict[str, float]) -> dict[str, float]:
        return {stick.control_of(d): _fraction(ax[columns[axis]], rest[columns[axis]],
                                               *travel[columns[axis]]) * sign
                for d, (axis, sign) in PUSHES.items()}

    def answer(ax: dict[str, float], _bt: set[str]) -> str | None:
        best = max(pushes(ax).items(), key=lambda kv: kv[1])
        return best[0] if best[1] >= PUSH else None

    def idle(ax: dict[str, float], _bt: set[str]) -> bool:
        return all(abs(v) < RELEASE for v in pushes(ax).values())

    return sum(_ask(task, stick.control_of(d), f"push {which}", INK, d, answer, idle)
               for d in CUE_DIRECTIONS)


def _ask_buttons(task: _Task) -> int:
    """Each button the pad has, asked for twice in a row. Returns the wrong answers."""
    return sum(_ask(task, name, f"press  {short_name(name)}", FACE.get(name, INK), None,
                    lambda _ax, bt: min(bt) if bt else None, lambda _ax, bt: not bt)
               for name in task.buttons)


def _measure(task: _Task, sticks: tuple[Stick, ...], has_buttons: bool) -> tuple[int, int]:
    """Each of ``sticks``, then each button if ``has_buttons``. Returns (confirmed, wrong)."""
    confirmed = wrong = 0
    for stick in sticks:
        which = (f"the {stick.name.replace('_', ' ').lower()}"
                 if len(task.controller.sticks) > 1 else "the stick")
        wrong += _calibrate_stick(task, stick, which)
        confirmed += len(CUE_DIRECTIONS)
    if has_buttons:
        wrong += _ask_buttons(task)
        confirmed += len(task.buttons)
    task.draw(["done"], LIVE)
    return confirmed, wrong


def calibrate(controller: Controller, screen: pygame.Surface, present: Callable[[], None],
              sticks: tuple[Stick, ...],
              to_canvas: Callable[[tuple[int, int]], tuple[int, int]] | None = None,
              ) -> CalibrationResult | None:
    """Measure ``sticks`` of ``controller`` on ``screen``, and write each a new calibration.

    Per stick: hands off until still, roll it round its edge until each direction
    has reached its stop, let go until still; then each of its directions asked
    for until it comes twice in a row, a wrong one named and asked again. With
    every stick the pad has, its buttons are asked for the same way. The
    controller goes on reading through the calibrations it had; the caller
    applies the new ones (:meth:`Controller.set_stick_calibration`).

    :param present: shows ``screen``.
    :param to_canvas: a window position -> ``screen``'s; given, the screen has a Cancel button.
    :return: what was written; None when ESC, Cancel or the window closing stopped
        it, with nothing written.
    :raises ValueError: for a stick the controller does not have.
    """
    unknown = [s.name for s in sticks if s not in controller.sticks]
    if unknown:
        raise ValueError(f"{controller.device_name!r} has no {' or '.join(unknown)}")
    task = _Task(controller, screen, present, to_canvas)
    try:
        confirmed, wrong = _measure(task, sticks, len(sticks) == len(controller.sticks))
    except _Quit:
        return None
    made = datetime.now(timezone.utc).replace(microsecond=0)
    written = {}
    for stick in sticks:
        rows = [r for r in task.samples if r["cue"] == stick.name]
        calibration = fit(rows, stick, controller.device_name, made)
        written[stick.name] = (write(calibration, rows), calibration)
    return CalibrationResult(written, confirmed, wrong)


# ------------------------------------------------------------ the fit
def _still_at(rows: list[dict], column: str) -> tuple[float, float]:
    """Where ``column`` rested over the last :data:`STILL_S` of ``rows``, and its dither."""
    if not rows:
        return 0.0, 0.0
    vals = [r[column] for r in rows if r["t"] >= rows[-1]["t"] - STILL_S]
    return statistics.median(vals), (max(vals) - min(vals)) / 2


def fit(rows: list[dict], stick: Stick, device_name: str, made: datetime) -> StickCalibration:
    """One stick's calibration from the frames that measured it: its axes' rest, stops and
    drift, and its thresholds.

    It presses at :data:`CAL_ON` of travel and releases under :data:`CAL_OFF`,
    raised to ten times the dither at rest for a stick that dithers.

    :param rows: the stick's frames of a calibration (raw readings), as
        :func:`calibrate` records them.
    :param made: when the measurement was made.
    """
    axes, dither = [], 0.0
    for column in columns_of(stick):
        cold, _ = _still_at([r for r in rows if r["phase"] == "cold"], column)
        rest, noise = _still_at([r for r in rows if r["phase"] == "rest"], column)
        col = [r[column] for r in rows]
        lo, hi = min(col), max(col)
        axes.append(AxisCalibration(round(rest, 5), round(lo, 5), round(hi, 5),
                                    round(abs(rest - cold), 5)))
        dither = max(dither, noise / max(min(hi - rest, rest - lo), 1e-3))
    on = round(min(max(CAL_ON, 10 * dither), 0.65), 3)
    tuning = replace(UNCALIBRATED, on=on, off=CAL_OFF if on == CAL_ON else round(0.6 * on, 3))
    return StickCalibration(device_name, stick.name, None, made, axes[0], axes[1], tuning)
