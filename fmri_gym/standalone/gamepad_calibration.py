"""``gamepad-calibration``: calibrate a controller, see what it sends, and tune how it is read.

A stick axis rests off centre, travels further one way than the other, and moves its centre
between sessions, so each is measured on the rig it is used on. :func:`calibrate` measures every
axis, waiting on the controller at every step, never on a clock, and writes the controller's
calibration as a new file (:func:`fmri_gym.inputs.calibration.write`), the one a run then reads.

The window shows what the controller plugged in sends and what its device
(:mod:`fmri_gym.inputs`) makes of it, through its newest calibration, and runs the calibration
in itself. Only its buttons act: the keyboard does nothing but ESC, since a program that turns a
pad into keystrokes would otherwise retune or save as the pad is pushed, and its keys cannot be
told from a person's. A tuning saved is a new file, never a change to the one read.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

import pygame

from ..config.rig import RigConfig
from ..inputs import (CONTROLLER_CLASSES_BY_NAME, CalibrationError, Controller,
                      first_controller_class_plugged_in)
from ..inputs.calibration import (AxisCalibration, ControllerCalibration, calibrations_dir,
                                  write)
from ..inputs.control_names import ControlNames
from ..inputs.controller import AXIS_CONTROLS, FACE_BUTTONS, short_name

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


#: The picture is drawn at this size and shown scaled to its window, which can
#: be any size from the second.
W, H = 1020, 880
MIN_W, MIN_H = 640, 552
#: Seconds of the plot, and between two looks for a changed calibration.
WINDOW_S, WATCH_S = 8.0, 1.0
AXIS_HUE = (79, 143, 247)
_PLOT_Y = 280                   # the top of the plot
_VALUE_W = 58                   # the room for a value between its - and +
#: Each tuning button: the field it moves, and by how much.
_TUNE_STEPS: dict[str, tuple[str, float]] = {
    "on-": ("on", -0.01), "on+": ("on", 0.01), "off-": ("off", -0.01), "off+": ("off", 0.01),
}


class Button(NamedTuple):
    """A button: what it says, the action it runs (:data:`_ACTIONS`), and where it sits."""

    label: str
    action: str
    rect: pygame.Rect


def _row(x: int, y: int, h: int, items: list[tuple[str, str]]) -> tuple[list[Button], int]:
    """Buttons side by side from ``x``, each as wide as its label. Returns them and the next x."""
    out = []
    for label, action in items:
        w = 22 + 7 * len(label)
        out.append(Button(label, action, pygame.Rect(x, y, w, h)))
        x += w + 6
    return out, x


def _words(axis_name: str) -> str:
    """An axis's name as a label: its short name, lower-cased."""
    return short_name(axis_name).lower()


def _layout(axis_names: tuple[str, ...]) -> tuple[Button, ...]:
    """The buttons for a pad with the axes ``axis_names``: the actions' row above the
    tuning's.

    In the tuning row, each value sits in the gap between its - and +.
    """
    top, _ = _row(20, 10, 30, [("Calibrate", "calibrate"), ("Save", "save"),
                               ("Reload", "reload"), ("Clear", "clear"), ("Quit", "quit")])
    shown, x = _row(20, 48, 26, [(_words(n), f"show-{n}") for n in axis_names])
    tuning = []
    for name in ("on", "off"):
        x += 20 + 7 * len(name)                     # its name, before the -
        tuning += [Button("-", f"{name}-", pygame.Rect(x, 48, 26, 26)),
                   Button("+", f"{name}+", pygame.Rect(x + 26 + _VALUE_W, 48, 26, 26))]
        x += 52 + _VALUE_W
    return (*top, *shown, *tuning)


def _retuned(axis: AxisCalibration, action: str) -> AxisCalibration:
    """An axis after one tuning button, kept within what a calibration file allows."""
    name, step = _TUNE_STEPS[action]
    low, high = {"on": (axis.off, 0.99), "off": (0.01, axis.on)}[name]
    return replace(axis, **{name: round(min(high, max(low, getattr(axis, name) + step)), 3)})


def _mtime(path: Path | None) -> float:
    return path.stat().st_mtime if path is not None and path.exists() else 0.0


def _watched(controller: Controller | None) -> tuple[float, float]:
    """When the calibrations' folder last changed, and the file ``controller`` reads."""
    return _mtime(calibrations_dir()), _mtime(controller and controller.calibration_path)


class _Canvas:
    """A picture of a fixed size, shown scaled to fit a window the user can resize."""

    def __init__(self, caption: str) -> None:
        pygame.init()
        desk = pygame.display.get_desktop_sizes()[0]
        scale = min(1.0, 0.9 * desk[0] / W, 0.9 * desk[1] / H)
        self.window = pygame.Window(caption, (int(W * scale), int(H * scale)), resizable=True)
        self.window.minimum_size = (MIN_W, MIN_H)
        self.surface = pygame.Surface((W, H))

    def _fit(self) -> tuple[float, int, int]:
        """The picture's scale in the window, and where its top left corner sits."""
        ww, wh = self.window.size
        s = min(ww / W, wh / H)
        return s, (ww - int(W * s)) // 2, (wh - int(H * s)) // 2

    def present(self) -> None:
        """Show the picture, scaled, with bars where the window's shape is not its own."""
        s, x, y = self._fit()
        out = self.window.get_surface()
        out.fill((0, 0, 0))
        out.blit(pygame.transform.smoothscale(self.surface, (int(W * s), int(H * s))), (x, y))
        self.window.flip()

    def to_canvas(self, pos: tuple[int, int]) -> tuple[int, int]:
        """A position in the window as one in the picture."""
        s, x, y = self._fit()
        return int((pos[0] - x) / s), int((pos[1] - y) / s)


def _wrapped(text: str, font: pygame.font.Font, width: int, lines: int) -> list[str]:
    """``text`` broken at spaces into at most ``lines`` lines of ``width`` pixels, the last cut."""
    out, line = [], ""
    for word in text.split(" "):
        if line and font.size(f"{line} {word}")[0] > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    if len(out) > lines:
        out = out[:lines]
        while font.size(out[-1] + " ...")[0] > width:
            out[-1] = out[-1][:-1]
        out[-1] += " ..."
    return out


def _vy(v: float, cy: int, half: int) -> int:
    """A value as a y pixel on the scrolling plot."""
    return int(cy - max(-1.0, min(1.0, v)) * half)


def _tx(t: float, now: float, px: int, pw: int) -> int:
    """A timestamp as an x pixel, newest at the right edge."""
    return int(px + pw - (now - t) / WINDOW_S * pw)


def _draw_tiles(screen: pygame.Surface, font: pygame.font.Font, controller: Controller | None,
                down: frozenset[str], by: int) -> None:
    """One tile per button the controller has."""
    names = controller.BUTTON_NAMES if controller else ()
    cols, bx, gap = 7, 20, 4
    bw = (W - 2 * bx - (cols - 1) * gap) // cols
    for k, name in enumerate(names):
        x, y = bx + (k % cols) * (bw + gap), by + (k // cols) * (62 + gap)
        is_down, hue = name in down, FACE.get(name, INK2)
        pygame.draw.rect(screen, hue if is_down else PANEL, (x, y, bw, 62), border_radius=4)
        pygame.draw.rect(screen, hue if is_down else GRID, (x, y, bw, 62), 1, border_radius=4)
        screen.blit(font.render(short_name(name), True, SURFACE if is_down else hue),
                    (x + 8, y + 6))


class _Scope:
    """The tool's window: what the controller sends, what is made of it, and the buttons."""

    def __init__(self, kinds: tuple[str, ...]) -> None:
        """Open the window, and the first controller plugged in of ``kinds``.

        :param kinds: the kinds of controller to look for
            (:data:`fmri_gym.inputs.CONTROLLER_CLASSES_BY_NAME`).
        """
        self.canvas = _Canvas("gamepad-calibration")
        font = "Menlo,Monaco,monospace"
        self.f_big, self.f, self.f_sm = (pygame.font.SysFont(font, n) for n in (17, 13, 11))
        self.kinds = kinds
        self.controller: Controller | None = None
        self.axis: str | None = None                # the axis shown
        self.loaded: ControllerCalibration | None = None   # the calibration as its file has it
        self.flash, self.flash_until = "", 0.0
        self.hist: deque = deque()
        self.log: deque = deque(maxlen=11)
        self.prev: frozenset = frozenset()
        self.running, self.t0 = True, time.perf_counter()
        self.reload("")
        self.next_watch = self.now() + WATCH_S

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def say(self, text: str, seconds: float = 4.0) -> None:
        self.flash, self.flash_until = text, self.now() + seconds

    def reload(self, said: str) -> None:
        """Open the controller plugged in again, through its newest calibration or the error
        it gives."""
        cls = first_controller_class_plugged_in(self.kinds)
        self.controller, self.error = None, ""
        if cls is not None:
            self.controller = cls.open(calibrated=False)
            try:
                self.controller.load_calibration()
            except CalibrationError as exc:
                self.error = str(exc)
        self.loaded = self.controller.calibration if self.controller else None
        self.status = "" if self.controller is None else self.controller.status()
        axis_names = self.controller.AXIS_NAMES if self.controller else ()
        self.buttons = _layout(axis_names)
        self.axis = self.axis if self.axis in axis_names else next(iter(axis_names), None)
        self.watched = _watched(self.controller)
        self.hist.clear()
        if said:
            self.say(said)

    def watch(self, now: float) -> None:
        """Every :data:`WATCH_S`, reload if a calibration was added or the one in use changed."""
        if now < self.next_watch:
            return
        self.next_watch = now + WATCH_S
        if _watched(self.controller) != self.watched:
            self.reload("a calibration changed: reloaded")

    def calibrate(self) -> None:
        """Run the calibration of every axis and button, in this window."""
        result = calibrate(self.controller, self.canvas.surface,
                           self.canvas.present, self.canvas.to_canvas)
        self.reload("NOT calibrated: cancelled; nothing written" if result is None
                    else f"calibrated: {result.path.name}")

    def save(self) -> None:
        """Write the retuned calibration as a new file."""
        source = self.controller.calibration_path.name
        tuned = replace(self.controller.calibration, tuned_from=source)
        self.reload(f"saved {write(tuned).name} from {source}")

    def is_retuned(self) -> bool:
        """Whether the calibration in use differs from its file's."""
        return self.controller is not None and self.controller.calibration != self.loaded

    def clear(self) -> None:
        self.hist.clear()
        self.log.clear()

    def show(self, axis_name: str) -> None:
        self.axis = axis_name
        self.hist.clear()

    def retune(self, action: str) -> None:
        cal = self.controller.calibration
        axes = {**cal.axes, self.axis: _retuned(cal.axes[self.axis], action)}
        self.controller.calibration = replace(cal, axes=axes)

    def is_enabled(self, action: str) -> bool:
        """Whether a button can do anything now."""
        if action == "calibrate":
            return self.controller is not None and not self.error
        if action == "save":
            return self.is_retuned()
        if action.startswith("show"):
            return self.axis is not None
        if action in _TUNE_STEPS:
            return self.controller is not None and self.controller.calibration is not None
        return True

    def events(self) -> None:
        """Clicks on the buttons, and ESC or the window's close to quit; the controller reads the
        rest."""
        events = pygame.event.get()
        if self.controller is not None:
            self.controller.handle(events, time.perf_counter())
        for e in events:
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self.click(self.canvas.to_canvas(e.pos))

    def click(self, at: tuple[int, int]) -> None:
        """Run the button under ``at`` (a position in the picture), if it can act."""
        for button in self.buttons:
            if button.rect.collidepoint(at) and self.is_enabled(button.action):
                _ACTIONS[button.action](self)

    def loop(self) -> None:
        clock = pygame.time.Clock()
        while self.running:
            self.events()
            self.watch(self.now())
            self.draw()
            clock.tick(60)
        pygame.quit()

    def _shown_axis(self) -> AxisCalibration | None:
        """The shown axis's calibration in use, or None uncalibrated."""
        cal = self.controller.calibration if self.controller else None
        return None if cal is None or self.axis is None else cal.axes[self.axis]

    def _shown_value(self) -> float:
        """The shown axis: calibrated, or raw when uncalibrated."""
        if self.axis is None:
            return 0.0
        value = self.controller.calibrated_value(self.axis)
        return self.controller.axis_values[self.axis] if value is None else value

    def record(self, now: float, controls: frozenset) -> None:
        """Log the presses and releases since the last frame, and add this frame to the plot."""
        for name in sorted(controls - self.prev):
            self.log.appendleft((now, f"{name:<24} pressed", LIVE))
        for name in sorted(self.prev - controls):
            self.log.appendleft((now, f"{name:<24} released", INK3))
        self.prev = controls
        self.hist.append((now, self._shown_value(), controls))
        while self.hist and now - self.hist[0][0] > WINDOW_S:
            self.hist.popleft()

    def draw(self) -> None:
        now, controller = self.now(), self.controller
        controls = controller.held() if controller else frozenset()
        self.record(now, controls)
        screen = self.canvas.surface
        screen.fill(SURFACE)
        self.draw_toolbar(screen)
        self.draw_status(screen, now)
        _draw_tiles(screen, self.f_big, controller, controls, 136)
        self.draw_plot(screen, now)
        lanes = self.draw_lanes(screen, controls, now)
        self.draw_log(screen, _PLOT_Y + 200 + 10 + lanes * 16 + 14)
        if controller is None:
            kinds = " or ".join(self.kinds) or "none: the rig lists no controller"
            screen.blit(self.f_big.render(f"no controller plugged in ({kinds})", True, WARN),
                        (20, _PLOT_Y + 230))
        self.canvas.present()

    def draw_status(self, screen: pygame.Surface, now: float) -> None:
        """The controller's status, the calibration in use or why there is none, and the
        last message."""
        path = self.controller.calibration_path if self.controller else None
        cal = self.error or (f"calibration in use: {path.name}" if path else
                             "UNCALIBRATED: no calibration of this pad; Calibrate makes one")
        for y, text, colour in ((84, self.status, INK2),
                                (98, cal, WARN if self.error or not path else INK2)):
            line = _wrapped(text, self.f_sm, W - 40, 1)[0]
            screen.blit(self.f_sm.render(line, True, colour), (20, y))
        if self.flash and now < self.flash_until:
            screen.blit(self.f.render(_wrapped(self.flash, self.f, W - 40, 1)[0], True,
                                      WARN if self.flash.startswith("NOT") else LIVE), (20, 114))

    def draw_toolbar(self, screen: pygame.Surface) -> None:
        mouse = self.canvas.to_canvas(pygame.mouse.get_pos())
        axis = self._shown_axis()
        values = {"on": f"{axis.on:.2f}" if axis else "--", "off": f"{axis.off:.2f}" if axis
                  else "--"}
        for label, action, rect in self.buttons:
            draw_button(screen, self.f_sm, rect, label, rect.collidepoint(mouse),
                        self.is_enabled(action), action == f"show-{self.axis}")
            if action[:-1] in values and action.endswith("-"):   # its name, and its value after
                name = self.f_sm.render(action[:-1], True, INK2)
                screen.blit(name, name.get_rect(midright=(rect.x - 6, rect.centery)))
                value = self.f.render(values[action[:-1]], True, INK)
                screen.blit(value, value.get_rect(center=(rect.right + _VALUE_W // 2,
                                                          rect.centery)))

    def draw_plot(self, screen: pygame.Surface, now: float) -> None:
        """The shown axis over time, against its thresholds, and its numbers."""
        axis, hist = self._shown_axis(), self.hist
        px, py, pw, ph = 20, _PLOT_Y, W - 40, 200
        cy, half = py + ph // 2, ph // 2
        pygame.draw.rect(screen, PANEL, (px, py, pw, ph))
        for lvl in (-1.0, -0.5, 0.5, 1.0):
            pygame.draw.line(screen, GRID, (px, _vy(lvl, cy, half)), (px + pw, _vy(lvl, cy, half)))
            screen.blit(self.f_sm.render(f"{lvl:+.1f}", True, INK3),
                        (px + 3, _vy(lvl, cy, half) - 12))
        pygame.draw.line(screen, (60, 68, 86), (px, cy), (px + pw, cy))
        for sgn in ((1, -1) if axis else ()):
            y_on, y_off = _vy(sgn * axis.on, cy, half), _vy(sgn * axis.off, cy, half)
            pygame.draw.line(screen, (120, 130, 150), (px, y_on), (px + pw, y_on))
            for seg in range(px, px + pw, 10):
                pygame.draw.line(screen, (80, 88, 106), (seg, y_off), (seg + 5, y_off))
        if len(hist) > 1:
            pygame.draw.lines(screen, AXIS_HUE, False, [(_tx(h[0], now, px, pw),
                                                         _vy(h[1], cy, half)) for h in hist], 2)
        screen.blit(self.f.render(_words(self.axis) if self.axis else "no axis", True, AXIS_HUE),
                    (px + 8, py + 6))
        screen.blit(self.f_sm.render(self._numbers(axis), True, INK3), (px + 160, py + 8))

    def _numbers(self, axis: AxisCalibration | None) -> str:
        """The shown axis's reading, raw and calibrated, and where it rests."""
        if self.axis is None:
            return ""
        raw = self.controller.axis_values[self.axis]
        if axis is None:
            return f"raw {raw:+.3f} (uncalibrated)"
        return (f"calibrated {self._shown_value():+.3f}   raw {raw:+.3f}   "
                f"rest {axis.rest:+.3f}")

    def draw_lanes(self, screen: pygame.Surface, controls: frozenset, now: float) -> int:
        """A lane per control of each axis, lit while it was pressed. Returns how many."""
        px, pw, hist = 20, W - 40, self.hist
        names = self.controller.axis_controls() if self.controller else ()
        for r, name in enumerate(names):
            y = _PLOT_Y + 210 + r * 16
            screen.blit(self.f_sm.render(name, True, LIVE if name in controls else INK3), (px, y))
            pygame.draw.rect(screen, PANEL, (px + 130, y + 2, pw - 130, 10))
            for a, b in zip(hist, list(hist)[1:]):
                if name in b[2]:
                    x1, x2 = _tx(a[0], now, px, pw), _tx(b[0], now, px, pw)
                    pygame.draw.rect(screen, LIVE, (max(x1, px + 130), y + 2, max(1, x2 - x1), 10))
        return len(names)

    def draw_log(self, screen: pygame.Surface, ly: int) -> None:
        screen.blit(self.f.render("events", True, INK2), (20, ly))
        for i, (t, text, colour) in enumerate(self.log):
            screen.blit(self.f_sm.render(f"{t:7.2f}s  {text}", True, colour),
                        (20, ly + 20 + i * 14))
        screen.blit(self.f_sm.render("the buttons along the top do everything; ESC quits",
                                     True, INK3), (20, H - 22))


#: What each button does.
_ACTIONS: dict[str, Callable[[_Scope], None]] = {
    "calibrate": _Scope.calibrate,
    **{f"show-{n}": (lambda s, n=n: s.show(n)) for n in AXIS_CONTROLS},
    **{a: (lambda s, a=a: s.retune(a)) for a in _TUNE_STEPS},
    "save": _Scope.save, "reload": lambda s: s.reload("reloaded"), "clear": _Scope.clear,
    "quit": lambda s: setattr(s, "running", False),
}


def main() -> None:
    """``gamepad-calibration [--rig <rig file>]``."""
    p = argparse.ArgumentParser(
        description="Calibrate the controller plugged in, see what it sends and what fmri-gym "
                    "makes of it, and tune how its stick axes are read, with the buttons along "
                    "the top. Calibrations are kept in " + os.fspath(calibrations_dir()))
    p.add_argument("--rig", default=None, metavar="RIG.json",
                   help='the rig file whose "inputs" to look for (default: any controller)')
    rig = p.parse_args().rig
    input_device_names = RigConfig.open(rig).inputs if rig else tuple(CONTROLLER_CLASSES_BY_NAME)
    _Scope(tuple(n for n in input_device_names if n in CONTROLLER_CLASSES_BY_NAME)).loop()


if __name__ == "__main__":
    main()