"""``fmri-pad``: calibrate a controller, see what it sends, and tune how it is read.

It shows what the pad plugged in sends and what :mod:`fmri_gym.pad` makes of
it, through the newest calibration of it (:mod:`fmri_gym.pad_calibration`), and
runs the calibration in its own window. Only its buttons act: the keyboard does
nothing but ESC, since a program that turns a pad into keystrokes would
otherwise retune or save as the pad is pushed, and its keys cannot be told from
a person's. A tuning saved is a new file, never a change to the one read.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Callable, NamedTuple

import pygame

from . import pad as pads
from . import pad_calibration as calibration
from .config.rig import RigConfig
from .pad import (MAX_DWELL_S, RIGHT_STICK, STICK, STICK_DIRECTIONS, CalibrationError, Pad,
                  Stick, StickTuning)
from .pad_calibration import (FACE, GRID, INK, INK2, INK3, LIVE, PANEL, SURFACE, WARN,
                              draw_arrow, draw_button)

#: The picture is drawn at this size and shown scaled to its window, which can
#: be any size from the second.
W, H = 1020, 880
MIN_W, MIN_H = 640, 552
#: Seconds of the plot, of the stick's trail, and between two looks for a changed calibration.
WINDOW_S, TRAIL_S, WATCH_S = 8.0, 0.8, 1.0
X_HUE, Y_HUE = (79, 143, 247), (240, 136, 62)
_PLOT_Y = 280                   # the top of the plot and of the stick's square
_VALUE_W = 58                   # the room for a value between its - and +
#: The arrow each stick direction is drawn as, and the angle it points at (y grows downwards).
_ARROW_OF = {**{d: d for d in STICK}, **dict(zip(RIGHT_STICK, STICK))}
_ANGLE_OF = dict(zip(STICK, (-math.pi / 2, math.pi / 2, math.pi, 0.0)))
#: Each tuning button: the field it moves, and by how much.
_TUNE_STEPS: dict[str, tuple[str, float]] = {
    "on-": ("on", -0.01), "on+": ("on", 0.01), "off-": ("off", -0.01), "off+": ("off", 0.01),
    "dwell-": ("dwell_s", -0.01), "dwell+": ("dwell_s", 0.01),
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


def _layout(sticks: tuple[Stick, ...]) -> tuple[Button, ...]:
    """The buttons for a pad with ``sticks``: the actions' row above the tuning's.

    In the tuning row, each value sits in the gap between its - and +.
    """
    top, _ = _row(20, 10, 30, [
        ("Calibrate", "calibrate"),
        *((f"Calibrate {s.value}", f"calibrate-{s.value}") for s in sticks),
        ("Save", "save"), ("Reload", "reload"), ("Clear", "clear"), ("Quit", "quit")])
    shown, x = _row(20, 48, 26, [(s.value.title(), f"show-{s.value}") for s in sticks])
    tuning = []
    for name in ("on", "off", "dwell"):
        x += 20 + 7 * len(name)                     # its name, before the -
        tuning += [Button("-", f"{name}-", pygame.Rect(x, 48, 26, 26)),
                   Button("+", f"{name}+", pygame.Rect(x + 26 + _VALUE_W, 48, 26, 26))]
        x += 52 + _VALUE_W
    diagonals = Button("Diagonals", "diag", pygame.Rect(x + 24, 48, 22 + 7 * 9, 26))
    return (*top, *shown, *tuning, diagonals)


def _retuned(t: StickTuning, action: str) -> StickTuning:
    """A stick's tuning after one tuning button, kept within what a calibration file allows."""
    if action == "diag":
        return replace(t, diagonal=0.7 if t.diagonal >= 1.0 else 1.0)
    name, step = _TUNE_STEPS[action]
    low, high = {"on": (t.off, 0.99), "off": (0.01, t.on), "dwell_s": (0.0, MAX_DWELL_S)}[name]
    return replace(t, **{name: round(min(high, max(low, getattr(t, name) + step)), 3)})


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _watched() -> tuple[float, float]:
    """When the calibrations' folder last changed, and the file in use."""
    found = pads.in_use()
    return _mtime(pads.calibrations_dir()), _mtime(found[0]) if found else 0.0


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
    """A calibrated value as a y pixel on the scrolling plot."""
    return int(cy - max(-1.0, min(1.0, v)) * half)


def _tx(t: float, now: float, px: int, pw: int) -> int:
    """A timestamp as an x pixel, newest at the right edge."""
    return int(px + pw - (now - t) / WINDOW_S * pw)


def _qpos(vx: float, vz: float, mid: tuple[int, int], q: int) -> tuple[int, int]:
    """A calibrated x/y pair as a point in the square stick plot."""
    return (int(mid[0] + max(-1, min(1, vx)) * q // 2),
            int(mid[1] + max(-1, min(1, vz)) * q // 2))


def _draw_tiles(screen: pygame.Surface, fonts, pad: Pad | None, down: set[str],
                keys: frozenset, by: int) -> None:
    """One tile per button the pad has: its name, where it comes from, and its key."""
    f_big, f_sm = fonts
    names = [c for c in pad.controls() if c not in STICK_DIRECTIONS] if pad else []
    cols, bx, gap = 7, 20, 4
    bw = (W - 2 * bx - (cols - 1) * gap) // cols
    for k, name in enumerate(names):
        x, y = bx + (k % cols) * (bw + gap), by + (k // cols) * (62 + gap)
        key, is_down, hue = pads.binding(name), name in down, FACE.get(name, INK2)
        pygame.draw.rect(screen, hue if is_down else PANEL, (x, y, bw, 62), border_radius=4)
        pygame.draw.rect(screen, hue if is_down else GRID, (x, y, bw, 62), 1, border_radius=4)
        ink = SURFACE if is_down else INK2
        screen.blit(f_big.render(name, True, SURFACE if is_down else hue), (x + 8, y + 6))
        screen.blit(f_sm.render(pad.source(name), True, ink), (x + 8, y + 42))
        screen.blit(f_big.render(key or "-", True, ink if key else INK3), (x + bw - 60, y + 36))
        if is_down and key in keys:
            pygame.draw.circle(screen, SURFACE, (x + bw - 10, y + 10), 4)


def _draw_rule(screen: pygame.Surface, stick: Stick, controls: frozenset,
               mid: tuple[int, int], q: int) -> None:
    """The rule that decides the stick: the on and off reach, and each direction's sector.

    One direction at a time is a distance from rest and a set of directions, not a
    pair of per-axis boxes, so the picture never implies a rule that is not the one used.
    """
    tuning = pads.tuning(stick)
    classifying = tuning.diagonal >= 1.0
    for frac, col in ((tuning.on, (120, 130, 150)), (tuning.off, (70, 78, 96))):
        r = int(frac * q // 2)
        if classifying:
            pygame.draw.circle(screen, col, mid, r, 1)
        else:
            pygame.draw.rect(screen, col, (mid[0] - r, mid[1] - r, r * 2, r * 2), 1)
    if not classifying:
        return
    reach = q // 2 - 4
    for name, base in zip(stick.directions, STICK):
        angle = _ANGLE_OF[base]
        end = (mid[0] + math.cos(angle) * reach, mid[1] + math.sin(angle) * reach)
        pygame.draw.line(screen, (58, 66, 84), mid, end, 1)
        draw_arrow(screen, base, (int(end[0]), int(end[1])), 9,
                   LIVE if name in controls else INK3)
        edge = angle + math.pi / 4                      # the boundary to the next
        pygame.draw.line(screen, (44, 50, 64), mid,
                         (mid[0] + math.cos(edge) * reach, mid[1] + math.sin(edge) * reach), 1)


def _draw_dots(screen: pygame.Surface, stick: Stick, raw: dict[int, float], controls: frozenset,
               trail: list, mid: tuple[int, int], q: int) -> tuple[float, float]:
    """The stick's trail and where it is now; the other stick a small dot. Returns its x and y."""
    for i, h in enumerate(trail):
        k = 40 + i // 2
        pygame.draw.circle(screen, (k, k + 6, k + 16), _qpos(h[1], h[2], mid, q), 2)
    other = next((s for s in Stick if s != stick and s.axes[0] in raw), None)
    if other is not None:
        ox, oy = (pads.normalise(i, raw.get(i, 0.0)) for i in other.axes)
        pygame.draw.circle(screen, INK3, _qpos(ox, oy, mid, q), 4)
    sx, sy = (pads.normalise(i, raw.get(i, 0.0)) for i in stick.axes)
    cur = _qpos(sx, sy, mid, q)
    pygame.draw.circle(screen, SURFACE, cur, 9)
    pygame.draw.circle(screen, LIVE if controls & set(stick.directions) else X_HUE, cur, 6)
    return sx, sy


def _draw_square(screen: pygame.Surface, fonts, stick: Stick, raw: dict[int, float],
                 controls: frozenset, trail: list, at: tuple[int, int, int]) -> None:
    """The stick shown: where it is, the rule deciding it, and its numbers."""
    f, f_sm = fonts
    qx, qy, q = at
    pygame.draw.rect(screen, PANEL, (qx, qy, q, q))
    mid = (qx + q // 2, qy + q // 2)
    pygame.draw.line(screen, GRID, (qx, mid[1]), (qx + q, mid[1]))
    pygame.draw.line(screen, GRID, (mid[0], qy), (mid[0], qy + q))
    _draw_rule(screen, stick, controls, mid, q)
    sx, sy = _draw_dots(screen, stick, raw, controls, trail, mid, q)
    screen.blit(f.render(f"{stick.value} stick position", True, INK2), (qx + 8, qy + 6))
    centre = " ".join(f"{pads.centre(i):+.3f}" for i in stick.axes)
    rows = [(f"{pads.tuning(stick).rule()}   pushed {math.hypot(sx, sy):.2f}", INK2),
            (f"calibrated {sx:+.3f} {sy:+.3f}", INK),
            (f"raw        {raw.get(stick.axes[0], 0.0):+.3f} "
             f"{raw.get(stick.axes[1], 0.0):+.3f}", INK3),
            (f"centre     {centre}  " + ("(tracked while idle)" if pads.is_calibrated(stick)
                                         else "(uncalibrated)"), INK3)]
    for k, (text, colour) in enumerate(rows):
        screen.blit(f_sm.render(text, True, colour), (qx + 8, qy + q - 60 + k * 14))


class _Scope:
    """The tool's window: what the pad sends, what is made of it, and the buttons."""

    def __init__(self, bindings: dict[str, str], has_rig: bool) -> None:
        self.canvas = _Canvas("fmri-pad")
        font = "Menlo,Monaco,monospace"
        self.f_big, self.f, self.f_sm = (pygame.font.SysFont(font, n) for n in (17, 13, 11))
        self.bindings, self.has_rig = bindings, has_rig
        self.flash, self.flash_until = "", 0.0
        self.hist: deque = deque()
        self.log: deque = deque(maxlen=11)
        self.prev: frozenset = frozenset()
        self.stick, self.running, self.t0 = Stick.LEFT, True, time.perf_counter()
        self.reload("")
        self.next_watch = self.now() + WATCH_S

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def say(self, text: str, seconds: float = 4.0) -> None:
        self.flash, self.flash_until = text, self.now() + seconds

    def reload(self, said: str) -> None:
        """Read the newest calibration of the pad plugged in again, or the error it gives."""
        try:
            self.status = pads.init(True, self.bindings, is_calibration_required=False)
            self.error = ""
        except CalibrationError as exc:
            pads.use(None)
            self.error = str(exc)
        pad = pads.first()
        if self.error:
            self.status = f"{pad.name} ({pad.layout.name}), read uncalibrated" if pad else ""
        sticks = pad.sticks() if pad else ()
        self.buttons = _layout(sticks)
        self.stick = self.stick if self.stick in sticks else Stick.LEFT
        self.watched = _watched()
        self.hist.clear()
        if said:
            self.say(said)

    def watch(self, now: float) -> None:
        """Every :data:`WATCH_S`, reload if a calibration was added or the one in use changed."""
        if now < self.next_watch:
            return
        self.next_watch = now + WATCH_S
        if _watched() != self.watched:
            self.reload("a calibration changed: reloaded")

    def calibrate(self, stick: Stick | None) -> None:
        """Run the calibration of ``stick``, or of every stick and button, in this window."""
        pad = pads.first()
        sticks = pad.sticks() if stick is None else (stick,)
        try:
            result = calibration.calibrate(pad, self.canvas.surface, self.canvas.present,
                                           sticks, self.canvas.to_canvas)
        except CalibrationError as exc:
            self.say(f"NOT calibrated: {exc}", 8.0)
            return
        self.reload("NOT calibrated: cancelled; nothing written" if result is None
                    else f"calibrated: {result.path.name}")

    def save(self) -> None:
        """Write the calibration in use, with each stick's tuning now, as a new file."""
        path, used = pads.in_use()
        tuned = replace(used, tuned_from=path.name, sticks={
            s: replace(c, tuning=pads.tuning(s)) for s, c in used.sticks.items()})
        self.reload(f"saved {pads.write(tuned).name}, tuned from {path.name}")

    def clear(self) -> None:
        self.hist.clear()
        self.log.clear()

    def show(self, stick: Stick) -> None:
        self.stick = stick
        self.hist.clear()

    def retune(self, action: str) -> None:
        pads.set_tuning(self.stick, _retuned(pads.tuning(self.stick), action))

    def is_enabled(self, action: str) -> bool:
        """Whether a button can do anything now."""
        if action.startswith("calibrate"):
            return pads.first() is not None and not self.error
        if action == "save":
            return pads.in_use() is not None
        return True

    def events(self) -> None:
        """Clicks on the buttons, and ESC or the window's close to quit."""
        for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN)):
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN and not getattr(e, "pad", False) \
                    and e.key == pygame.K_ESCAPE:
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
            pads.pump()
            self.draw()
            clock.tick(60)
        pygame.quit()

    def record(self, now: float, controls: frozenset, raw: dict[int, float]) -> None:
        """Log the presses and releases since the last frame, and add this frame to the plot."""
        for name in sorted(controls - self.prev):
            key = pads.binding(name) or ("no key bound" if self.has_rig else "no --rig given")
            self.log.appendleft((now, f"{name:<10} pressed   -> {key}", LIVE))
        for name in sorted(self.prev - controls):
            self.log.appendleft((now, f"{name:<10} released", INK3))
        self.prev = controls
        sx, sy = (pads.normalise(i, raw.get(i, 0.0)) for i in self.stick.axes)
        self.hist.append((now, sx, sy, controls))
        while self.hist and now - self.hist[0][0] > WINDOW_S:
            self.hist.popleft()

    def draw(self) -> None:
        now, pad, controls = self.now(), pads.first(), pads.held_controls()
        raw, down = pad.read() if pad else ({}, set())
        self.record(now, controls, raw)
        screen = self.canvas.surface
        screen.fill(SURFACE)
        self.draw_toolbar(screen)
        self.draw_status(screen, now)
        _draw_tiles(screen, (self.f_big, self.f_sm), pad, down, pads.held(), 136)
        lanes = self.draw_plot(screen, controls, now)
        trail = [h for h in self.hist if now - h[0] < TRAIL_S]
        _draw_square(screen, (self.f, self.f_sm), self.stick, raw, controls, trail,
                     (676, _PLOT_Y, 310))
        self.draw_lanes(screen, pad, controls, now)
        self.draw_log(screen, _PLOT_Y + 200 + 10 + lanes * 16 + 14)
        if pad is None:
            screen.blit(self.f_big.render("no controller plugged in", True, WARN),
                        (684, _PLOT_Y + 150))
        self.canvas.present()

    def draw_status(self, screen: pygame.Surface, now: float) -> None:
        """The pad's status, the calibration in use or why there is none, and the last message."""
        found = pads.in_use()
        cal = (self.error or (f"calibration in use: {found[0]}" if found else
                              "UNCALIBRATED: no calibration of this pad; Calibrate makes one"))
        for y, text, colour in ((84, self.status, INK2),
                                (98, cal, WARN if self.error or not found else INK2)):
            line = _wrapped(text, self.f_sm, W - 40, 1)[0]
            screen.blit(self.f_sm.render(line, True, colour), (20, y))
        if self.flash and now < self.flash_until:
            screen.blit(self.f.render(_wrapped(self.flash, self.f, W - 40, 1)[0], True,
                                      WARN if self.flash.startswith("NOT") else LIVE), (20, 114))

    def draw_toolbar(self, screen: pygame.Surface) -> None:
        mouse = self.canvas.to_canvas(pygame.mouse.get_pos())
        tuning = pads.tuning(self.stick)
        values = {"on": f"{tuning.on:.2f}", "off": f"{tuning.off:.2f}",
                  "dwell": f"{tuning.dwell_s * 1000:.0f} ms"}
        for label, action, rect in self.buttons:
            on = (action == f"show-{self.stick.value}"
                  or action == "diag" and tuning.diagonal < 1.0)
            draw_button(screen, self.f_sm, rect, label, rect.collidepoint(mouse),
                        self.is_enabled(action), on)
            if action[:-1] in values and action.endswith("-"):   # its name, and its value after
                name = self.f_sm.render(action[:-1], True, INK2)
                screen.blit(name, name.get_rect(midright=(rect.x - 6, rect.centery)))
                value = self.f.render(values[action[:-1]], True, INK)
                screen.blit(value, value.get_rect(center=(rect.right + _VALUE_W // 2,
                                                          rect.centery)))

    def draw_plot(self, screen: pygame.Surface, controls: frozenset, now: float) -> int:
        """The shown stick's two axes over time, against its thresholds.

        :return: how many lanes go below it.
        """
        tuning, hist = pads.tuning(self.stick), self.hist
        px, py, pw, ph = 20, _PLOT_Y, 630, 200
        cy, half = py + ph // 2, ph // 2
        pygame.draw.rect(screen, PANEL, (px, py, pw, ph))
        for lvl in (-1.0, -0.5, 0.5, 1.0):
            pygame.draw.line(screen, GRID, (px, _vy(lvl, cy, half)), (px + pw, _vy(lvl, cy, half)))
            screen.blit(self.f_sm.render(f"{lvl:+.1f}", True, INK3),
                        (px + 3, _vy(lvl, cy, half) - 12))
        pygame.draw.line(screen, (60, 68, 86), (px, cy), (px + pw, cy))
        for sgn in (1, -1):
            y_on, y_off = _vy(sgn * tuning.on, cy, half), _vy(sgn * tuning.off, cy, half)
            pygame.draw.line(screen, (120, 130, 150), (px, y_on), (px + pw, y_on))
            for seg in range(px, px + pw, 10):
                pygame.draw.line(screen, (80, 88, 106), (seg, y_off), (seg + 5, y_off))
        if len(hist) > 1:
            for idx, hue in ((1, X_HUE), (2, Y_HUE)):
                pygame.draw.lines(screen, hue, False, [(_tx(h[0], now, px, pw),
                                                        _vy(h[idx], cy, half)) for h in hist], 2)
            for a, b in zip(hist, list(hist)[1:]):
                if (b[3] - a[3]) & set(self.stick.directions):
                    x = _tx(b[0], now, px, pw)
                    pygame.draw.line(screen, LIVE, (x, py), (x, py + ph), 1)
        screen.blit(self.f.render(f"{self.stick.value} stick x", True, X_HUE), (px + 8, py + 6))
        screen.blit(self.f.render(f"{self.stick.value} stick y", True, Y_HUE), (px + 150, py + 6))
        screen.blit(self.f_sm.render("calibrated units", True, INK3), (px + pw - 100, py + 6))
        return len(self._lanes(pads.first()))

    @staticmethod
    def _lanes(pad: Pad | None) -> list[str]:
        return [d for d in (pad.controls() if pad else STICK) if d in STICK_DIRECTIONS]

    def draw_lanes(self, screen: pygame.Surface, pad: Pad | None, controls: frozenset,
                   now: float) -> None:
        """A lane per stick direction, lit while it was pressed."""
        px, pw, hist = 20, 630, self.hist
        for r, name in enumerate(self._lanes(pad)):
            y = _PLOT_Y + 210 + r * 16
            draw_arrow(screen, _ARROW_OF[name], (px + 8, y + 7), 7,
                       LIVE if name in controls else INK3)
            if name in RIGHT_STICK:
                screen.blit(self.f_sm.render("RS", True, INK3), (px + 20, y))
            pygame.draw.rect(screen, PANEL, (px + 46, y + 2, pw - 46, 10))
            for a, b in zip(hist, list(hist)[1:]):
                if name in b[3]:
                    x1, x2 = _tx(a[0], now, px, pw), _tx(b[0], now, px, pw)
                    pygame.draw.rect(screen, LIVE, (max(x1, px + 46), y + 2, max(1, x2 - x1), 10))

    def draw_log(self, screen: pygame.Surface, ly: int) -> None:
        screen.blit(self.f.render("events", True, INK2), (20, ly))
        for i, (t, text, colour) in enumerate(self.log):
            screen.blit(self.f_sm.render(f"{t:7.2f}s  {text}", True, colour),
                        (20, ly + 20 + i * 14))
        screen.blit(self.f_sm.render("the buttons along the top do everything; ESC quits",
                                     True, INK3), (20, H - 22))


#: What each button does.
_ACTIONS: dict[str, Callable[[_Scope], None]] = {
    "calibrate": lambda s: s.calibrate(None),
    **{f"calibrate-{st.value}": (lambda s, st=st: s.calibrate(st)) for st in Stick},
    **{f"show-{st.value}": (lambda s, st=st: s.show(st)) for st in Stick},
    **{a: (lambda s, a=a: s.retune(a)) for a in (*_TUNE_STEPS, "diag")},
    "save": _Scope.save, "reload": lambda s: s.reload("reloaded"), "clear": _Scope.clear,
    "quit": lambda s: setattr(s, "running", False),
}


def main() -> None:
    """``fmri-pad [--rig <rig file>]``."""
    p = argparse.ArgumentParser(
        description="Calibrate the controller plugged in, see what it sends and what fmri-gym "
                    "makes of it, and tune how its sticks are read, with the buttons along the "
                    "top. Calibrations are kept in " + os.fspath(pads.calibrations_dir()))
    p.add_argument("--rig", default=None, metavar="RIG.json",
                   help="the rig file whose bindings to show beside each control")
    rig = p.parse_args().rig
    _Scope(RigConfig.open(rig).bindings if rig else {}, has_rig=rig is not None).loop()
