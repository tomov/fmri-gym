"""Fixed-size pygame display: aspect-fit game frames, fixation, text, survey.

Engine-agnostic -- it only ever receives RGB numpy frames (H,W,3) from
env.render(), so ALE / stable-retro / any gym env all present identically:
letterboxed and centered in one fixed window, giving uniform screen geometry.

Timing
------
Everything is composed on an off-screen canvas and shown by one
:meth:`Display._present`, which asks SDL for a vsync-locked flip:
``pygame.display.flip()`` then blocks until the vertical blank at which the
new image starts scanning out. The stamp taken right after it
(:attr:`last_flip_perf`, returned by every ``draw_*``) is that onset, and
callbacks queued with :meth:`call_on_flip` run at it -- where frame triggers
are sent.

A flip only blocks when the swap chain is full, so :meth:`idle` re-presents
the current canvas every refresh while the session waits, keeping the chain
primed. :attr:`vsync` is measured at start-up, not read from a flag: a burst
of flips is timed, and the flag is set only if they block for about a refresh
period (SDL's headless driver and some compositors claim vsync and never
block). Without it, :meth:`idle` sleeps in 1 ms slices and frame onsets are
what the flip reported.

The remaining unknown is any constant offset between the flip returning and
photons (swap-chain depth, panel latency); that is what a photodiode is for.
``python -m fmri_gym.display`` reports what a given machine achieves, on the
``monitor`` the session uses (:func:`list_monitors`).
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import TYPE_CHECKING, Any, Callable

import pygame

if TYPE_CHECKING:
    import numpy as np

BG_COLOR = (0, 0, 0)
TEXT_COLOR = (220, 220, 220)
FIX_COLOR = (255, 255, 255)
HUD_COLOR = (235, 235, 235)
# Text drawn ON a frame, where gray would read as part of the picture. Light
# sky blue: bright over both dark and light art, and cool enough not to be
# mistaken for the red a game may reserve.
OVERLAY_COLOR = (150, 215, 255)


class Display:
    """Fixed-size pygame window for game frames, fixation, and text.

    :ivar vsync: whether flips were measured to block on the vertical blank.
    :ivar refresh_rate: the monitor's refresh rate in Hz (0 if unknown).
    :ivar last_flip_perf: ``perf_counter`` of the most recent flip.
    """

    def __init__(
        self,
        size: tuple[int, int] = (1024, 768),
        fullscreen: bool = False,
        caption: str = "fmri-gym",
        vsync: bool = True,
        monitor: int = 0,
    ) -> None:
        """Create and show the display window.

        :param size: requested ``(width, height)`` in pixels (windowed mode;
            fullscreen uses the desktop resolution, so frames are never
            rescaled twice).
        :param fullscreen: if ``True``, open in fullscreen mode.
        :param caption: window title string.
        :param vsync: ask for a flip locked to the vertical blank (default).
        :param monitor: the monitor to open on, by :func:`list_monitors` index.
        """
        self._monitor = monitor
        self._requested_size = size
        self._fullscreen = fullscreen
        self._caption = caption
        self._vsync_requested = vsync
        self._on_flip: list[tuple[Callable[..., Any], tuple]] = []
        self.last_flip_perf: float | None = None
        self._init_display()

    def _init_display(self) -> None:
        """(Re)initialize pygame and create the screen, canvas and fonts."""
        pygame.init()
        pygame.mouse.set_visible(False)
        self.screen = self._open()
        pygame.display.set_caption(self._caption)
        self.size = self.screen.get_size()
        self.canvas = pygame.Surface(self.size)
        self.refresh_rate = int(pygame.display.get_current_refresh_rate() or 0)
        self.vsync = bool(pygame.display.is_vsync()) and self._flips_block()
        if self._vsync_requested and not self.vsync:
            print("display: flips do not lock to the refresh here -- frame onsets are "
                  "what the flip reports (run `python -m fmri_gym.display` to check)",
                  file=sys.stderr)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 28)
        self.fix_font = pygame.font.Font(pygame.font.get_default_font(), 80)
        self.hud_font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self._labels: dict[str, pygame.Surface] = {}    # _draw_hud's rendered lines

    def _open(self) -> pygame.Surface:
        """Open the window, trying for a vsync-locked flip.

        A plain surface is tried first (fullscreen at desktop resolution, so
        nothing is rescaled). Some drivers only honour vsync through SDL's
        renderer path, which pygame exposes as ``SCALED`` and which needs an
        explicit logical size; that is the fallback, and a plain unsynced
        window the last resort.

        :return: the screen surface.
        """
        size = (0, 0) if self._fullscreen else self._requested_size
        flags = pygame.FULLSCREEN if self._fullscreen else 0
        monitor = self._monitor
        if not self._vsync_requested:
            return pygame.display.set_mode(size, flags, display=monitor)
        screen = _try_mode(size, flags, vsync=1, display=monitor)
        if screen is not None and pygame.display.is_vsync():
            return screen
        logical_size = screen.get_size() if screen is not None else self._requested_size
        scaled = _try_mode(logical_size, flags | pygame.SCALED, vsync=1, display=monitor)
        if scaled is not None:
            return scaled
        return pygame.display.set_mode(size, flags, display=monitor)

    def _flips_block(self, n_flips: int = 24) -> bool:
        """Measure whether a burst of flips waits for the blank.

        :param n_flips: flips to time after a short priming burst.
        :return: ``True`` if the mean interval is at least half a refresh
            period (or 2 ms when the rate is unknown).
        """
        flip_perfs = [self._present() for _ in range(n_flips + 8)][8:]
        mean_ms = statistics.mean((later - earlier) * 1000
                                  for earlier, later in zip(flip_perfs, flip_perfs[1:]))
        floor_ms = 500.0 / self.refresh_rate if self.refresh_rate else 2.0
        return mean_ms >= floor_ms

    def ensure(self) -> None:
        """Re-create the window if pygame display was torn down.

        Some gym envs call ``pygame.display.quit()`` on ``close()``, which
        destroys the shared window; call this after closing an env.
        """
        if not pygame.display.get_init() or not pygame.get_init():
            self._init_display()

    def call_on_flip(self, fn: Callable[..., Any], *args: Any) -> None:
        """Queue ``fn(*args)`` to run right after the next flip returns.

        One-shot. With vsync the flip returns at the vertical blank, so this is
        the closest a program gets to the moment the frame appears -- where a
        frame trigger belongs.

        :param fn: callable to run.
        :param args: its positional arguments.
        """
        self._on_flip.append((fn, args))

    def _present(self) -> float:
        """Show the canvas: blit, flip, stamp, run the queued on-flip callbacks.

        :return: ``perf_counter`` right after the flip returned.
        """
        self.screen.blit(self.canvas, (0, 0))
        pygame.display.flip()
        flip_perf = time.perf_counter()
        self.last_flip_perf = flip_perf
        callbacks, self._on_flip = self._on_flip, []
        for fn, args in callbacks:
            fn(*args)
        return flip_perf

    def redraw(self) -> float:
        """Present the current canvas again (one refresh when vsync-locked).

        :return: ``perf_counter`` of the flip.
        """
        return self._present()

    def idle(self, deadline_perf: float, poll: float = 0.001) -> None:
        """Pass a little time before ``deadline_perf`` without losing frame timing.

        Vsync-locked: re-present the canvas, which blocks until the next blank
        and keeps the swap chain primed. Otherwise sleep at most ``poll``
        seconds, so callers polling the keyboard in a loop stamp events to
        about a millisecond.

        :param deadline_perf: ``perf_counter`` the caller is waiting for.
        :param poll: maximum sleep when not vsync-locked.
        """
        if self.vsync:
            self._present()
            return
        time.sleep(max(0.0, min(poll, deadline_perf - time.perf_counter())))

    def draw_frame(
        self,
        frame_rgb: np.ndarray,
        hud_lines: list[str] | None = None,
        overlay: tuple[list[str], float] | None = None,
    ) -> float:
        """Blit an RGB frame, aspect-fit and centered with black pad.

        With ``hud_lines``, the frame is fit below a strip above it where the
        lines are drawn (see :meth:`_draw_hud`), so the text never covers a
        pixel the subject is playing on. An ``overlay`` is the other trade,
        drawn over the frame itself (see :meth:`_draw_overlay`).

        :param frame_rgb: frame array shaped ``(H, W, 3)``.
        :param hud_lines: status lines from the adapter's
            :meth:`~fmri_gym.adapters.base.EnvAdapter.hud`, or ``None`` for none.
        :param overlay: ``(lines, y_frac)`` from the adapter's
            :meth:`~fmri_gym.adapters.base.EnvAdapter.overlay`, or ``None``.
        :return: ``perf_counter`` of the flip that showed it.
        """
        self.canvas.fill(BG_COLOR)
        frame_h, frame_w = frame_rgb.shape[:2]
        surface = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))  # -> (W,H)
        # The HUD is packed to the frame's width, and the strip it needs shrinks
        # a height-limited frame: fit, pack, and fit again until they agree.
        rows: list[list[pygame.Surface]] = []
        strip_height = 0
        while True:
            scale = min(self.size[0] / frame_w, (self.size[1] - strip_height) / frame_h)
            fit_w, fit_h = int(frame_w * scale), int(frame_h * scale)
            if not hud_lines:
                break
            rows = self._hud_rows(hud_lines, fit_w)
            needed_height = len(rows) * self.hud_font.get_linesize() + 8
            if needed_height <= strip_height:
                break
            strip_height = needed_height
        surface = pygame.transform.scale(surface, (fit_w, fit_h))
        rect = surface.get_rect(center=(self.size[0] // 2,
                                        strip_height + (self.size[1] - strip_height) // 2))
        self.canvas.blit(surface, rect.topleft)
        if rows:
            self._draw_hud(rows, rect)
        if overlay:
            self._draw_overlay(overlay, rect)
        return self._present()

    def _draw_hud(self, rows: list[list[pygame.Surface]], rect: pygame.Rect) -> None:
        """Draw the HUD in the strip above the frame, as rows of lines laid out left to right.

        Within a row the first line sits flush with the frame's left edge and
        the last flush with its right, the others spaced evenly between -- so
        the two-line default (time left, score) reads as a left and a right
        corner, and a longer status list as a row of fields. Rows are what
        :meth:`_hud_rows` split the lines into when they do not fit one; the
        last row sits just above the frame.

        :param rows: the rendered lines, by row.
        :param rect: the rect the frame was blitted into.
        """
        y = rect.top - 4
        for row in reversed(rows):
            y -= self.hud_font.get_linesize()
            if len(row) == 1:
                self.canvas.blit(row[0], (rect.left, y))
                continue
            free = rect.width - sum(surface.get_width() for surface in row)
            gap = free / (len(row) - 1)
            x = float(rect.left)
            for surface in row:
                self.canvas.blit(surface, (round(x), y))
                x += surface.get_width() + gap

    def _draw_overlay(
        self, overlay: tuple[list[str], float], rect: pygame.Rect
    ) -> None:
        """Draw lines over the frame on a black backdrop, horizontally centered.

        The counterpart of :meth:`_draw_hud`, and the opposite trade: the strip
        is the right place for something the subject reads now and then, and
        the wrong place for something they are acting on, which has to be where
        they are already looking. Covering game pixels is the cost, so an
        adapter should put a line here only while it is in use.

        Drawn at display resolution over the scaled frame, not into the array,
        so the text is not magnified by the same unfiltered scale as the art --
        and so the frame the adapter logged stays the frame the engine made.

        :param overlay: ``(lines, y_frac)``, where ``y_frac`` is the fraction
            of the frame's height the lines are centered on (0.5 = middle).
        :param rect: the rect the frame was blitted into.
        """
        lines, y_frac = overlay
        surfaces = [self.hud_font.render(line, True, OVERLAY_COLOR) for line in lines]
        if not surfaces:
            return
        total_height = sum(surface.get_height() for surface in surfaces)
        width = max(surface.get_width() for surface in surfaces)
        top = rect.top + int(rect.height * y_frac) - total_height // 2
        backdrop = pygame.Rect(0, 0, width + 16, total_height + 10)
        backdrop.center = (rect.centerx, top + total_height // 2)
        self.canvas.fill(BG_COLOR, backdrop)
        y = top
        for surface in surfaces:
            self.canvas.blit(surface, surface.get_rect(midtop=(rect.centerx, y)))
            y += surface.get_height()

    def _hud_rows(self, lines: list[str], width: int) -> list[list[pygame.Surface]]:
        """Render the HUD lines and pack them into rows no wider than the frame.

        :param lines: the adapter's status lines.
        :param width: the width they have to fit, in pixels.
        :return: rows of rendered lines, in order.
        """
        gap = self.hud_font.size("  ")[0]
        rows: list[list[pygame.Surface]] = [[]]
        used = 0
        for surface in (self._label(line) for line in lines):
            if rows[-1] and used + gap + surface.get_width() > width:
                rows.append([])
                used = 0
            rows[-1].append(surface)
            used += surface.get_width() + (gap if len(rows[-1]) > 1 else 0)
        return rows

    def _label(self, text: str) -> pygame.Surface:
        """Render one HUD line, cached: a line only changes when its number does.

        :param text: the line.
        :return: its surface.
        """
        if text not in self._labels:
            if len(self._labels) > 64:
                self._labels.clear()
            self._labels[text] = self.hud_font.render(text, True, HUD_COLOR, BG_COLOR)
        return self._labels[text]

    def _wrap(self, font: pygame.font.Font, line: str, max_width: int) -> list[str]:
        """Word-wrap one logical line so no rendered line exceeds ``max_width`` px.

        Leading whitespace (used for indented controls tables) is preserved on
        the first physical row. Very long single words are left intact.

        :param font: font used to measure rendered width.
        :param line: one logical line (may already contain leading indent).
        :param max_width: maximum pixel width per physical row.
        :return: list of physical lines to render.
        """
        if font.size(line)[0] <= max_width:
            return [line]
        indent = line[:len(line) - len(line.lstrip())]
        words = line.split()
        wrapped, current = [], indent
        for word in words:
            trial = (current + " " + word) if current.strip() else (indent + word)
            if font.size(trial)[0] <= max_width or not current.strip():
                current = trial
            else:
                wrapped.append(current)
                current = indent + word
        wrapped.append(current)
        return wrapped

    def draw_text(
        self,
        text: str,
        color: tuple[int, int, int] = TEXT_COLOR,
        font: pygame.font.Font | None = None,
        align: str = "center",
    ) -> float:
        """Render multi-line text, block-centered vertically.

        ``align="center"`` centers each line horizontally (default, for
        messages); ``align="left"`` left-aligns all lines against a common left
        edge so the whole block is horizontally centered (good for controls
        tables). Long lines are word-wrapped to fit the window.

        :param text: multi-line string (``\\n``-separated).
        :param color: RGB text color.
        :param font: pygame font; defaults to the display's body font.
        :param align: ``"center"`` or ``"left"`` horizontal alignment mode.
        :return: ``perf_counter`` of the flip that showed it.
        """
        font = font or self.font
        self.canvas.fill(BG_COLOR)
        max_width = int(self.size[0] * 0.92)
        lines = []
        for logical_line in text.split("\n"):
            lines.extend(self._wrap(font, logical_line, max_width))
        surfaces = [font.render(line, True, color) for line in lines]
        total_height = sum(surface.get_height() for surface in surfaces)
        y = (self.size[1] - total_height) // 2
        block_left = (self.size[0]
                      - max((surface.get_width() for surface in surfaces), default=0)) // 2
        for surface in surfaces:
            if align == "left":
                rect = surface.get_rect(topleft=(block_left, y))
            else:
                rect = surface.get_rect(center=(self.size[0] // 2, y + surface.get_height() // 2))
            self.canvas.blit(surface, rect)
            y += surface.get_height()
        return self._present()

    def draw_fixation(self) -> float:
        """Draw a centered white ``+`` fixation cross.

        :return: ``perf_counter`` of the flip that showed it.
        """
        return self.draw_text("+", color=FIX_COLOR, font=self.fix_font)

    def describe(self) -> dict[str, Any]:
        """What was actually opened, for the manifest.

        :return: size, fullscreen, monitor, measured vsync, refresh rate, SDL driver.
        """
        return {"size": list(self.size), "requested_size": list(self._requested_size),
                "fullscreen": self._fullscreen, "monitor": self._monitor, "vsync": self.vsync,
                "refresh_rate": self.refresh_rate, "driver": pygame.display.get_driver()}

    def measure_flips(self, n_flips: int = 240) -> dict[str, Any]:
        """Flip ``n_flips`` times and summarise the intervals.

        With vsync the mean should be ``1000 / refresh_rate`` ms and the
        spread well under a millisecond; a tiny mean means flips return
        without waiting for the blank.

        :param n_flips: number of timed flips (after a short warm-up).
        :return: ``mean_ms``, ``sd_ms``, ``min_ms``, ``max_ms``, ``late``: how
            many intervals exceeded 1.5 median periods (a refresh was missed),
            and the ``intervals_ms`` themselves.
        """
        for _ in range(30):
            self._present()
        flip_perfs = []
        for flip in range(n_flips + 1):
            self.canvas.fill((flip % 2 * 40,) * 3)
            flip_perfs.append(self._present())
        intervals_ms = [(later - earlier) * 1000
                        for earlier, later in zip(flip_perfs, flip_perfs[1:])]
        period_ms = statistics.median(intervals_ms)
        return {"mean_ms": statistics.mean(intervals_ms), "sd_ms": statistics.pstdev(intervals_ms),
                "min_ms": min(intervals_ms), "max_ms": max(intervals_ms),
                "late": sum(interval > 1.5 * period_ms for interval in intervals_ms),
                "intervals_ms": intervals_ms}

    def close(self) -> None:
        """Shut down pygame (closes the window)."""
        pygame.quit()


def _try_mode(size: tuple[int, int], flags: int, vsync: int,
              display: int) -> pygame.Surface | None:
    """``set_mode`` that returns ``None`` instead of raising."""
    try:
        return pygame.display.set_mode(size, flags, display=display, vsync=vsync)
    except pygame.error:
        return None


def monitor_label(index: int, monitor: tuple[int, int, int]) -> str:
    """``"1: 1920x1080 @ 60 Hz"``: a monitor as :func:`list_monitors` gives it."""
    width, height, hz = monitor
    return f"{index}: {width}x{height} @ {hz or '?'} Hz"


def list_monitors() -> list[tuple[int, int, int]]:
    """This machine's monitors, in ``monitor`` index order: ``(width, height, Hz)`` each."""
    started = pygame.display.get_init()
    pygame.display.init()
    try:
        return [(width, height, hz) for (width, height), hz
                in zip(pygame.display.get_desktop_sizes(),
                       pygame.display.get_desktop_refresh_rates())]
    finally:
        if not started:  # an open window needs it; a caller that only asked does not
            pygame.display.quit()


def check_monitor(monitor: int) -> None:
    """:raises ValueError: if this machine has no such monitor; the message lists them."""
    monitors = list_monitors()
    if not 0 <= monitor < len(monitors):
        labels = "; ".join(monitor_label(index, found) for index, found in enumerate(monitors))
        raise ValueError(f"--monitor {monitor}: this machine has {len(monitors)}: {labels}")


def quit_like_esc(_signum: int, _frame: object) -> None:
    """A SIGINT handler: Ctrl+C becomes the window's quit event, handled where ESC is.

    Python's own KeyboardInterrupt lands anywhere: between two fields of a
    frame, before a game block is saved. The quit event is read between frames.
    """
    if pygame.display.get_init():
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return
    print("Ctrl+C: no window to quit right now; press it again if the run goes on",
          file=sys.stderr)


def is_locked(info: dict[str, Any], stats: dict[str, Any]) -> bool:
    """Whether flips are locked to the refresh: vsync on, the mean at the period, a tight spread.

    :param info: :meth:`Display.describe`.
    :param stats: :meth:`Display.measure_flips`.
    """
    if not (info["vsync"] and info["refresh_rate"]):
        return False
    expected_ms = 1000.0 / info["refresh_rate"]
    return bool(abs(stats["mean_ms"] - expected_ms) < 0.1 * expected_ms and stats["sd_ms"] < 1.0)


def _selftest() -> None:
    """``python -m fmri_gym.display``: report whether flips lock to the refresh.

    With ``--outdir``, also writes ``display.npz`` (the intervals) and
    ``display.json`` (the display and the statistics) there, and exits
    non-zero when the flips are not locked: a rig check stops on it.
    """
    import argparse
    p = argparse.ArgumentParser(description="Measure flip timing on this machine.")
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--monitor", type=int, default=0)
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--seconds", type=float, help="flip for this long instead of --n flips")
    p.add_argument("--outdir", help="write display.npz + display.json here; fail if not locked")
    args = p.parse_args()
    width, height = (int(x) for x in args.size.lower().split("x"))
    display = Display((width, height), fullscreen=args.fullscreen, vsync=not args.no_vsync,
                      monitor=args.monitor)
    info = display.describe()
    if args.seconds:
        args.n = int(args.seconds * (info["refresh_rate"] or 60))
    stats = display.measure_flips(args.n)
    display.close()
    intervals_ms = stats.pop("intervals_ms")
    expected_ms = 1000.0 / info["refresh_rate"] if info["refresh_rate"] else float("nan")
    print(f"driver={info['driver']} size={info['size']} vsync={info['vsync']} "
          f"refresh={info['refresh_rate']} Hz (period {expected_ms:.2f} ms)")
    print("flip interval: mean {mean_ms:.2f} ms  sd {sd_ms:.2f}  min {min_ms:.2f}  "
          "max {max_ms:.2f}  late (missed a refresh) {late}/{n}".format(n=args.n, **stats))
    locked = is_locked(info, stats)
    print("verdict:", "LOCKED to the refresh" if locked else
          "NOT locked -- try --fullscreen, disable the desktop compositor, or check the GPU driver")
    if not args.outdir:
        return
    _save_selftest(args.outdir, info, {"n": args.n, **stats}, locked, intervals_ms)
    if not locked:
        raise RuntimeError("display: flips are NOT locked to the refresh; every timing "
                           "measured after this would be meaningless")


def _save_selftest(outdir: str, info: dict, stats: dict, locked: bool,
                   intervals_ms: list[float]) -> None:
    import json
    import os

    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "display.npz"),
                        intervals_ms=np.asarray(intervals_ms))
    with open(os.path.join(outdir, "display.json"), "w") as file:
        json.dump({"display": info, "flips": stats, "locked": locked}, file, indent=2)


if __name__ == "__main__":
    _selftest()
