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
(:attr:`last_flip`, returned by every ``draw_*``) is that onset, and callbacks
queued with :meth:`call_on_flip` run at it -- where frame triggers are sent.

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
# mistaken for the red a game may reserve (crafter-for-brain-scan v0.33 picked
# it for the same reason, and matching it keeps one rig's subjects at home in
# the other).
ON_FRAME_COLOR = (150, 215, 255)


class Display:
    """Fixed-size pygame window for game frames, fixation, and text.

    :ivar vsync: whether flips were measured to block on the vertical blank.
    :ivar refresh_rate: the monitor's refresh rate in Hz (0 if unknown).
    :ivar last_flip: ``perf_counter`` of the most recent flip.
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
        self._req_size = size
        self._fullscreen = fullscreen
        self._caption = caption
        self._want_vsync = vsync
        self._on_flip: list[tuple[Callable[..., Any], tuple]] = []
        self.last_flip: float | None = None
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
        if self._want_vsync and not self.vsync:
            print("display: flips do not lock to the refresh here -- frame onsets are "
                  "what the flip reports (run `python -m fmri_gym.display` to check)",
                  file=sys.stderr)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 28)
        self.fix_font = pygame.font.Font(pygame.font.get_default_font(), 80)
        self.hud_font = pygame.font.Font(pygame.font.get_default_font(), 24)

    def _open(self) -> pygame.Surface:
        """Open the window, trying for a vsync-locked flip.

        A plain surface is tried first (fullscreen at desktop resolution, so
        nothing is rescaled). Some drivers only honour vsync through SDL's
        renderer path, which pygame exposes as ``SCALED`` and which needs an
        explicit logical size; that is the fallback, and a plain unsynced
        window the last resort.

        :return: the screen surface.
        """
        size = (0, 0) if self._fullscreen else self._req_size
        flags = pygame.FULLSCREEN if self._fullscreen else 0
        at = self._monitor
        if not self._want_vsync:
            return pygame.display.set_mode(size, flags, display=at)
        screen = _try_mode(size, flags, vsync=1, display=at)
        if screen is not None and pygame.display.is_vsync():
            return screen
        logical = screen.get_size() if screen is not None else self._req_size
        scaled = _try_mode(logical, flags | pygame.SCALED, vsync=1, display=at)
        if scaled is not None:
            return scaled
        return pygame.display.set_mode(size, flags, display=at)

    def _flips_block(self, n: int = 24) -> bool:
        """Measure whether a burst of flips waits for the blank.

        :param n: flips to time after a short priming burst.
        :return: ``True`` if the mean interval is at least half a refresh
            period (or 2 ms when the rate is unknown).
        """
        stamps = [self._present() for _ in range(n + 8)][8:]
        mean_ms = statistics.mean((b - a) * 1000 for a, b in zip(stamps, stamps[1:]))
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
        t = time.perf_counter()
        self.last_flip = t
        callbacks, self._on_flip = self._on_flip, []
        for fn, args in callbacks:
            fn(*args)
        return t

    def redraw(self) -> float:
        """Present the current canvas again (one refresh when vsync-locked).

        :return: ``perf_counter`` of the flip.
        """
        return self._present()

    def idle(self, deadline: float, poll: float = 0.001) -> None:
        """Pass a little time before ``deadline`` without losing frame timing.

        Vsync-locked: re-present the canvas, which blocks until the next blank
        and keeps the swap chain primed. Otherwise sleep at most ``poll``
        seconds, so callers polling the keyboard in a loop stamp events to
        about a millisecond.

        :param deadline: ``perf_counter`` the caller is waiting for.
        :param poll: maximum sleep when not vsync-locked.
        """
        if self.vsync:
            self._present()
            return
        time.sleep(max(0.0, min(poll, deadline - time.perf_counter())))

    def draw_frame(
        self,
        rgb: np.ndarray,
        overlay: list[str] | None = None,
        on_frame: tuple[list[str], float] | None = None,
    ) -> float:
        """Blit an RGB frame, aspect-fit and centered with black pad.

        :param rgb: frame array shaped ``(H, W, 3)``.
        :param overlay: optional short status lines to draw alongside the
            frame; ``None`` (the default) draws the frame alone, exactly as
            before this argument existed.
        :param on_frame: optional ``(lines, y_frac)`` drawn over the frame
            itself rather than in the margin; ``None`` (the default) draws
            nothing. See :meth:`_draw_on_frame`.
        :return: ``perf_counter`` of the flip that showed it.
        """
        self.canvas.fill(BG_COLOR)
        h, w = rgb.shape[:2]
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))  # -> (W,H)
        scale = min(self.size[0] / w, self.size[1] / h)
        dw, dh = int(w * scale), int(h * scale)
        surf = pygame.transform.scale(surf, (dw, dh))
        rect = surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        self.canvas.blit(surf, rect.topleft)
        if overlay:
            self._draw_overlay(overlay, rect)
        if on_frame:
            self._draw_on_frame(on_frame, rect)
        return self._present()

    def _draw_overlay(self, lines: list[str], rect: pygame.Rect) -> None:
        """Draw status lines in the letterbox margin, or over the frame.

        Aspect-fitting a square frame into a 4:3 window leaves a black bar on
        each side, and text there covers no pixel the subject is playing on.
        That matters beyond tidiness: whatever the overlay says has to be
        state the env already reports, so the model harness sees the same
        numbers, and it must not hide game content from one side of the
        comparison. Lines are wrapped to the bar, so split long labels on
        spaces rather than underscores. A frame that fills the window has no
        bar; then the text lands top-left over the frame, on a black box so it
        stays readable.

        :param lines: short status strings, already formatted by the adapter.
        :param rect: rect the frame was blitted into, i.e. where the bar ends.
        """
        pad = 8
        column = rect.left if rect.left > 4 * pad else self.size[0]
        y = pad
        for raw in lines:
            for line in self._wrap(self.hud_font, raw, column - 2 * pad):
                surf = self.hud_font.render(line, True, HUD_COLOR)
                box = surf.get_rect(topleft=(pad, y))
                self.canvas.fill(BG_COLOR, box)
                self.canvas.blit(surf, box)
                y += self.hud_font.get_height()

    def _draw_on_frame(
        self, on_frame: tuple[list[str], float], rect: pygame.Rect
    ) -> None:
        """Draw lines over the frame on a black backdrop, horizontally centered.

        The counterpart of :meth:`_draw_overlay`, and the opposite trade: the
        margin is the right place for something the subject reads now and then,
        and the wrong place for something they are acting on, which has to be
        where they are already looking. Covering game pixels is the cost, so a
        caller should put a line here only while it is in use.

        Drawn at display resolution over the scaled frame, not into the array,
        so the text is not magnified by the same unfiltered scale as the art --
        and so the frame the adapter logged stays the frame the engine made.

        ``y_frac`` exists because "where they are already looking" is not
        always the middle of the picture: a game that draws its own HUD into
        the bottom of its frame leaves the played part above centre.

        :param on_frame: ``(lines, y_frac)``, where ``y_frac`` is the fraction
            of the frame's height the block is centered on (0.5 = middle).
        :param rect: rect the frame was blitted into.
        """
        lines, y_frac = on_frame
        surfs = [self.hud_font.render(ln, True, ON_FRAME_COLOR) for ln in lines]
        if not surfs:
            return
        total_h = sum(s.get_height() for s in surfs)
        width = max(s.get_width() for s in surfs)
        cx = rect.centerx
        top = rect.top + int(rect.height * y_frac) - total_h // 2
        backdrop = pygame.Rect(0, 0, width + 16, total_h + 10)
        backdrop.center = (cx, top + total_h // 2)
        self.canvas.fill(BG_COLOR, backdrop)
        y = top
        for surf in surfs:
            self.canvas.blit(surf, surf.get_rect(midtop=(cx, y)))
            y += surf.get_height()

    def _wrap(self, font: pygame.font.Font, line: str, max_w: int) -> list[str]:
        """Word-wrap one logical line so no rendered line exceeds ``max_w`` px.

        Leading whitespace (used for indented controls tables) is preserved on
        the first physical row. Very long single words are left intact.

        :param font: font used to measure rendered width.
        :param line: one logical line (may already contain leading indent).
        :param max_w: maximum pixel width per physical row.
        :return: list of physical lines to render.
        """
        if font.size(line)[0] <= max_w:
            return [line]
        indent = line[:len(line) - len(line.lstrip())]
        words = line.split()
        out, cur = [], indent
        for w in words:
            trial = (cur + " " + w) if cur.strip() else (indent + w)
            if font.size(trial)[0] <= max_w or not cur.strip():
                cur = trial
            else:
                out.append(cur)
                cur = indent + w
        out.append(cur)
        return out

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
        max_w = int(self.size[0] * 0.92)
        lines = []
        for raw in text.split("\n"):
            lines.extend(self._wrap(font, raw, max_w))
        surfs = [font.render(ln, True, color) for ln in lines]
        total_h = sum(s.get_height() for s in surfs)
        y = (self.size[1] - total_h) // 2
        block_left = (self.size[0] - max((s.get_width() for s in surfs), default=0)) // 2
        for surf in surfs:
            if align == "left":
                rect = surf.get_rect(topleft=(block_left, y))
            else:
                rect = surf.get_rect(center=(self.size[0] // 2, y + surf.get_height() // 2))
            self.canvas.blit(surf, rect)
            y += surf.get_height()
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
        return {"size": list(self.size), "requested_size": list(self._req_size),
                "fullscreen": self._fullscreen, "monitor": self._monitor, "vsync": self.vsync,
                "refresh_rate": self.refresh_rate, "driver": pygame.display.get_driver()}

    def measure_flips(self, n: int = 240) -> dict[str, float]:
        """Flip ``n`` times and summarise the intervals.

        With vsync the mean should be ``1000 / refresh_rate`` ms and the
        spread well under a millisecond; a tiny mean means flips return
        without waiting for the blank.

        :param n: number of timed flips (after a short warm-up).
        :return: ``mean_ms``, ``sd_ms``, ``min_ms``, ``max_ms``.
        """
        for _ in range(30):
            self._present()
        stamps = []
        for i in range(n + 1):
            self.canvas.fill((i % 2 * 40,) * 3)
            stamps.append(self._present())
        d = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
        return {"mean_ms": statistics.mean(d), "sd_ms": statistics.pstdev(d),
                "min_ms": min(d), "max_ms": max(d)}

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
    w, h, hz = monitor
    return f"{index}: {w}x{h} @ {hz or '?'} Hz"


def list_monitors() -> list[tuple[int, int, int]]:
    """This machine's monitors, in ``monitor`` index order: ``(width, height, Hz)`` each."""
    started = pygame.display.get_init()
    pygame.display.init()
    try:
        return [(w, h, hz) for (w, h), hz in zip(pygame.display.get_desktop_sizes(),
                                                  pygame.display.get_desktop_refresh_rates())]
    finally:
        if not started:  # an open window needs it; a caller that only asked does not
            pygame.display.quit()


def check_monitor(monitor: int) -> None:
    """:raises ValueError: if this machine has no such monitor; the message lists them."""
    monitors = list_monitors()
    if not 0 <= monitor < len(monitors):
        labels = "; ".join(monitor_label(i, m) for i, m in enumerate(monitors))
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


def _selftest() -> None:
    """``python -m fmri_gym.display``: report whether flips lock to the refresh."""
    import argparse
    p = argparse.ArgumentParser(description="Measure flip timing on this machine.")
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--monitor", type=int, default=0)
    p.add_argument("--n", type=int, default=240)
    args = p.parse_args()
    w, h = (int(x) for x in args.size.lower().split("x"))
    d = Display((w, h), fullscreen=args.fullscreen, vsync=not args.no_vsync,
                monitor=args.monitor)
    info = d.describe()
    stats = d.measure_flips(args.n)
    d.close()
    expect = 1000.0 / info["refresh_rate"] if info["refresh_rate"] else float("nan")
    print(f"driver={info['driver']} size={info['size']} vsync={info['vsync']} "
          f"refresh={info['refresh_rate']} Hz (period {expect:.2f} ms)")
    print("flip interval: mean {mean_ms:.2f} ms  sd {sd_ms:.2f}  min {min_ms:.2f}  "
          "max {max_ms:.2f}".format(**stats))
    locked = (info["vsync"] and abs(stats["mean_ms"] - expect) < 0.1 * expect
              and stats["sd_ms"] < 1.0)
    print("verdict:", "LOCKED to the refresh" if locked else
          "NOT locked -- try --fullscreen, disable the desktop compositor, or check the GPU driver")


if __name__ == "__main__":
    _selftest()
