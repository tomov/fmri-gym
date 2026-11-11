"""Measure a screen's image with a ruler: a square resized on the screen until it measures
:data:`SQUARE_CM` each way, from which the image's width and height in cm follow.

A size typed from a spec sheet is the likeliest to be wrong (a nominal diagonal, a projector's
screen taken for its image); a ruler held to the image is not. Shown fullscreen on the
participant's screen at its own resolution; each side is resized on its own, so pixels that
are not square are measured too::

    python -m humans.standalone.rig_setup.ruler --monitor 1 [--width-cm 70.8 --height-cm 39.8]

Enter prints the image's ``{"width_cm", "height_cm"}`` as JSON and exits 0; ESC exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys

import pygame

#: The square's side, as it should measure on the screen.
SQUARE_CM = 10.0
#: What the measurement exits with.
MEASURED, STOPPED = 0, 2
#: The square's first side, as a fraction of the image's width, when no size is known.
_FIRST_GUESS = 0.2
#: Pixels an arrow key resizes a side by; with Shift held.
_STEP_PX, _SHIFT_STEP_PX = 1, 10
_INK, _DIM = (235, 235, 235), (150, 150, 150)


def square_px(side_px: tuple[int, int], resolution: tuple[int, int]) -> pygame.Rect:
    """The square, ``side_px`` across and down, centred on a screen of ``resolution``."""
    square = pygame.Rect(0, 0, *side_px)
    square.center = (resolution[0] // 2, resolution[1] // 2)
    return square


def image_cm(side_px: tuple[int, int], resolution: tuple[int, int]) -> dict[str, float]:
    """The image's width and height in cm, when a square ``side_px`` measures
    :data:`SQUARE_CM` each way."""
    return {"width_cm": round(resolution[0] * SQUARE_CM / side_px[0], 1),
            "height_cm": round(resolution[1] * SQUARE_CM / side_px[1], 1)}


def first_side_px(resolution: tuple[int, int], width_cm: float | None,
                  height_cm: float | None) -> tuple[int, int]:
    """The square's side to start from: :data:`SQUARE_CM` by the size given, else a guess."""
    if width_cm and height_cm:
        return (round(resolution[0] * SQUARE_CM / width_cm),
                round(resolution[1] * SQUARE_CM / height_cm))
    side = round(resolution[0] * _FIRST_GUESS)
    return side, side


def draw_square(screen: pygame.Surface, side_px: tuple[int, int]) -> None:
    """The square, what to do, and the image's size it comes to."""
    screen.fill((0, 0, 0))
    font = pygame.font.Font(None, max(24, screen.get_height() // 30))
    square = square_px(side_px, screen.get_size())
    pygame.draw.rect(screen, _INK, square, 2)
    size = image_cm(side_px, screen.get_size())
    lines = ((f"Hold a ruler to the screen and resize the square until it is {SQUARE_CM:g} cm "
              f"wide and {SQUARE_CM:g} cm high.", _INK),
             ("Left/Right: width   Up/Down: height   Shift: faster   Enter: done   ESC: cancel",
              _DIM))
    for i, (line, colour) in enumerate(lines):
        text = font.render(line, True, colour)
        screen.blit(text, text.get_rect(midtop=(screen.get_width() // 2,
                                                  40 + i * font.get_linesize())))
    text = font.render(f"The image is then {size['width_cm']:g} x {size['height_cm']:g} cm",
                       True, _INK)
    screen.blit(text, text.get_rect(midtop=(square.centerx, square.bottom + 16)))
    pygame.display.flip()


def resize_until_measured(screen: pygame.Surface,
                          side_px: tuple[int, int]) -> tuple[int, int] | None:
    """Resize the square with the arrow keys until Enter: its side then; ``None`` on ESC."""
    grow = {pygame.K_RIGHT: (1, 0), pygame.K_LEFT: (-1, 0),
            pygame.K_UP: (0, 1), pygame.K_DOWN: (0, -1)}
    while True:
        draw_square(screen, side_px)
        event = pygame.event.wait()
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                         and event.key == pygame.K_ESCAPE):
            return None
        if event.type != pygame.KEYDOWN:
            continue
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return side_px
        if event.key in grow:
            step = _SHIFT_STEP_PX if event.mod & pygame.KMOD_SHIFT else _STEP_PX
            across, down = grow[event.key]
            side_px = (max(10, side_px[0] + across * step), max(10, side_px[1] + down * step))


def measure_image(monitor: int, width_cm: float | None,
                  height_cm: float | None) -> dict[str, float] | None:
    """Measure the image on ``monitor`` with a ruler; ``None`` when cancelled."""
    pygame.display.init()
    pygame.font.init()
    try:
        resolution = pygame.display.get_desktop_sizes()[monitor]
        screen = pygame.display.set_mode(resolution, pygame.FULLSCREEN, display=monitor)
        side_px = resize_until_measured(screen, first_side_px(resolution, width_cm, height_cm))
        return None if side_px is None else image_cm(side_px, resolution)
    finally:
        pygame.quit()


def main() -> None:
    """``python -m humans.standalone.rig_setup.ruler``: the module docstring."""
    p = argparse.ArgumentParser(description="Measure a screen's image with a ruler: resize a "
                                            "square on it until it measures as labelled.")
    p.add_argument("--monitor", type=int, default=0, help="the participant's screen, by index")
    p.add_argument("--width-cm", type=float, help="the image's width so far, to start from")
    p.add_argument("--height-cm", type=float, help="the image's height so far, to start from")
    args = p.parse_args()
    measured = measure_image(args.monitor, args.width_cm, args.height_cm)
    if measured is None:
        sys.exit(STOPPED)
    print(json.dumps(measured))
    sys.exit(MEASURED)


if __name__ == "__main__":
    main()
