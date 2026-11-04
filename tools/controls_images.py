"""Draw each game's controls image: a gamepad drawing with the controls it uses named by their actions.

The labels file names the underlay (a drawing of an Xbox 360 controller, whose
controls sit where :data:`CONTROLS` says) and maps each image, by its path from
the file, to its labels: one control name (as a curriculum writes it) to the
lines of text beside it. The controls a game leaves unused are washed out.
A key starting with ``##`` is a comment.

    uv run python tools/controls_images.py configs/images/controls/labels.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

PAPER, TEXT, LEADER = (255, 255, 255), (25, 25, 25), (70, 70, 70)
#: the part of the underlay that is the controller, and its scale in the image
CROP, SCALE = pygame.Rect(50, 50, 1180, 800), 0.75
#: room for the labels on each side of the controller, and under it
MARGIN, BELOW = 420, 90


@dataclass(frozen=True)
class Control:
    """Where one control is on the underlay, in its own pixels."""

    #: the area washed out when the game leaves the control unused
    area: pygame.Rect
    #: where the leader line to its label starts
    anchor: tuple[int, int]
    #: which side of the controller its label goes: "left", "right" or "below"
    side: str


CONTROLS = {
    "LEFT_TRIGGER": Control(pygame.Rect(285, 69, 75, 95), (285, 150), "left"),
    "RIGHT_TRIGGER": Control(pygame.Rect(920, 69, 75, 95), (995, 150), "right"),
    "LEFT_STICK": Control(pygame.Rect(200, 375, 165, 165), (205, 456), "left"),
    "BUTTON_Y": Control(pygame.Rect(950, 305, 90, 82), (1037, 345), "right"),
    "BUTTON_B": Control(pygame.Rect(1050, 390, 76, 66), (1123, 421), "right"),
    "BUTTON_A": Control(pygame.Rect(949, 472, 78, 64), (1024, 503), "right"),
    "BUTTON_X": Control(pygame.Rect(859, 400, 76, 61), (896, 459), "below"),
}


def _on_image(point: tuple[int, int]) -> tuple[int, int]:
    """An underlay point, in the controls image's pixels."""
    return (MARGIN + round((point[0] - CROP.x) * SCALE), round((point[1] - CROP.y) * SCALE))


def draw_label(canvas: pygame.Surface, font: pygame.font.Font, control: Control,
               lines: list[str]) -> None:
    """``lines`` beside ``control``, with a leader line from the control to them."""
    anchor = _on_image(control.anchor)
    surfaces = [font.render(line, True, TEXT) for line in lines]
    height = sum(s.get_height() for s in surfaces)
    if control.side == "below":
        top = canvas.get_height() - height - 10
        end = (anchor[0], top - 6)
    else:
        top = anchor[1] - height // 2
        x = MARGIN - 20 if control.side == "left" else canvas.get_width() - MARGIN + 20
        end = (x + 10, anchor[1]) if control.side == "left" else (x - 10, anchor[1])
    pygame.draw.line(canvas, LEADER, anchor, end, 3)
    pygame.draw.circle(canvas, LEADER, anchor, 6)
    for surface in surfaces:
        rect = surface.get_rect()
        if control.side == "below":
            rect.midtop = (anchor[0], top)
        elif control.side == "left":
            rect.topright = (MARGIN - 20, top)
        else:
            rect.topleft = (canvas.get_width() - MARGIN + 20, top)
        canvas.blit(surface, rect)
        top += surface.get_height()


def write_controls_image(underlay: pygame.Surface, labels: dict[str, list[str]],
                         path: Path) -> None:
    """Draw ``underlay`` with ``labels`` (control name: lines) and save it as a PNG at ``path``.

    :raises ValueError: if a label names a control the underlay's map does not have.
    """
    unknown = set(labels) - set(CONTROLS)
    if unknown:
        raise ValueError(f"{path.name}: unknown control(s) {sorted(unknown)}; the controls are "
                         f"{sorted(CONTROLS)}")
    controller = underlay.subsurface(CROP).copy()
    wash = pygame.Surface(controller.get_size(), pygame.SRCALPHA)
    for name, control in CONTROLS.items():
        if name not in labels:
            pygame.draw.ellipse(wash, (*PAPER, 190), control.area.move(-CROP.x, -CROP.y))
    controller.blit(wash, (0, 0))
    size = (round(CROP.width * SCALE), round(CROP.height * SCALE))
    controller = pygame.transform.smoothscale(controller, size)
    canvas = pygame.Surface((size[0] + 2 * MARGIN, size[1] + BELOW))
    canvas.fill(controller.get_at((0, 0)))     # smoothscale leaves the white a shade darker
    canvas.blit(controller, (MARGIN, 0))
    font = pygame.font.Font(pygame.font.get_default_font(), 42)
    for name, lines in labels.items():
        draw_label(canvas, font, CONTROLS[name], lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(canvas, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels_file", type=Path,
                        help='JSON: {"underlay": path, "images": {path: {control: lines}}}')
    args = parser.parse_args()
    spec = json.loads(args.labels_file.read_text())
    pygame.init()
    pygame.display.set_mode((1, 1))
    underlay = pygame.image.load(str(args.labels_file.parent / spec["underlay"])).convert()
    for image, labels in spec["images"].items():
        if image.startswith("##"):
            continue
        path = args.labels_file.parent / image
        write_controls_image(underlay, labels, path)
        print(path)


if __name__ == "__main__":
    main()
