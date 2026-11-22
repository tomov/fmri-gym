"""Draw each game's controls image: a gamepad drawing with the controls it uses named by their actions.

The labels file names the underlay (a drawing of an Xbox 360 controller, whose
controls sit where :data:`CONTROLS` says) and maps each image, by its path from
the file, to its labels: one control name (as a curriculum writes it) to the
lines of text beside it, a list among them being lines centered on each other. The
controls a game leaves unused are greyed out, as are the drawing's controls no controller
rig has, and the names printed on it are painted over.
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
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from humans.inputs.control_names import ControlNames  # noqa: E402

PAPER, TEXT, LEADER = (255, 255, 255), (25, 25, 25), (70, 70, 70)
#: the part of the underlay that is the controller, and its scale in the image
CROP, SCALE = pygame.Rect(50, 50, 1180, 800), 0.75
#: room for the labels on each side of the controller, and under it
MARGIN, BELOW = 510, 90
#: the space between two controls' labels, more than between a label's lines so each
#: label reads as one; and the least space between a label and the image's edge
LABEL_GAP, EDGE = 46, 10


#: what an unused control fades to: its pixels keep this share of their contrast with
#: the body's grey around it (this one when none is around), and lose their colour;
#: the white around the drawing stays white
GREYED_OUT_CONTRAST, GREYED_OUT_GREY = 0.2, 205


@dataclass(frozen=True)
class Control:
    """Where one control is on the underlay, in its own pixels."""

    #: the area greyed out when the game leaves the control unused
    area: pygame.Rect
    #: whether the area is the ellipse in that rect (a button, a stick) or the rect itself
    is_round: bool
    #: where the leader line to its label starts
    anchor: tuple[int, int]
    #: which side of the controller its label goes: "left", "right" or "below"
    side: str
    #: a label of several lines grows upward from its last line, beside the control,
    #: rather than both ways from its middle (the top label of a column)
    grows_up: bool = False


#: each control a label may name, and the stick as a whole as ``LEFT_STICK``
CONTROLS = {
    ControlNames.CONTROLLER_AXIS_TRIGGERLEFT:
        Control(pygame.Rect(285, 65, 77, 103), False, (285, 150), "left"),
    ControlNames.CONTROLLER_AXIS_TRIGGERRIGHT:
        Control(pygame.Rect(920, 65, 82, 103), False, (995, 150), "right"),
    "LEFT_STICK": Control(pygame.Rect(138, 300, 288, 285), True, (205, 456), "left"),
    ControlNames.CONTROLLER_BUTTON_Y:
        Control(pygame.Rect(940, 296, 110, 104), True, (1037, 345), "right", grows_up=True),
    ControlNames.CONTROLLER_BUTTON_B:
        Control(pygame.Rect(1038, 380, 102, 92), True, (1123, 421), "right"),
    ControlNames.CONTROLLER_BUTTON_A:
        Control(pygame.Rect(936, 460, 104, 92), True, (1024, 503), "right"),
    ControlNames.CONTROLLER_BUTTON_X:
        Control(pygame.Rect(846, 388, 102, 90), True, (896, 459), "below"),
}
#: the drawing's controls no controller rig has, always greyed out: (area, is_round), an
#: area being a rect or the corners of a polygon
OTHER_CONTROLS = {
    "LEFT_BUMPER": ([(166, 222), (283, 182), (288, 165), (374, 165), (394, 196), (394, 236),
                     (172, 262)], False),
    "RIGHT_BUMPER": ([(1114, 222), (997, 182), (992, 165), (906, 165), (886, 196), (886, 236),
                      (1108, 262)], False),
    "DPAD": (pygame.Rect(318, 470, 294, 248), True),
    "RIGHT_STICK": (pygame.Rect(660, 470, 286, 242), True),
    "BACK": (pygame.Rect(472, 396, 80, 62), True),
    "START": (pygame.Rect(729, 396, 80, 62), True),
    "GUIDE": (pygame.Rect(560, 356, 158, 134), True),
}
#: the names printed on the drawing (LT, LB, RT, RB, BACK, START, 1-4), painted over
PRINTED_NAMES = [pygame.Rect(*r) for r in (
    (298, 145, 36, 18), (285, 171, 50, 21), (948, 145, 37, 19), (953, 170, 45, 20),
    (458, 378, 58, 36), (760, 378, 64, 36),
    (572, 363, 14, 20), (692, 363, 16, 20), (572, 474, 14, 19), (692, 474, 14, 19))]


def _on_image(point: tuple[int, int]) -> tuple[int, int]:
    """An underlay point, in the controls image's pixels."""
    return (MARGIN + round((point[0] - CROP.x) * SCALE), round((point[1] - CROP.y) * SCALE))


def _area_mask(area: pygame.Rect | list[tuple[int, int]],
               is_round: bool) -> tuple[pygame.Rect, np.ndarray]:
    """An area's bounding rect, and which of its pixels (indexed x, y) are in the area."""
    if isinstance(area, list):
        xs, ys = zip(*area)
        bounds = pygame.Rect(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
        stencil = pygame.Surface(bounds.size)
        pygame.draw.polygon(stencil, (255, 255, 255), [(x - bounds.x, y - bounds.y) for x, y in area])
        return bounds, pygame.surfarray.array_red(stencil) > 0
    if not is_round:
        return area, np.ones(area.size, bool)
    x = (np.arange(area.width) + 0.5) / area.width * 2 - 1
    y = (np.arange(area.height) + 0.5) / area.height * 2 - 1
    return area, x[:, None] ** 2 + y[None, :] ** 2 <= 1


def _grey_out(pixels: np.ndarray, area: pygame.Rect | list[tuple[int, int]],
              is_round: bool) -> None:
    """Fade the drawing in ``area`` to the colourless grey of the body around it.

    Each pixel keeps :data:`GREYED_OUT_CONTRAST` of its contrast with that grey, which is
    the median of the area's edge, so the area's border does not show.
    """
    area, mask = _area_mask(area, is_round)
    region = pixels[area.left:area.right, area.top:area.bottom]
    luminance = region @ np.array([0.299, 0.587, 0.114])
    inside = mask & (luminance < 250)
    edge = inside & ~(np.roll(mask, 3, 0) & np.roll(mask, -3, 0)
                      & np.roll(mask, 3, 1) & np.roll(mask, -3, 1))
    body = np.median(luminance[edge]) if edge.any() else GREYED_OUT_GREY
    faded = body + (luminance - body) * GREYED_OUT_CONTRAST
    region[inside] = np.round(faded[inside])[:, None]


def _paint_over(pixels: np.ndarray, rect: pygame.Rect) -> None:
    """Fill ``rect`` with the colour most of its edge has: the surface the text is printed on."""
    region = pixels[rect.left:rect.right, rect.top:rect.bottom]
    edge = np.concatenate([region[0], region[-1], region[:, 0], region[:, -1]])
    region[:] = np.median(edge, axis=0).astype(np.uint8)


def _render_label(font: pygame.font.Font, lines: list[str | list[str]],
                  side: str) -> pygame.Surface:
    """``lines`` one under another on a transparent surface, flush toward the controller from
    ``side``; a list among them is a group of lines centered on each other, set as one line is."""
    groups = []
    for line in lines:
        texts = [line] if isinstance(line, str) else line
        rows = [font.render(text, True, TEXT) for text in texts]
        group = pygame.Surface((max(r.get_width() for r in rows),
                                sum(r.get_height() for r in rows)), pygame.SRCALPHA)
        top = 0
        for row in rows:
            group.blit(row, row.get_rect(midtop=(group.get_width() // 2, top)))
            top += row.get_height()
        groups.append(group)
    label = pygame.Surface((max(g.get_width() for g in groups),
                            sum(g.get_height() for g in groups)), pygame.SRCALPHA)
    top = 0
    for group in groups:
        x = {"left": label.get_width() - group.get_width(), "right": 0,
             "below": (label.get_width() - group.get_width()) // 2}[side]
        label.blit(group, (x, top))
        top += group.get_height()
    return label


def _label_tops(canvas_height: int, wanted_tops: list[int], heights: list[int]) -> list[int]:
    """The tops of one side's labels, given top to bottom: each as near the top it wants as
    keeps :data:`LABEL_GAP` between labels and all of them on the canvas."""
    tops = []
    for wanted, height in zip(wanted_tops, heights):
        floor = tops[-1] + heights[len(tops) - 1] + LABEL_GAP if tops else EDGE
        tops.append(max(wanted, floor))
    ceiling = canvas_height - EDGE
    for i in reversed(range(len(tops))):
        tops[i] = min(tops[i], ceiling - heights[i])
        ceiling = tops[i] - LABEL_GAP
    return tops


def draw_labels(canvas: pygame.Surface, font: pygame.font.Font,
                labels: dict[str, list[str | list[str]]]) -> None:
    """Each control's lines beside it, with a leader line from the control to them."""
    by_side: dict[str, list[tuple[Control, pygame.Surface]]] = {}
    for name, lines in labels.items():
        control = CONTROLS[name]
        label = _render_label(font, lines, control.side)
        by_side.setdefault(control.side, []).append((control, label))
    for side, placed in by_side.items():
        placed.sort(key=lambda control_and_label: control_and_label[0].anchor[1])
        anchors = [_on_image(control.anchor) for control, _ in placed]
        heights = [label.get_height() for _, label in placed]
        tops = [_wanted_top(control, anchor, label, font, canvas.get_height())
                for (control, label), anchor in zip(placed, anchors)]
        tops = _label_tops(canvas.get_height(), tops, heights)
        for (control, label), anchor, top in zip(placed, anchors, tops):
            _draw_label(canvas, side, anchor, label, top)


def _wanted_top(control: Control, anchor: tuple[int, int], label: pygame.Surface,
                font: pygame.font.Font, canvas_height: int) -> int:
    """Where ``label``'s top goes beside ``control`` when no other label is in the way: level
    with it, or, under the controller, at the image's bottom edge."""
    if control.side == "below":
        return canvas_height - EDGE - label.get_height()
    if control.grows_up:
        return anchor[1] - label.get_height() + font.get_linesize() // 2
    return anchor[1] - label.get_height() // 2


def _draw_label(canvas: pygame.Surface, side: str, anchor: tuple[int, int],
                label: pygame.Surface, top: int) -> None:
    """``label`` at ``top`` on ``side``, and the leader line from ``anchor`` to it."""
    rect = label.get_rect()
    if side == "below":
        rect.midtop = (anchor[0], top)
        end = (anchor[0], rect.top - 6)
    else:
        if side == "left":
            rect.topright = (MARGIN - 20, top)
        else:
            rect.topleft = (canvas.get_width() - MARGIN + 20, top)
        # straight across when the label is level with the control, else to its nearest line
        end = (rect.right + 10 if side == "left" else rect.left - 10,
               min(max(anchor[1], rect.top + 20), rect.bottom - 20))
    pygame.draw.line(canvas, LEADER, anchor, end, 3)
    pygame.draw.circle(canvas, LEADER, anchor, 6)
    canvas.blit(label, rect)


def write_controls_image(underlay: pygame.Surface, labels: dict[str, list[str | list[str]]],
                         path: Path) -> None:
    """Draw ``underlay`` with ``labels`` (control name: lines) and save it as a PNG at ``path``.

    :raises ValueError: if a label names a control the underlay's map does not have.
    """
    unknown = set(labels) - set(CONTROLS)
    if unknown:
        raise ValueError(f"{path.name}: unknown control(s) {sorted(unknown)}; the controls are "
                         f"{sorted(CONTROLS)}")
    drawing = underlay.copy()
    pixels = pygame.surfarray.pixels3d(drawing)
    for rect in PRINTED_NAMES:
        _paint_over(pixels, rect)
    unused = [(c.area, c.is_round) for name, c in CONTROLS.items() if name not in labels]
    for area, is_round in unused + list(OTHER_CONTROLS.values()):
        _grey_out(pixels, area, is_round)
    del pixels
    controller = drawing.subsurface(CROP).copy()
    size = (round(CROP.width * SCALE), round(CROP.height * SCALE))
    controller = pygame.transform.smoothscale(controller, size)
    canvas = pygame.Surface((size[0] + 2 * MARGIN, size[1] + BELOW))
    canvas.fill(controller.get_at((0, 0)))     # smoothscale leaves the white a shade darker
    canvas.blit(controller, (MARGIN, 0))
    font = pygame.font.Font(pygame.font.get_default_font(), 42)
    draw_labels(canvas, font, labels)
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
