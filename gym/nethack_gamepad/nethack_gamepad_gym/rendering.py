"""One desktop canvas using upstream NetHack tiles and the supplied DS bitmap font."""

from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Renderer:
    """Display native window contents; no authored gameplay text or command definitions."""

    def __init__(self, runtime: Path) -> None:
        self.atlas = Image.open(runtime / "tiles.bmp").convert("RGB")
        self.font = ImageFont.load(str(runtime / "font.pil"))
        self.tiles: dict[int, Image.Image] = {}
        self.canvas = Image.new("RGB", (1280, 720))

    def _text(self, text: str, x: int, y: int, color=(235, 235, 235), scale=2) -> None:
        text = text.encode("latin-1", errors="replace").decode("latin-1")
        image = Image.new("RGBA", (max(1, len(text) * 5), 12))
        ImageDraw.Draw(image).text((0, 0), text, font=self.font, fill=(*color, 255))
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        self.canvas.paste(image, (x, y), image)

    def _tile(self, index: int) -> Image.Image:
        if index not in self.tiles:
            columns = self.atlas.width // 16
            x, y = (index % columns) * 16, (index // columns) * 16
            if y + 16 > self.atlas.height:
                raise ValueError(f"Native tile {index} exceeds the matching upstream atlas")
            self.tiles[index] = self.atlas.crop((x, y, x + 16, y + 16))
        return self.tiles[index]

    def draw(self, session, controller) -> np.ndarray:
        """Render the map and the current native input boundary on one screen."""
        self.canvas = Image.new("RGB", (1280, 720))
        has_map = any(tile >= 0 for row in session.tiles for tile in row)
        if has_map:
            statuses = [window for window in session.windows.values() if window["type"] == 2]
            lines = statuses[0]["lines"] if statuses else []
            for row, line in enumerate(lines[:2]):
                self._text(line, 16, 8 + row * 22)
            for y, row in enumerate(session.tiles):
                for x, tile in enumerate(row):
                    if tile >= 0:
                        self.canvas.paste(self._tile(tile), (8 + x * 16, 56 + y * 16))
            # Preserve the native map cursor, including when the hero is invisible.
            draw = ImageDraw.Draw(self.canvas)
            for window in session.windows.values():
                x, y = window.get("column", 0), window["row"]
                if window["type"] == 3 and 1 <= x <= 79 and 0 <= y < 21:
                    left, top = 8 + (x - 1) * 16, 56 + y * 16
                    draw.rectangle((left, top, left + 15, top + 15), outline=(255, 255, 255))
            message = " ".join(session.messages[-2:])
            for row, line in enumerate(textwrap.wrap(message, 124)[:2]):
                self._text(line, 16, 404 + row * 22)
        else:
            self._text("NetHack 3.6.0", 16, 16, scale=3)
        if has_map:
            self._text(f"Mode: {session.mode}   Revivals: {session.revivals}",
                       16, 454, (160, 190, 210), scale=2)
        title_y = 482 if has_map else 70
        self._text(controller.title[:124], 16, title_y, (190, 220, 255))
        body_y = title_y + 32
        if controller.kind in ("text", "keyboard"):
            self._keyboard(controller, body_y)
        elif controller.kind == "ack" or session.request.get("how") == 0:
            self._document(controller, body_y, 6 if has_map else 19)
        else:
            self._menu(controller, body_y, 4 if has_map else 17)
        footer = ("Arrows: move cursor  Z: select location  X: cancel  C: keyboard"
                  if controller.kind == "position" else
                  "Arrows: move/select  Z: confirm/pick up  X: cancel/search  C: commands/keys  V: inventory")
        self._text(footer, 16, 694)
        return np.array(self.canvas, dtype=np.uint8)

    def _menu(self, controller, y: int, rows: int) -> None:
        entries = controller.entries()
        if not entries:
            return
        columns = 3 if controller.kind == "direction" else 1
        capacity = rows * columns
        start = controller.cursor // capacity * capacity
        width = 1240 // columns
        draw = ImageDraw.Draw(self.canvas)
        for index in range(start, min(start + capacity, len(entries))):
            row, column = divmod(index - start, columns)
            x, top = 20 + column * width, y + row * 32
            if index == controller.cursor:
                draw.rectangle((x - 4, top, x + width - 12, top + 27), fill=(28, 60, 105))
            entry = entries[index]
            prefix = "+ " if entry.value in controller.selected else ""
            self._text((prefix + entry.label)[:width // 10 - 2], x, top,
                       (235, 235, 235) if entry.enabled else (160, 160, 160))
        self._text(f"{controller.cursor + 1}/{len(entries)}", 16, 666, scale=1)

    def _document(self, controller, y: int, rows: int) -> None:
        lines = controller.document_lines
        for index, line in enumerate(lines[controller.scroll:controller.scroll + rows]):
            self._text(line, 20, y + index * 24)
        self._text("Arrows: scroll   Z: continue", 16, 666, scale=1)

    def _keyboard(self, controller, y: int) -> None:
        self._text(controller.typed[-120:], 20, y, (255, 225, 130))
        y += 30
        draw = ImageDraw.Draw(self.canvas)
        index = 0
        for row, keys in enumerate(controller.keyboard_rows()):
            pitch = 1240 / sum(key.width for key in keys)
            x = 20.0
            for key in keys:
                width = key.width * pitch
                top = y + row * 25
                if index == controller.cursor:
                    draw.rectangle((int(x), top, int(x + width - 3), top + 23), fill=(28, 60, 105))
                self._text(key.label, int(x + 3), top)
                x += width
                index += 1
        modes = [name for name in ("shift", "caps", "ctrl", "alt") if getattr(controller, name)]
        self._text(" ".join(modes), 20, 678, scale=1)
