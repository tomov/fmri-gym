"""Fixed-size pygame display: aspect-fit game frames, fixation, text, survey.

Engine-agnostic -- it only ever receives RGB numpy frames (H,W,3) from
env.render(), so ALE / stable-retro / any gym env all present identically:
letterboxed and centered in one fixed window, giving uniform screen geometry.
"""

from __future__ import annotations

import pygame

BG_COLOR = (0, 0, 0)
TEXT_COLOR = (220, 220, 220)
FIX_COLOR = (255, 255, 255)


class Display:
    def __init__(self, size=(1024, 768), fullscreen=False, caption="fmri-gym"):
        self._req_size = size
        self._flags = pygame.FULLSCREEN if fullscreen else 0
        self._caption = caption
        self._init_display()

    def _init_display(self):
        pygame.init()
        pygame.mouse.set_visible(False)
        self.screen = pygame.display.set_mode(self._req_size, self._flags)
        pygame.display.set_caption(self._caption)
        self.size = self.screen.get_size()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 28)
        self.fix_font = pygame.font.Font(pygame.font.get_default_font(), 80)

    def ensure(self):
        """Re-create the window if something (e.g. a gym env's close(), which
        calls pygame.display.quit()) tore down the shared pygame display."""
        if not pygame.display.get_init() or not pygame.get_init():
            self._init_display()

    def draw_frame(self, rgb):
        """Blit an RGB frame (H,W,3), aspect-fit and centered with black pad."""
        self.screen.fill(BG_COLOR)
        h, w = rgb.shape[:2]
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))  # -> (W,H)
        scale = min(self.size[0] / w, self.size[1] / h)
        dw, dh = int(w * scale), int(h * scale)
        surf = pygame.transform.scale(surf, (dw, dh))
        rect = surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        self.screen.blit(surf, rect.topleft)
        pygame.display.flip()

    def _wrap(self, font, line, max_w):
        """Word-wrap one logical line so no rendered line exceeds max_w px.

        Leading whitespace (used for indented controls tables) is preserved on
        the first physical row. Very long single words are left intact.
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

    def draw_text(self, text, color=TEXT_COLOR, font=None, align="center"):
        """Render multi-line text, block-centered vertically.

        align="center" centers each line horizontally (default, for messages);
        align="left" left-aligns all lines against a common left edge so the
        whole block is horizontally centered (good for controls tables).
        Long lines are word-wrapped to fit the window.
        """
        font = font or self.font
        self.screen.fill(BG_COLOR)
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
            self.screen.blit(surf, rect)
            y += surf.get_height()
        pygame.display.flip()

    def draw_fixation(self):
        self.draw_text("+", color=FIX_COLOR, font=self.fix_font)

    def close(self):
        pygame.quit()
