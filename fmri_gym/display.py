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

    def draw_text(self, text, color=TEXT_COLOR, font=None):
        font = font or self.font
        self.screen.fill(BG_COLOR)
        lines = text.split("\n")
        total_h = sum(font.size(ln)[1] for ln in lines)
        y = (self.size[1] - total_h) // 2
        for ln in lines:
            surf = font.render(ln, True, color)
            rect = surf.get_rect(center=(self.size[0] // 2, y + surf.get_height() // 2))
            self.screen.blit(surf, rect)
            y += surf.get_height()
        pygame.display.flip()

    def draw_fixation(self):
        self.draw_text("+", color=FIX_COLOR, font=self.fix_font)

    def close(self):
        pygame.quit()
