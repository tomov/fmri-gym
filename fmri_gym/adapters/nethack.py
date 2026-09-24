"""NetHack adapter (nle, the NetHack Learning Environment).

Distinct from the `minihack` backend: base NLE (`NetHack*-v0`) does NOT provide a
`pixel` observation -- only the ASCII terminal (`tty_chars` / `tty_colors`, a
24x80 grid) plus `glyphs`/`blstats`/`message`, with a Discrete(23) action space
whose values are ASCII keycodes (NetHack's vi-keys: k/l/j/h = N/E/S/W, etc.).

NetHack is a terminal game, so we render the TTY buffer to a pixel frame (a
monospace text grid) for display -- faithful to how the game actually looks.
Arrow keys map to the 4 cardinal movement actions; `blstats` (score, HP, depth,
...) are logged as analysis variables. No pixel obs and no savestate over the
gym API -> reconstruction is seed + action replay.

Requires: `pip install nle` (already present if minihack is installed) and
`setuptools<81` (pkg_resources).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# NetHack TTY palette (16 colors), indexed by tty_colors (0..15).
_TTY_PALETTE = np.array([
    (0, 0, 0), (170, 0, 0), (0, 170, 0), (170, 85, 0),
    (0, 0, 170), (170, 0, 170), (0, 170, 170), (170, 170, 170),
    (85, 85, 85), (255, 85, 85), (85, 255, 85), (255, 255, 85),
    (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
], dtype=np.uint8)

# Arrow-key names -> the ASCII keycode NLE uses for that compass move
# (vi-keys: h=west 104, j=south 106, k=north 107, l=east 108).
_ARROW_TO_KEYCODE = {"UP": 107, "RIGHT": 108, "DOWN": 106, "LEFT": 104}


class NetHackAdapter(EnvAdapter):
    name: str = "nethack"

    def _make(self, spec: dict) -> gym.Env:
        import nle  # noqa: F401  (registers NetHack*-v0 env ids)
        env = gym.make(spec.get("game", "NetHackScore-v0"))
        # Map each arrow's target keycode to its Discrete action index (the
        # action list holds the keycodes as its values/enum).
        actions = list(env.unwrapped.actions)
        code_to_idx = {int(a): i for i, a in enumerate(actions)}
        self._key_to_action = {}
        for name, code in _ARROW_TO_KEYCODE.items():
            if code in code_to_idx:
                self._key_to_action[name] = code_to_idx[code]
        # ENTER (13) is handy for menus/prompts.
        if 13 in code_to_idx:
            self._key_to_action["RETURN"] = code_to_idx[13]
        self._cell = spec.get("cell_px", 10)  # pixel size of one TTY cell
        self._last = None
        return env

    def _keyspec(self) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in self._key_to_action.items()}
        # noop: NLE has no true no-op; default to the first action.
        return SingleKeySpec(combos=combos, noop=0)

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed)
        self._last = obs
        return obs, info

    def render(self) -> np.ndarray:
        return _tty_to_rgb(self._last, self._cell)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        self._last = obs
        variables = {}
        for k in ("blstats", "glyphs", "message"):
            if isinstance(obs, dict) and k in obs:
                variables[k] = np.asarray(obs[k])
        return FrameState(blob=None, variables=variables)

    def rich_state(self, obs: Any, info: dict) -> dict | None:
        """Thin wrapper so Session (which calls ``adapter.rich_state(obs,
        info)`` by name) finds this hook -- see :meth:`get_rich_state`."""
        return self.get_rich_state(obs, info)

    def get_rich_state(self, obs: Any, info: dict) -> dict | None:
        """See :func:`nle_rich_state` (shared with :mod:`.minihack`, the
        other NLE-based adapter): decodes ``blstats`` into named fields,
        the current text message, inventory, and every visible monster/
        object/trap with its NetHack-assigned plain-English description --
        everything base NLE's default observation keys (``glyphs`` /
        ``blstats`` / ``chars`` / ``inv_*`` / ``screen_descriptions``,
        all present without any extra ``gym.make`` configuration for this
        backend) make available beyond the raw arrays :meth:`capture`
        already logs.

        :param obs: the latest observation dict.
        :param info: unused -- NLE's own state lives entirely in ``obs``.
        """
        return nle_rich_state(obs)


def _tty_to_rgb(obs: Any, cell: int) -> np.ndarray:
    """Render NLE's (24,80) tty_chars/tty_colors grid to an RGB image."""
    if not isinstance(obs, dict) or "tty_chars" not in obs:
        return np.zeros((240, 800, 3), dtype=np.uint8)
    import pygame
    if not pygame.font.get_init():
        pygame.font.init()
    chars = np.asarray(obs["tty_chars"])
    colors = np.asarray(obs["tty_colors"])
    rows, cols = chars.shape
    # Size a monospace font by cell height, then MEASURE its actual glyph box
    # and lay the grid out on exactly that pitch -- so cells never overlap
    # (the previous bug) and there are no gaps. Terminal cells are taller than
    # wide, which the measured (gw, gh) naturally reproduces.
    font = _mono_font(pygame, int(cell * 2))
    gw, gh = font.size("W")
    surf = pygame.Surface((cols * gw, rows * gh))
    surf.fill((0, 0, 0))
    for r in range(rows):
        for c in range(cols):
            ch = int(chars[r, c])
            if ch in (0, 32):  # null / space
                continue
            col = _TTY_PALETTE[int(colors[r, c]) & 0x0F]
            glyph = font.render(chr(ch), True, tuple(int(x) for x in col))
            # center the glyph horizontally in its cell (glyph width may be < gw)
            surf.blit(glyph, (c * gw + (gw - glyph.get_width()) // 2, r * gh))
    return pygame.surfarray.array3d(surf).transpose(1, 0, 2)


def _nle_text(buf: Any) -> str:
    """Decode one NLE fixed-width, null-terminated byte buffer (``message``,
    an ``inv_strs`` row, one ``screen_descriptions`` cell) to a plain str."""
    return bytes(np.asarray(buf)).split(b"\x00", 1)[0].decode("ascii", "ignore")


def _nle_blstats(obs: dict) -> dict[str, int]:
    """Decode the raw ``blstats`` array into ``{name: value}`` using NLE's
    own field-index constants (``nethack.NLE_BL_*``) -- the array alone
    doesn't say which slot is HP vs. depth vs. gold."""
    from nle import nethack
    arr = np.asarray(obs["blstats"])
    stats = {}
    for name in dir(nethack):
        if not name.startswith("NLE_BL_"):
            continue
        idx = getattr(nethack, name)
        if idx < len(arr):
            stats[name[len("NLE_BL_"):]] = int(arr[idx])
    return stats


def _nle_inventory(obs: dict) -> list[dict]:
    """Every carried item (letter/object-class/glyph/description), decoded
    from the ``inv_*`` observation keys -- empty ``letter`` (``0``) slots
    are unused inventory rows, skipped."""
    if "inv_letters" not in obs:
        return []
    letters = obs["inv_letters"]
    oclasses = obs.get("inv_oclasses", [0] * len(letters))
    glyphs = obs.get("inv_glyphs", [0] * len(letters))
    strs = obs.get("inv_strs", [None] * len(letters))
    items = []
    for letter, oclass, glyph, desc in zip(letters, oclasses, glyphs, strs):
        if int(letter) == 0:
            continue
        items.append({
            "letter": chr(int(letter)), "object_class": int(oclass),
            "glyph": int(glyph),
            "description": _nle_text(desc) if desc is not None else "",
        })
    return items


def _nle_entities(obs: dict) -> list[dict]:
    """Every visible monster/object/trap (not plain terrain), with its
    ``(row, col)`` and NetHack's own plain-English description -- the
    closest thing here to ViZDoom's ``labels`` or Crafter's ``semantic``
    map: ground truth about what's actually in view, not just a glyph id."""
    if "glyphs" not in obs:
        return []
    from nle import nethack
    glyphs = np.asarray(obs["glyphs"])
    descs = np.asarray(obs["screen_descriptions"]) if "screen_descriptions" in obs else None
    entities = []
    for y in range(glyphs.shape[0]):
        for x in range(glyphs.shape[1]):
            g = int(glyphs[y, x])
            if (nethack.glyph_is_monster(g) or nethack.glyph_is_object(g)
                    or nethack.glyph_is_trap(g)):
                entry = {"position": [y, x], "glyph": g}
                if descs is not None:
                    entry["description"] = _nle_text(descs[y, x])
                entities.append(entry)
    return entities


def nle_rich_state(obs: Any) -> dict | None:
    """Shared by :class:`NetHackAdapter` and :class:`.minihack.MiniHackAdapter`
    (both NLE-based): decode ``blstats`` into named fields, the current
    message, carried inventory, and every visible monster/object/trap --
    everything NLE's observation dict makes available beyond the raw
    ``glyphs``/``blstats``/``message`` arrays :meth:`EnvAdapter.capture`
    already logs as opaque arrays.

    :param obs: the latest observation dict (``None``/non-dict before the
        first ``reset()``, or if a curriculum's ``observation_keys``
        override dropped ``blstats`` entirely).
    :return: the dict described above, or ``None`` if ``obs`` has no
        ``blstats`` to decode.
    """
    if not isinstance(obs, dict) or "blstats" not in obs:
        return None
    return {
        "blstats": _nle_blstats(obs),
        "message": _nle_text(obs.get("message", b"")),
        "inventory": _nle_inventory(obs),
        "entities": _nle_entities(obs),
    }


def _mono_font(pygame: Any, size: int) -> Any:
    """A real fixed-width font (falls back gracefully across systems)."""
    for name in ("dejavusansmono", "liberationmono", "couriernew", "monospace"):
        try:
            f = pygame.font.SysFont(name, size)
            if f is not None:
                return f
        except Exception:
            continue
    return pygame.font.Font(pygame.font.get_default_font(), size)
