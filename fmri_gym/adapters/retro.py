"""stable-retro adapter (NES / SNES / Genesis / GB / ... via libretro).

Maps stable-retro behind the standard EnvAdapter interface:
- keymap: keyboard -> the game's console buttons (MultiBinary action vector);
- per-frame exact savestate via em.get_state()/set_state() (bit-exact, verified);
- state variables: the console RAM plus the game's decoded `info` variables
  (score/lives/... from the integration's data.json), surfaced uniformly.

Notes verified against stable_retro 1.0.1:
- The emulator object is env.unwrapped.em; the libretro RAM view must be
  refreshed with data.update_ram() before get_ram() after a bare set_state.
- Named levels load via env.unwrapped.load_state(name) then reset().
- retro allows only ONE emulator per process; the session opens/closes one env
  per block, so this is respected as long as blocks don't overlap.
"""

from __future__ import annotations

import stable_retro as retro

from .base import EnvAdapter, FrameState, KeySpec

# Keyboard -> console button. Same scheme as the interactive retro player.
# We map by button NAME; each game reports its own button ordering via
# env.buttons, so the adapter builds the action vector for that ordering.
_KEY_TO_BUTTON = {
    "Z": ("BUTTON", "A"), "X": ("B",), "C": ("C",),
    "A": ("X",), "S": ("Y",), "D": ("Z",),
    "Q": ("L",), "W": ("R",),
    "UP": ("UP",), "DOWN": ("DOWN",), "LEFT": ("LEFT",), "RIGHT": ("RIGHT",),
    "RETURN": ("START", "RESET"), "TAB": ("MODE", "SELECT"),
}


class RetroAdapter(EnvAdapter):
    name = "retro"

    def __init__(self, save_pixels: bool = False):
        # save_pixels accepted for interface symmetry; retro frames are already
        # reconstructable from the per-frame state, so pixels aren't stored.
        self.save_pixels = save_pixels

    def make(self, spec):
        return retro.make(
            game=spec["game"], scenario=spec.get("scenario"),
            render_mode="rgb_array")

    def keymap(self, env) -> KeySpec:
        buttons = list(env.unwrapped.buttons)   # e.g. ["B","A","MODE",...,"C"]
        btn_index = {b: i for i, b in enumerate(buttons)}

        def action_for(held_key):
            vec = [0] * len(buttons)
            for target in _KEY_TO_BUTTON.get(held_key, ()):
                if target in btn_index:
                    vec[btn_index[target]] = 1
            return vec

        # Build single-key combos; the session's resolver ORs multiple held keys
        # via most-specific match, but console buttons need true simultaneity, so
        # we instead register per-key vectors and combine them in resolve() below.
        combos = {}
        for key in _KEY_TO_BUTTON:
            vec = action_for(key)
            if any(vec):
                combos[frozenset([key])] = vec
        noop = [0] * len(buttons)
        ks = KeySpec(combos=combos, noop=noop)
        # Override resolve: OR together the button vectors of ALL held keys, so
        # e.g. holding RIGHT+Z fires while moving (multiple buttons at once).
        ks.resolve = _make_multi_resolver(combos, noop)
        return ks

    def reset(self, env, seed, spec):
        state = spec.get("state")
        if state:
            env.unwrapped.load_state(state)
        return env.reset()

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
        u = env.unwrapped
        u.data.update_ram()
        variables = {"ram": u.get_ram().copy()}
        # Surface the game's decoded integration variables (score/lives/...).
        for k, v in (info or {}).items():
            variables[f"info_{k}"] = v
        # em.get_state() is ~1 MB for Genesis; only snapshot on stride frames.
        blob = u.em.get_state() if want_blob else None
        return FrameState(blob=blob, variables=variables)

    def restore(self, env, blob):
        u = env.unwrapped
        u.em.set_state(blob)
        u.data.update_ram()


def _make_multi_resolver(combos, noop):
    """Return a resolve(held) that ORs the button vectors of all held keys."""
    def resolve(held):
        vec = list(noop)
        for keys, action in combos.items():
            (key,) = tuple(keys)
            if key in held:
                for i, a in enumerate(action):
                    if a:
                        vec[i] = 1
        return vec
    return resolve
