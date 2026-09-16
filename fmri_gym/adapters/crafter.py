"""Crafter adapter (danijar/crafter).

Crafter uses the OLD gym API shape -- reset() returns obs only, step() returns a
4-tuple (obs, reward, done, info) with no `truncated`, and it doesn't register
cleanly under gymnasium. We wrap crafter.Env directly and normalize it to the
gymnasium contract the session loop expects.

The observation IS the RGB frame (default 64x64x3; bump via env_kwargs.size), so
render() just returns obs. Choose a `size` that is an integer fraction of the
display height (384 -> x2 in a 1024x768 window): display.draw_frame scales
unfiltered, so a fractional factor smears crafter's pixel art, and the HUD --
four status icons plus 16 inventory counters, drawn by the env itself into the
bottom two of the nine tile rows -- is the part that suffers first.

Two traps, both about reproducibility:

- `crafter.Env.reset()` takes no seed. The world seed is
  hash((env seed, episode counter)) and the counter increments on every reset,
  so a second reset() on one object silently builds a DIFFERENT world and the
  session's `episode_seeds` would be a fiction. We rebuild crafter.Env(seed=...)
  per episode instead, which costs 91 ms on top of a ~2 s worldgen reset
  (measured at size 384) and makes the seed alone determine the episode.
- `length` defaults to a 10000-step cap. A scanner block is bounded by its own
  duration, so configs pass `length: 0` (no cap) and every `done` is a real
  death. If a config does set a cap, the step-limit case is reported as
  `truncated` and only death as `terminated`.

Seed + actions do NOT replay crafter exactly, and that is an engine bug rather
than a rig one. Every tenth step `Env.step` rebalances creatures per chunk, and
`World.chunks` holds each chunk's objects in a Python *set*; set order follows
object id(), so the despawn pick `creatures[random.randint(...)]` lands on a
different animal between two runs of the same seed and actions. Measured
2026-09-15: terrain is bit-identical (worldgen is seeded properly) while
creatures, and with them the rendered frame, diverge from step 10 onwards.
Sorting that list by position makes 300 random-action steps bit-identical, so
the one-line fix belongs in a crafter fork the model harness uses too. Until
then `log_frames` is the honest answer: the stored pixels, not a replay, are
the record of what the subject saw. Cost, measured on a real 300 s block at
5 fps (1506 frames, size 384): 6.5 KB and ~5 ms for a median daylit frame, but
crafter mixes per-pixel noise into the view at night, so night frames run to
210 KB and the block totals 30 MB of frames inside a 27 MB npz.

Rendering is not side-effect free at night, which is a second trap. That noise
is drawn from `world.random`, the same stream the creatures use, so any render
outside the step loop shifts every later draw. `render` therefore hands back the
frame `step` already produced rather than calling the engine again, and
`restore` puts the RNG back after the one render it cannot avoid. Measured:
restoring a night anchor and stepping to the end of the episode reproduces the
log exactly, and does not if one extra render is inserted first.

Crafter has no savestate API, but the whole Env pickles to ~145 KB in ~1.3 ms,
so `capture` returns that blob and `restore` loads it. A restored copy renders
the same frame bit-for-bit by day; at night only the state is preserved, since
the noise is a fresh draw. Storing one per frame would cost ~220 MB a block,
hence `state_stride` in the config (50 = one anchor every 10 s at 5 fps, counted
per episode). An anchor is an exact snapshot -- all 36 in the measured block
restored to a bit-identical state -- and so a sound branch point for analysis or
a model rollout. It is not a way to recompute the frames after it: unpickling
gives the objects new id()s, so 21 of those 36 continuations left the log at the
next creature rebalance, 9 to 109 steps on. The same one-line sort fixes that
case too (verified 2026-09-15).

`info` carries the whole symbolic state -- 16 inventory counters (health, food,
drink, energy, then materials and tools), 22 achievement counters, the player's
world position, and a 64x64 grid of material/object ids -- so `capture` logs all
four and `block_extra` ships the tables that decode them.

Crafter ships no sound at all (56 assets, every one a PNG), so `sound` is left
at the base class's None and nothing is played for these blocks.
"""

from __future__ import annotations

import pickle
import zlib
from typing import Any

import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# Crafter's Discrete(17): 0=noop, 1-4 move left/right/up/down, 5=do, 6=sleep,
# 7-10 place stone/table/furnace/plant, 11-13 make wood/stone/iron pickaxe,
# 14-16 make wood/stone/iron sword. Arrows + SPACE + S are the six keys the
# scanner paradigm already teaches; R/T/F/P and 1-6 are the letters crafter's
# own run_gui.py uses for the other ten, so every action is reachable (noop =
# no key held). Without them place and make are dead keys and the tech tree
# above "collect wood" cannot be played at all.
# Source: https://github.com/danijar/crafter/blob/master/crafter/data.yaml
_DEFAULT_KEYMAP = {
    "LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4, "SPACE": 5, "S": 6,
    "R": 7, "T": 8, "F": 9, "P": 10,
    "1": 11, "2": 12, "3": 13, "4": 14, "5": 15, "6": 16,
}


def _semantic_names(env: Any) -> list[str] | None:
    """Names for the ids in crafter's ``info["semantic"]`` grid, in id order.

    Read off the env's own two tables rather than hardcoded: crafter appends a
    fresh id for any material it is asked to place that was not in its initial
    list, so the mapping is only fully known once the block has been played.

    :param env: the crafter env that produced the semantic grids.
    :return: names indexed by id, or ``None`` if this crafter has no semantic
        view (then the grid is logged without a legend).
    """
    view = getattr(env, "_sem_view", None)
    if view is None:
        return None
    names = {i: mat or "void" for mat, i in view._mat_ids.items()}
    names.update({i: cls.__name__.lower() for cls, i in view._obj_ids.items()})
    return [names.get(i, "?") for i in range(max(names) + 1)]


class CrafterAdapter(EnvAdapter):
    name: str = "crafter"

    def _make(self, spec: dict) -> Any:
        # Built here so `env` is valid before the first reset; reset() then
        # rebuilds it per episode from that episode's seed.
        self._kwargs = dict(spec.get("env_kwargs", {}))
        self._log_frames = bool(spec.get("log_frames", False))
        self._last_obs = None
        return self._build(spec.get("seed"))

    def _build(self, seed: int | None) -> Any:
        """Construct a crafter env whose first episode is fixed by ``seed``."""
        import crafter
        return crafter.Env(seed=seed, **self._kwargs)

    def _keyspec(self) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _DEFAULT_KEYMAP.items()}
        return SingleKeySpec(combos=combos, noop=0)

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        """Start an episode on a world determined by ``seed`` alone.

        :param seed: episode seed; ``None`` reuses the env as built, and then
            crafter's episode counter rather than the log decides the world.
        :return: ``(obs, info)``; crafter's reset returns no info, so ``{}``.
        """
        if seed is not None:
            self.env = self._build(seed)
        self._last_obs = np.asarray(self.env.reset())
        return self._last_obs, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, done, info = self.env.step(int(action))
        self._last_obs = np.asarray(obs)
        # crafter's `done` is death OR its own step cap; only death is terminal,
        # and health rides along in `info`, so the two are separable here.
        alive = info["inventory"]["health"] > 0
        return self._last_obs, reward, done and not alive, done and alive, info

    def render(self) -> np.ndarray:
        # obs IS the RGB frame, and re-rendering it would consume the world RNG
        # after dark (see the module docstring), so hand back what step made.
        return self._last_obs

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        """Log this frame's symbolic state, and its pixels if asked to.

        :param obs: the frame crafter returned from ``step``.
        :param info: crafter's info dict for the same step.
        :param want_blob: whether this frame is a savestate anchor.
        :return: the frame's :class:`FrameState`.
        """
        variables = {
            "inventory": list(info["inventory"].values()),
            "achievements": list(info["achievements"].values()),
            "player_pos": np.asarray(info["player_pos"]),
            # SemanticView hands out a fresh array today; copy anyway, because a
            # view onto the live map would be overwritten in place next step.
            "semantic": np.asarray(info["semantic"]).copy(),
        }
        if self._log_frames:
            # Kept as uint8 rather than bytes: a list of bytes becomes a numpy
            # "S" array, which strips the trailing NULs a zlib stream can end
            # with. Decode with zlib.decompress -> frombuffer -> frame_shape.
            packed = zlib.compress(self._last_obs.tobytes(), 6)
            variables["frame_zlib"] = np.frombuffer(packed, np.uint8)
        blob = pickle.dumps(self.env, protocol=5) if want_blob else None
        return FrameState(blob=blob, variables=variables)

    def restore(self, blob: bytes) -> None:
        """Replace the env with the pickled one in ``blob``.

        :param blob: bytes previously returned as :attr:`FrameState.blob`.
        """
        self.env = pickle.loads(blob)
        world = getattr(self.env, "_world", None)
        if world is None:
            self._last_obs = np.asarray(self.env.render())
            return
        # Rewind the stream this render just drew night noise from, so the
        # continuation sees the draws the recorded run saw.
        state = world.random.get_state()
        self._last_obs = np.asarray(self.env.render())
        world.random.set_state(state)

    def block_extra(self) -> dict:
        """Block-level legends for the per-frame variables.

        :return: id-indexed name arrays for the action space, the inventory
            slots, the achievements and (when available) the semantic grid.
        """
        import crafter
        # constants.items / .achievements are what the player's dicts are built
        # from, so their order is the order `capture` logs the values in
        # (checked against a live env, 2026-09-15).
        extra = {
            "action_names": np.array(self.env.action_names),
            "inventory_names": np.array(list(crafter.constants.items)),
            "achievement_names": np.array(list(crafter.constants.achievements)),
        }
        names = _semantic_names(self.env)
        if names:
            extra["semantic_names"] = np.array(names)
        if self._log_frames:
            extra["frame_shape"] = np.array(self._last_obs.shape)
        return extra
