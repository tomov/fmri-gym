"""ARC-AGI-3 adapter -- turn-based grid puzzles, via ``arc3-gym``.

ARC-AGI-3 (arcprize.org) is a set of interactive games on a 64x64 grid of 16
colours, each in several levels, ending in ``WIN`` or ``GAME_OVER``. AI agents
are scored on these same games, so people in the scanner and models meet one
rule set. ``arc3_gym`` (``gym/arc3/``) is the Gymnasium env: the ``arc-agi``
toolkit is not one, and that package puts the contract in front of it, running
the toolkit offline from game files fetched once (README "ARC-AGI-3 games").

A phase's ``game`` is a four-letter game id (``ls20``). A phase's ``keys`` index
the game's simple actions in ascending order: the game's ``ACTION1..ACTION5``
that it offers, then ``ACTION7`` (undo) if it does. ACTION1-4 are up, down, left
and right; ACTION5 is a game-specific interact action. The games that need a
click (``ACTION6``) are not supported: the scanner has no pointer, and the env
refuses them at start-up. Use ``turn_based: true``: a move is a key press. No
savestate -> seed + action replay.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .base import EnvAdapter, FrameState


class Arc3Adapter(EnvAdapter):
    name: str = "arc3"

    def _make(self, spec: dict) -> gym.Env:
        import arc3_gym  # noqa: F401  (registers Arc3-v0)

        kwargs = {k: spec[k] for k in ("environments_dir", "cell_px") if k in spec}
        return gym.make("Arc3-v0", render_mode="rgb_array", game=spec["game"], **kwargs)

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        state = self.last_info.get("state")
        if terminated and state == "WIN":
            return "won", "Game won"
        if terminated and state == "GAME_OVER":
            return "lost", "Game over"
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        variables = dict(info)
        variables["grid"] = obs.copy()
        return FrameState(blob=None, variables=variables)
