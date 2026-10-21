"""Every backend logs its game's symbolic state every step, so it is read from the log, not a replay.

Each step: every observation key (pictures aside) and ``info`` key is logged, but
for :data:`NOT_LOGGED`, each with its reason; and each backend's
:data:`ENGINE_FIELDS` (and :data:`ONCE_FIELDS`), the state its env does not hand
over by itself. A backend missing from that table fails.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import pytest

from fmri_gym.adapters import get_adapter

STEPS = 12

#: ``info`` fields the run logs on each ``frame`` record beside the variables (fmri_gym/run.py).
FRAME_FIELDS = {"reward", "env_action"}

#: ``(backend, info or obs key)`` left out of the log on purpose, and why.
NOT_LOGGED = {
    ("rushhour", "info['t_ms']"): "the engine's wall clock at the press; the run's clock columns "
                                  "hold the time, and a replay could never match it",
    ("rushhour", "info['trial_ms']"): "the same wall clock, from the trial's start",
    ("crafter", "info['discount']"): "0 on the step that kills, else 1: terminated says it",
    ("nethack", "obs['tty_chars']"): "the terminal the frame is drawn from; glyphs, message "
                                     "and blstats carry what it shows",
    ("nethack", "obs['tty_colors']"): "the terminal's colors, likewise",
    ("stk", "obs"): "a scaled encoding of logged fields (stk_gym.obs.encode)",
    ("stk", "info['is_success']"): "the same as the logged finished",
    ("stk", "info['lap']"): "the same as the logged finished_laps",
    ("stk", "info['race_time']"): "the same as the logged time",
}

#: The variables that carry each backend's state beyond its env's observation, every frame.
ENGINE_FIELDS = {
    "gym": {"obs"},
    "ale": {"ram"},
    "retro": {"ram"},
    "crafter": {"semantic", "inventory", "player_pos", "daylight", "facing", "objects"},
    "minihack": {"glyphs", "blstats", "message", "inv_strs", "inv_glyphs", "specials"},
    "nethack": {"glyphs", "blstats", "message", "inv_strs", "misc"},
    # objects/labels/player_variables/notifications cost real per-frame engine work
    # nothing on the live path needs, so capture() does not log them -- a replay pass
    # wanting a symbolic transcript reads them from the engine itself instead.
    "vizdoom": set(),
    "coom": {"game_variables", "tic"},
    "vgdl": {"symbolic_state", "events"},
    "baba": {"grid", "ruleset", "agent_pos"},
    "baba_auto": {"state", "play_state"},
    "rushhour": {"cars", "selected"},
    "aigamestore": {"state_gamePhase", "state_score"},
    "arc3": {"grid"},
    "stk": {"karts", "lookahead", "total_laps", "speed", "distance_down_track"},
}

#: Variables logged once an episode, on its first frame: what does not change within one.
ONCE_FIELDS = {
    "coom": {"sectors"},
}


def test_capture_logs_everything(game_phase):
    backend, path, phase = game_phase
    assert backend in ENGINE_FIELDS, (f"{backend}: say in ENGINE_FIELDS which logged variables "
                                      "carry its state")
    try:
        adapter = get_adapter(backend, dict(phase))
    except (ModuleNotFoundError, gymnasium.error.DependencyNotInstalled) as e:
        if backend != "gym":
            raise
        pytest.skip(f"{phase['game']} needs an extra not synced here: {e}")
    seed = int(phase.get("seed", 1000))
    missing: dict[str, str] = {}
    seen: set[str] = set()
    try:
        # The run logs what a step returns; a reset's observation is not a frame.
        adapter.reset(seed)
        pool = [adapter.keymap.resolve(combo) for combo in adapter.keymap.combos]
        for step in range(STEPS):
            obs, _, terminated, truncated, info = adapter.step(pool[step % len(pool)])
            variables = adapter.capture(obs, info, want_blob=False).variables
            missing.update(_missing(backend, obs, info, variables))
            missing.update({k: "(engine field, every frame)"
                            for k in ENGINE_FIELDS[backend] - set(variables)})
            seen |= set(variables)
            if terminated or truncated:
                seed += 1
                adapter.reset(seed)
    finally:
        adapter.close()
    missing.update({k: "(engine field, once an episode)"
                    for k in ONCE_FIELDS.get(backend, set()) - seen})
    assert not missing, (f"{backend} ({path}) does not log: "
                         + "; ".join(f"{k} {v}" for k, v in sorted(missing.items())))


def _missing(backend: str, obs: Any, info: Any, variables: dict) -> dict[str, str]:
    """What of ``obs`` and ``info`` is not in ``variables``, by name, with its shape."""
    out = {}
    if isinstance(obs, dict):
        for key, value in obs.items():
            if not _is_picture(value) and not _logged(key, value, variables):
                out[f"obs[{key!r}]"] = _describe(value)
    elif obs is not None and not _is_picture(obs) \
            and not any(_same(obs, v) for v in variables.values()):
        out["obs"] = _describe(obs)
    for key, value in (info or {}).items():
        if key not in FRAME_FIELDS and not _logged(key, value, variables):
            out[f"info[{key!r}]"] = _describe(value)
    return {k: v for k, v in out.items() if (backend, k) not in NOT_LOGGED}


def _logged(key: str, value: Any, variables: dict) -> bool:
    """Whether ``value`` is in ``variables`` under its name, a prefixed or split name, or any."""
    if key in variables or f"info_{key}" in variables:
        return True
    if isinstance(value, dict) and value and all(f"{key}_{k}" in variables for k in value):
        return True
    if value is None or isinstance(value, (bool, int, float, np.generic)):
        # A number matches too much by value alone (any 0 is every other 0).
        return False
    return any(_same(value, v) for v in variables.values())


def _same(a: Any, b: Any) -> bool:
    """Equal values: Python's ``==`` where it gives a verdict, element by element for arrays."""
    try:
        verdict = a == b
        if isinstance(verdict, bool):
            return verdict
    except (ValueError, TypeError):
        pass
    try:
        return np.asarray(a).shape == np.asarray(b).shape and bool(np.array_equal(a, b))
    except (ValueError, TypeError):
        return False


def _is_picture(value: Any) -> bool:
    """An RGB(A) image: the frame, which frames.h5 already holds."""
    a = np.asarray(value) if isinstance(value, np.ndarray) else None
    return a is not None and a.ndim == 3 and a.shape[-1] in (3, 4) and a.dtype == np.uint8


def _describe(value: Any) -> str:
    """``(24, 80) uint8`` for an array, else the type and a short repr."""
    if isinstance(value, np.ndarray):
        return f"{value.shape} {value.dtype}"
    text = repr(value)
    return f"{type(value).__name__} {text[:40]}{'...' if len(text) > 40 else ''}"
