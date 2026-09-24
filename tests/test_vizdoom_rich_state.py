"""Tests for VizDoomAdapter.get_rich_state() -- the optional, backend-specific
counterpart to capture() that queries every piece of scene state ViZDoom's
backend can report (see VizDoomAdapter.get_rich_state's own docstring).
Skipped if vizdoom isn't installed.
"""

from __future__ import annotations

import json

import pytest

try:
    import vizdoom  # noqa: F401
    HAVE_VIZDOOM = True
except ImportError:
    HAVE_VIZDOOM = False

pytestmark = pytest.mark.skipif(not HAVE_VIZDOOM, reason="vizdoom not installed")


def _adapter(**env_kwargs):
    from fmri_gym.adapters.vizdoom import VizDoomAdapter
    return VizDoomAdapter({
        "backend": "vizdoom", "game": "VizdoomDefendCenter-v1",
        "env_kwargs": env_kwargs,
    })


def test_rich_state_without_extra_buffers_still_gives_every_game_variable():
    """objects/labels/sectors need the scenario's own env_kwargs to opt in
    (they're off by default in ViZDoom itself) -- game_variables doesn't:
    get_rich_state() queries every vzd.GameVariable directly
    (game.get_game_variable), not just Defend the Center's own
    available_game_variables ({AMMO2, HEALTH}), so it's always the full
    set regardless of scenario config."""
    import vizdoom as vzd

    adapter = _adapter()
    try:
        obs, info = adapter.reset(seed=1)
        obs, reward, terminated, truncated, info = adapter.step(0)
        rich = adapter.rich_state(obs, info)
        all_names = {n for n in dir(vzd.GameVariable) if not n.startswith("_") and n not in ("name", "value")}
        assert set(rich["game_variables"]) == all_names
        assert {"AMMO2", "HEALTH"} <= set(rich["game_variables"])
        assert rich["game_variables"]["HEALTH"] == 100.0
        assert rich["objects"] == []
        assert rich["labels"] == []
        assert rich["sectors"] == []
        assert rich["episode"]["doom_map"] == "map01"
        assert rich["episode"]["is_player_dead"] is False
        json.dumps(rich)
    finally:
        adapter.close()


def test_rich_state_with_buffers_enabled_exposes_objects_labels_sectors():
    adapter = _adapter(objects_info_enabled=True, labels_buffer_enabled=True,
                        sectors_info_enabled=True)
    try:
        obs, info = adapter.reset(seed=1)
        for _ in range(3):
            obs, reward, terminated, truncated, info = adapter.step(0)
        rich = adapter.rich_state(obs, info)

        assert rich["objects"], "expected at least the player object"
        obj = rich["objects"][0]
        assert set(obj) == {"id", "name", "position", "angle", "pitch", "roll", "velocity"}
        assert len(obj["position"]) == 3

        assert rich["labels"], "expected at least the player's own on-screen label"
        label = rich["labels"][0]
        assert set(label) == {"object_id", "object_name", "category", "value", "bbox", "position"}
        assert len(label["bbox"]) == 4

        assert rich["sectors"]
        assert set(rich["sectors"][0]) == {"floor_height", "ceiling_height"}

        # Every field must already be JSON-safe (no numpy/SWIG scalars leaking through).
        json.dumps(rich)
    finally:
        adapter.close()
