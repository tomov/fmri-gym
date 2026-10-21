"""A COOM env's sound on a Mac without OpenAL Soft: it stops before the engine starts."""

from __future__ import annotations

import pytest


@pytest.fixture
def coom_env(monkeypatch):
    """``coom_gym.env`` on a macOS where OpenAL Soft is not found."""
    from coom_gym import env as coom_env

    monkeypatch.setattr(coom_env.sys, "platform", "darwin")
    monkeypatch.setattr(coom_env, "use_openal_soft", lambda: None)
    return coom_env


def test_sound_on_raises_with_the_fix(coom_env):
    with pytest.raises(RuntimeError, match="brew install openal-soft"):
        coom_env.COOMEnv("pitfall", audio_buffer_enabled=True)


def test_sound_off_plays_without_it(coom_env):
    env = coom_env.COOMEnv("pitfall")
    env.reset(seed=1)
    env.close()
