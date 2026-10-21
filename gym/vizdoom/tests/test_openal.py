"""``use_openal_soft``: OpenAL Soft on dyld's fallback path, on macOS."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def openal(monkeypatch, tmp_path):
    """The module, on a macOS with no OpenAL anywhere but the directories a test makes."""
    from vizdoom_gym import openal

    monkeypatch.setattr(openal.sys, "platform", "darwin")
    monkeypatch.setattr(openal, "_DYLD_DEFAULTS", (str(tmp_path / "default"),))
    monkeypatch.setattr(openal, "_OPENAL_DIRS", (str(tmp_path / "brew"),))
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("HOMEBREW_PREFIX", raising=False)
    return openal


def _lib(d):
    d.mkdir(parents=True)
    (d / "libopenal.1.dylib").touch()
    return str(d)


def test_keg_goes_first_and_dyld_defaults_stay(openal, tmp_path):
    brew = _lib(tmp_path / "brew")
    assert openal.use_openal_soft() == brew
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == f"{brew}:{tmp_path / 'default'}"


def test_homebrew_prefix_is_looked_in(openal, tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    monkeypatch.setenv("HOMEBREW_PREFIX", str(prefix))
    lib = _lib(prefix / "opt/openal-soft/lib")
    assert openal.use_openal_soft() == lib


def test_found_on_the_path_already_changes_nothing(openal, tmp_path, monkeypatch):
    own = _lib(tmp_path / "own")
    _lib(tmp_path / "brew")
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", own)
    assert openal.use_openal_soft() == own
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == own


def test_none_installed(openal):
    assert openal.use_openal_soft() is None
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ


def test_not_macos_is_left_alone(openal, tmp_path, monkeypatch):
    _lib(tmp_path / "brew")
    monkeypatch.setattr(openal.sys, "platform", "linux")
    assert openal.use_openal_soft() is None
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ
