"""OpenAL Soft on dyld's library path, for ViZDoom's sound on macOS.

ViZDoom's engine makes its sound, the audio buffer included, with OpenAL Soft,
which it loads by name (``libopenal.1.dylib``) when ``game.init()`` starts it.
The macOS wheels do not bundle it, and Homebrew's ``openal-soft`` is keg-only,
outside dyld's search path, so the engine falls back to Apple's OpenAL, which
cannot render into the buffer: every ``obs["audio"]`` is zeros, and ViZDoom
raises nothing (https://github.com/Farama-Foundation/ViZDoom/issues/757).
"""

from __future__ import annotations

import os
import sys

_OPENAL_LIB = "libopenal.1.dylib"
_OPENAL_DIRS = ("/opt/homebrew/opt/openal-soft/lib",   # Homebrew, Apple silicon
                "/usr/local/opt/openal-soft/lib",      # Homebrew, Intel
                "/opt/local/lib")                      # MacPorts
_DYLD_DEFAULTS = (os.path.expanduser("~/lib"), "/usr/local/lib", "/usr/lib")


def use_openal_soft() -> str | None:
    """Put OpenAL Soft's directory first on ``DYLD_FALLBACK_LIBRARY_PATH``, on macOS.

    Setting the variable replaces dyld's default directories, so they are kept
    after it. A process started afterwards, such as the engine, inherits it.

    :return: the directory holding ``libopenal.1.dylib``; None off macOS, or
        when none is installed.
    """
    if sys.platform != "darwin":
        return None
    path = [d for d in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if d]
    path = path or list(_DYLD_DEFAULTS)
    prefix = os.environ.get("HOMEBREW_PREFIX")
    for d in path + ([f"{prefix}/opt/openal-soft/lib"] if prefix else []) + list(_OPENAL_DIRS):
        if os.path.exists(os.path.join(d, _OPENAL_LIB)):
            if d not in path:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([d, *path])
            return d
    return None
