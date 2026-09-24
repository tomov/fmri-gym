"""fmri-gym: one fMRI experiment framework for any Gymnasium-compatible game.

The experiment loop is engine-agnostic; each game engine is a small EnvAdapter
that wraps the underlying environment for one game block.
"""

from __future__ import annotations

import os

# pygame prints a banner on import, to stdout -- where fmri-ses
# answers a session script's SES= line. Set here, before anything imports
# pygame, so it holds for every entry point. The versions it shows go in the
# manifest instead.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from .adapters import get_adapter
from .run import Run, Clock
from .display import Display
from .audio import Audio
from .triggers import Triggers
from .logging import Logger

__all__ = ["Run", "Clock", "Display", "Audio", "Triggers", "Logger", "get_adapter"]
