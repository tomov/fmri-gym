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
