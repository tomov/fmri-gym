"""``python -m arc3_gym GAME [GAME ...]``: download games into the default games folder."""

from __future__ import annotations

import sys

from .env import fetch

if len(sys.argv) < 2:
    sys.exit("usage: python -m arc3_gym GAME [GAME ...]   (for example: ls20 tr87)")
fetch(sys.argv[1:])
