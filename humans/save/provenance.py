"""What wrote the data: the code that ran, and the machine it ran on."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from importlib.metadata import version
from typing import Any

#: The checkout this package is in, when it runs from one.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def versions() -> dict[str, str]:
    """fmri-gym, its git checkout (``-dirty``: edited), and the libraries timing depends on."""
    import numpy
    import pygame
    try:
        git = subprocess.run(["git", "-C", _REPO, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        git = ""
    return {"fmri_gym": version("fmri-gym"), "git": git or "n/a",
            "python": platform.python_version(), "numpy": numpy.__version__,
            "pygame": pygame.version.ver, "sdl": ".".join(map(str, pygame.get_sdl_version()))}


def platform_info() -> dict[str, Any]:
    """The machine: OS, architecture, host name, and the clock behind ``perf_counter``."""
    clock = time.get_clock_info("perf_counter")
    return {"system": platform.platform(), "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "perf_counter": {"implementation": clock.implementation,
                             "resolution": clock.resolution, "monotonic": clock.monotonic}}
