"""The suite's command-line options, declared where pytest reads them in time.

pytest scans the command line for paths before any ``conftest.py`` under
``tests/`` is loaded, and an option it does not know yet looks like a path
with a value: ``--backend gym`` names ``gym/``, a real directory, so pytest
anchors its start-up on that directory and finds the option's definition
only once collection reaches ``tests/``, too late to accept it. A ``conftest``
at the root is loaded before the arguments are parsed whatever the anchor,
so the options live here; everything they drive is in ``tests/conftest.py``.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("fmri-gym", "which game backends the suite plays")
    g.addoption("--backend", action="append", default=[], metavar="NAME",
                help="play these backends (repeat or comma-separate; 'all', 'none'); "
                     "a named backend this machine cannot run fails")
    g.addoption("--changed", nargs="?", const="origin/main", default=None, metavar="BASE",
                help="play the backends the files changed since BASE touch "
                     "(default base: origin/main); a core change selects every backend")
    g.addoption("--strict-backends", action="store_true",
                help="a selected backend this machine cannot run fails instead of skipping")
    g.addoption("--thorough", action="store_true",
                help="every config of a selected backend, and the long sweeps")
    g.addoption("--no-audio", action="store_true",
                help="headless runs leave the sound device alone (CI, or no output here)")
    g.addoption("--frames", type=int, default=120,
                help="frames per episode the recordings play (default 120)")
    g.addoption("--run-seconds", type=float, default=15.0,
                help="length of each headless run; the game plays 10 s less (default 15; "
                     "CI: 25)")
    g.addoption("--window", action="store_true",
                help="headless runs open a real window (one at a time) instead of SDL's "
                     "dummy driver, to catch what only a display shows")
