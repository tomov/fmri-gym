"""Which backends a set of changed files touches, so the local suite plays those and not the rest.

A change to an adapter, to the gym package under it, to one of its configs or
to its checkout touches that backend *directly*: the suite must be able to
play it, and refuses to pretend otherwise when it is not installed. A change
to the core (``humans/*.py`` outside the adapters, the run loop's entry
points, the recorder, the dependencies, the tests themselves) touches every
backend; the ones this machine cannot run are then skipped, visibly.

Command line, for the pre-push habit and the CI step that mirrors it::

    python -m tests.changed              # against origin/main
    python -m tests.changed main
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from humans.config.curriculum import is_common_file
from tests.backends import BACKENDS, REPO, game_phases, phase_backend

#: Paths (repo-relative prefixes) whose change concerns every backend.
CORE_PREFIXES = ("humans/", "fmri_play.py", "agents/", "pyproject.toml", "uv.lock",
                 "tests/", ".github/")


@dataclass
class Selection:
    """The backends to test, and which of them the change hit directly.

    :ivar direct: backends whose own files changed; they must run.
    :ivar core: whether a shared file changed, which selects every backend.
    """

    direct: set[str] = field(default_factory=set)
    core: bool = False

    @property
    def backends(self) -> list[str]:
        """Every backend to test, in the table's order."""
        if self.core:
            return list(BACKENDS)
        return [b for b in BACKENDS if b in self.direct]


def changed_files(base: str) -> list[str]:
    """Repo-relative paths that differ from ``base``: committed, staged, unstaged and untracked.

    :param base: the ref the branch forked from (its merge base with HEAD is used).
    :raises RuntimeError: when git cannot resolve ``base``.
    """
    def git(*args: str) -> list[str]:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
        return [line for line in out.stdout.splitlines() if line]

    files = set(git("diff", "--name-only", f"{base}...HEAD"))
    files |= set(git("diff", "--name-only", "HEAD"))
    files |= set(git("ls-files", "--others", "--exclude-standard"))
    return sorted(files)


def select(paths: list[str]) -> Selection:
    """The backends ``paths`` touch (see the module docstring for the rules)."""
    selection = Selection()
    for path in paths:
        if path.startswith("humans/adapters/"):
            _adapter(Path(path).stem, selection)
        elif path.startswith(CORE_PREFIXES) and path != "tests/README.md":
            selection.core = True
        elif path.startswith("configs/") and path.endswith(".json"):
            selection.direct |= _config_backends(REPO / path)
        elif path.startswith("gym/"):
            gym_dir = path.split("/")[1]
            selection.direct |= {b.name for b in BACKENDS.values() if gym_dir in b.gym_dirs}
        elif path.startswith("external/"):
            name = path.split("/")[1]
            selection.direct |= {b.name for b in BACKENDS.values() if b.external == name}
    return selection


def _adapter(stem: str, selection: Selection) -> None:
    """An adapter module is its backend's; the shared ones are the core's."""
    if stem in ("base", "keymap", "__init__"):
        selection.core = True
    elif stem == "default":
        selection.direct.add("gym")
    elif stem in BACKENDS:
        selection.direct.add(stem)
    else:
        # A new adapter without a row in the table: the suite cannot play it,
        # and test_configs says so. Selecting the core runs that test.
        selection.core = True


def _config_backends(path: Path) -> set[str]:
    """The backends a config's game phases name, or a shared phase's; none for a file that
    is gone or broken."""
    if not path.exists():
        return set()
    try:
        if is_common_file(str(path)):
            with open(path) as f:
                return {phase_backend(json.load(f))}
        return {phase_backend(g) for g in game_phases(path)}
    except (ValueError, AttributeError):
        return set()     # test_configs reports the broken file


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    selection = select(changed_files(base))
    if selection.core:
        print("core changed: every backend")
    for name in selection.backends:
        print(f"{name}{'  (direct)' if name in selection.direct else ''}")
    if not selection.backends:
        print("no backend touched")


if __name__ == "__main__":
    main()
