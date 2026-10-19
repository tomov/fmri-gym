"""The backends the suite knows: what installs each, how to tell it is here, which configs stand for it.

One row per ``get_adapter`` branch (``test_configs`` fails when the two drift).
A row says what the suite needs in order to run the backend on this machine --
the pyproject extra, the import that proves it was synced, a checkout under
``external/``, a browser, downloaded game files -- and :meth:`Backend.missing`
turns the first unmet need into one line that names the fix. It also says
which curricula stand for the backend when the suite is run lightly (the
*smoke* configs: one game phase each goes through the whole chain of checks),
and why CI cannot run it, if it cannot.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from fmri_gym.curriculum import is_common_file, load_curriculum

REPO = Path(__file__).resolve().parents[1]
CURRICULA = REPO / "configs" / "curricula"


@dataclass(frozen=True)
class Backend:
    """One game engine as the suite sees it.

    :ivar name: the ``backend`` a game phase names.
    :ivar extra: the pyproject extra that installs it; ``None`` for the core.
    :ivar module: importable once the extra is synced.
    :ivar smoke: the curricula that stand for it (repo-relative paths).
    :ivar gym_dirs: the ``gym/<dir>`` packages it runs on, so a change there selects it.
    :ivar external: the ``external/<name>`` checkout it reads, if any.
    :ivar marker: a file the checkout must hold, relative to it.
    :ivar ci: why CI cannot run it; ``None`` when it can.
    """

    name: str
    extra: str | None
    module: str
    smoke: tuple[str, ...]
    gym_dirs: tuple[str, ...] = ()
    external: str | None = None
    marker: str = ""
    ci: str | None = None

    def installed(self) -> bool:
        """Whether the backend's package imports here."""
        return importlib.util.find_spec(self.module) is not None

    def missing(self) -> str | None:
        """Why this machine cannot run the backend, with the fix; ``None`` when it can."""
        if not self.installed():
            extra = f"uv sync --extra {self.extra}" if self.extra else "uv sync"
            return f"{self.module} is not installed: {extra} (and every other extra you need)"
        if self.external and not (REPO / "external" / self.external / self.marker).exists():
            return (f"external/{self.external} is not checked out: "
                    "git submodule update --init")
        return _RUNTIME_NEEDS.get(self.name, lambda: None)()


def _chrome() -> str | None:
    """aigamestore drives the system Chrome through Playwright."""
    for name in ("google-chrome", "google-chrome-stable", "chrome", "chromium"):
        if shutil.which(name):
            return None
    if Path("/Applications/Google Chrome.app").exists():
        return None
    return "no Chrome for Playwright: install Google Chrome, or uv run playwright install chrome"


def _arc3_files() -> str | None:
    """arc3 plays game files fetched once into external/arc3."""
    games = REPO / "external" / "arc3"
    if games.exists() and any(games.iterdir()):
        return None
    return ("external/arc3 holds no game files: uv run python -m arc3_gym ls20 "
            "(once per game id, with the network on)")


def _stk() -> str | None:
    """stk needs the Linux x86_64 game pack and a real GL display."""
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        return None
    return (f"supertuxkart-gym has no game pack for {platform.system()} {platform.machine()}; "
            "STK_ENV_BIN names a binary of your own")


_RUNTIME_NEEDS = {"aigamestore": _chrome, "arc3": _arc3_files, "stk": _stk}

_C = "configs/curricula/"
BACKENDS: dict[str, Backend] = {b.name: b for b in (
    Backend("gym", None, "gymnasium", (_C + "gym/classic_control.json",)),
    Backend("ale", "ale", "ale_py", (_C + "ale/all.json",)),
    Backend("retro", "retro", "retro", (_C + "retro/airstriker.json",),
            ci="game ROMs have to be imported by hand (README 'Running stable-retro games')"),
    Backend("crafter", "crafter", "crafter_gym",
            (_C + "crafter/L1.json", _C + "crafter/L4.json"),
            gym_dirs=("crafter",)),
    Backend("minihack", "minihack", "minihack",
            (_C + "minihack/room5x5.json", _C + "minihack/river_reposition.json"),
            gym_dirs=("minihack", "replay_state")),
    Backend("nethack", "nethack", "nle", (_C + "nethack/score.json",),
            gym_dirs=("minihack", "replay_state")),
    Backend("vizdoom", "vizdoom", "vizdoom_gym", (_C + "vizdoom/basic.json",),
            gym_dirs=("vizdoom", "replay_state")),
    Backend("coom", "coom", "coom_gym", (_C + "coom/pitfall.json",),
            gym_dirs=("coom", "replay_state"), external="coom",
            marker="COOM/env/scenarios/pitfall/conf.cfg"),
    Backend("vgdl", "vgdl", "vgdl_gym", (_C + "vgdl/aliens.json",),
            gym_dirs=("vgdl",), external="vgdl", marker="games/aliens_v0/aliens.txt"),
    Backend("baba", "baba", "baba_gym", (_C + "baba/make_win.json",), gym_dirs=("baba",)),
    Backend("baba_auto", "baba_auto", "baba_auto_gym", (_C + "baba_auto/baba_is_you.json",),
            gym_dirs=("baba_auto", "replay_state"), external="baba_auto",
            marker="CMakeLists.txt"),
    Backend("rushhour", "rushhour", "rushhour_gym", (_C + "rushhour/easy.json",),
            gym_dirs=("replay_state",)),
    Backend("aigamestore", "aigamestore", "aigamestore_gym", (_C + "aigamestore/game1.json",),
            gym_dirs=("aigamestore", "replay_state")),
    Backend("arc3", "arc3", "arc3_gym", (_C + "arc3/ls20.json",), gym_dirs=("arc3",)),
    Backend("stk", "stk", "stk_gym", (_C + "stk/hacienda.json",), gym_dirs=("replay_state",),
            ci="the game pack is Linux x86_64 only and renders through a real GL display"),
)}

#: The backends whose headless runs stand for the run loop when only core files changed:
#: a real-time phase (gym), a turn-based one (minihack), held-key combos with sound (vizdoom).
LOOP_SAMPLE = ("gym", "minihack", "vizdoom")

#: The backends a hosted Linux runner can play.
CI_BACKENDS = tuple(b.name for b in BACKENDS.values() if b.ci is None)


def game_phases(config_path: Path) -> list[dict]:
    """The game phases of a curriculum file, in order, includes resolved."""
    return [p for p in load_curriculum(str(config_path))["curriculum"] if p["type"] == "game"]


def phase_backend(phase: dict) -> str:
    """The backend a game phase names (``gym`` when it names none)."""
    return phase.get("backend", "gym")


def all_configs() -> list[Path]:
    """Every runnable curriculum file in the repo, sorted."""
    return sorted(p for p in CURRICULA.rglob("*.json") if not is_common_file(str(p)))


def configs_of(backend: str) -> list[Path]:
    """Every runnable config with a game phase on ``backend``."""
    return [p for p in all_configs()
            if any(phase_backend(g) == backend for g in game_phases(p))]
