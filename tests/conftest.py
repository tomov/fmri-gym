"""How the suite picks its backends, and what it does about one this machine cannot run.

The engine-free tests always run. The per-backend ones run for the backends
*selected*: every backend by default (CI), the ones a change touches with
``--changed`` (the pre-push habit, see :mod:`tests.changed`), or the ones named
with ``--backend``. A selected backend that is not installed here, or lacks a
checkout, a browser or its game files, is **skipped with the fix in the
reason** -- unless it was asked for by name, the change hit its own files, or
``--strict-backends`` is on (CI), in which case the test **fails**: a suite
that silently vouched for a backend it never played would be the worst thing
it could do (AGENTS.md, rule 4).

``--thorough`` plays every config of a selected backend through the quick
check and runs the long sweeps, where the default plays the backend's smoke
configs only. The headless runs of the real loop are the slow part: a core
change under ``--changed`` plays them for the backends it touches directly
and for :data:`~tests.backends.LOOP_SAMPLE`, not for every backend, unless
``--thorough``. ``--window`` runs them in a real window. ``--no-audio`` keeps the headless runs off the sound device, for
a machine without one. The options themselves are declared in the repo root's
``conftest.py``, which says why they cannot live here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.backends import BACKENDS, LOOP_SAMPLE, REPO, configs_of, game_phases, phase_backend
from tests.changed import changed_files, select


@dataclass
class Plan:
    """What this session plays: the selected backends and the policy for missing ones."""

    selected: list[str]
    direct: set[str]
    #: the selected backends whose smoke configs also get a headless run of the real loop
    headless: list[str]
    strict: bool
    thorough: bool
    no_audio: bool
    frames: int
    run_seconds: float
    has_window: bool
    why: str

    def gate(self, backend: str) -> None:
        """Skip or fail the current test when this machine cannot run ``backend``."""
        reason = BACKENDS[backend].missing()
        if reason is None:
            return
        if self.strict or backend in self.direct:
            pytest.fail(f"{backend} must run here but cannot: {reason}", pytrace=False)
        pytest.skip(f"{backend}: {reason}")


def _plan(config: pytest.Config) -> Plan:
    names = [n for arg in config.getoption("--backend") for n in arg.split(",") if n]
    base = config.getoption("--changed")
    if names and base is not None:
        raise pytest.UsageError("give --backend or --changed, not both")
    direct: set[str] = set()
    is_core = False
    if base is not None:
        selection = select(changed_files(base))
        selected, direct, is_core = selection.backends, selection.direct, selection.core
        why = f"files changed against {base}" + (" (a core change)" if selection.core else "")
    elif names == ["none"]:
        selected, why = [], "--backend none"
    elif names and names != ["all"]:
        unknown = sorted(set(names) - set(BACKENDS))
        if unknown:
            raise pytest.UsageError(f"--backend: unknown {unknown}; known: {sorted(BACKENDS)}")
        selected = [b for b in BACKENDS if b in names]
        direct, why = set(selected), "named on the command line"
    else:
        selected, why = list(BACKENDS), "every backend"
    thorough = config.getoption("--thorough")
    headless = selected
    if is_core and not thorough:
        headless = [b for b in selected if b in direct or b in LOOP_SAMPLE]
    return Plan(selected, direct, headless, config.getoption("--strict-backends"), thorough,
                config.getoption("--no-audio"), config.getoption("--frames"),
                config.getoption("--run-seconds"), config.getoption("--window"), why)


def pytest_configure(config: pytest.Config) -> None:
    config.fmri_plan = _plan(config)
    config.fmri_recordings = {}


def pytest_report_header(config: pytest.Config) -> list[str]:
    plan: Plan = config.fmri_plan
    missing = {b: BACKENDS[b].missing() for b in plan.selected}
    lines = [f"fmri-gym backends ({plan.why}): "
             + (", ".join(b for b in plan.selected if not missing[b]) or "none")]
    if plan.headless != plan.selected:
        lines.append("  headless runs: " + (", ".join(plan.headless) or "none")
                     + " (a core change; --thorough runs them all)")
    for b, reason in missing.items():
        if reason:
            verb = "FAIL" if plan.strict or b in plan.direct else "skip"
            lines.append(f"  {verb} {b}: {reason}")
    return lines


# --- parametrization over the selected backends -----------------------------

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    plan: Plan = metafunc.config.fmri_plan
    if "smoke_config" in metafunc.fixturenames:
        backends = plan.headless if metafunc.function.__name__ == "test_headless_run" \
            else plan.selected
        pairs = [(b, c) for b in backends for c in BACKENDS[b].smoke]
        metafunc.parametrize("smoke_config", pairs, indirect=True,
                             ids=[f"{b}/{Path(c).stem}" for b, c in pairs])
    if "game_phase" in metafunc.fixturenames:
        triples = _phases_to_play(plan)
        metafunc.parametrize("game_phase", triples, indirect=True,
                             ids=[f"{b}/{Path(c).stem}#{i}" for b, c, i in triples])


def _phases_to_play(plan: Plan) -> list[tuple[str, str, int]]:
    """``(backend, config, phase index)`` for the quick check: smoke firsts, or everything."""
    out = []
    for b in plan.selected:
        if plan.thorough:
            for path in configs_of(b):
                rel = str(path.relative_to(REPO))
                out += [(b, rel, i) for i, g in enumerate(game_phases(path))
                        if phase_backend(g) == b]
        else:
            out += [(b, c, 0) for c in BACKENDS[b].smoke]
    return out


@dataclass(frozen=True)
class Smoke:
    """One smoke config of one backend."""

    backend: str
    path: str


@pytest.fixture
def plan(request: pytest.FixtureRequest) -> Plan:
    return request.config.fmri_plan


@pytest.fixture
def smoke_config(request: pytest.FixtureRequest) -> Smoke:
    backend, path = request.param
    request.config.fmri_plan.gate(backend)
    return Smoke(backend, path)


@pytest.fixture
def game_phase(request: pytest.FixtureRequest) -> tuple[str, str, dict]:
    """``(backend, config path, the phase dict)`` for one game phase of one config."""
    backend, path, index = request.param
    request.config.fmri_plan.gate(backend)
    phase = game_phases(REPO / path)[index]
    assert phase_backend(phase) == backend
    return backend, path, phase


# --- tests that belong to one backend without being parametrized over it ------

def _owner_of(path: Path) -> str | None:
    """The backend whose gym package's own tests these are (``gym/<dir>/tests/...``)."""
    try:
        parts = path.relative_to(REPO).parts
    except ValueError:
        return None
    if len(parts) < 3 or parts[0] != "gym" or parts[2] != "tests":
        return None
    owners = [b.name for b in BACKENDS.values() if parts[1] in b.gym_dirs]
    return parts[1] if parts[1] in owners else (owners[0] if owners else None)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect the one-backend tests of backends not selected; the gate fixture does the rest."""
    plan: Plan = config.fmri_plan
    keep, dropped = [], []
    for item in items:
        marker = item.get_closest_marker("backend")
        backend = marker.args[0] if marker else _owner_of(item.path)
        if backend is None or backend in plan.selected:
            if backend and not marker:
                item.add_marker(pytest.mark.backend(backend))
            keep.append(item)
        else:
            dropped.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep


@pytest.fixture(autouse=True)
def _backend_marker_gate(request: pytest.FixtureRequest) -> None:
    marker = request.node.get_closest_marker("backend")
    if marker:
        request.config.fmri_plan.gate(marker.args[0])


# --- shared helpers -----------------------------------------------------------

def write_curriculum(phases: list[dict], path: Path) -> Path:
    """Write ``phases`` as a config file and return its path."""
    path.write_text(json.dumps({"curriculum": phases}, indent=1))
    return path
