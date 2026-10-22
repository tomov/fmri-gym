"""Every config in the repo is one ``fmri-play`` would start, and the suite's table covers every backend.

Engine-free: nothing is built. ``validate_curriculum`` is the check ``fmri_play``
runs before the window opens, so a config that fails here fails at the desk.
"""

from __future__ import annotations

import re

import pytest

from fmri_gym.config import rig
from fmri_gym.config.curriculum import Curriculum, load_curriculum, validate_curriculum
from fmri_gym.config.phases import GamePhase, Message
from fmri_gym.config.rig import RigConfig
from fmri_gym.config.run import RunConfig, key_needs
from tests.backends import BACKENDS, REPO, all_configs, game_phases, phase_backend

CONFIGS = all_configs()
#: The repo's rig files that play with a controller and bind its controls.
PAD_RIGS = [r for r in (RigConfig.open(str(p)) for p in sorted(rig.REPO_RIGS.glob("*/*.json")))
            if r.has_pad and r.bindings]
#: The directions; every other key a game's action needs is a button.
ARROWS = {"UP", "DOWN", "LEFT", "RIGHT"}
#: A key name written bare in a screen's text, where ``[KEY]`` shows what the rig presses.
BARE_KEY = re.compile(r"(?<![\[\w])(SPACE|RETURN|ENTER|LSHIFT|SHIFT|BACKSPACE|TAB|UP|DOWN|"
                      r"LEFT|RIGHT|ARROWS|COMMA|PERIOD)(?![\]\w])")


def _button_actions(game: GamePhase) -> int:
    """How many of a game's actions take a button of their own: a key alone, not an arrow.

    An action done by arrows, or by a combo of keys other actions already take, adds none.
    """
    return sum(any(len(c) == 1 and c[0] not in ARROWS for c in need.combos)
               and not any(set(c) <= ARROWS for c in need.combos)
               for need in key_needs(game) if need.what.startswith("action "))


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_config_validates(path):
    problems = validate_curriculum(load_curriculum(str(path)))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("pad", PAD_RIGS, ids=[r.path for r in PAD_RIGS])
@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_a_six_button_controller_plays_it(path, pad):
    """A rig with a controller plays every curriculum whose games fit six buttons, only those."""
    curriculum = Curriculum.load(str(path))
    run = RunConfig(curriculum=curriculum, rig=pad, subject="sub-01", ses=1, run=1,
                    duration=None, end_fixation=10.0, resume=False, loop=False, dummy_trigger=True)
    is_too_big = any(_button_actions(p) > 6 for p in curriculum.phases if isinstance(p, GamePhase))
    if is_too_big:
        assert run.problems(), "plays on six buttons, though a game in it has more actions"
    else:
        assert not run.problems(), "\n".join(run.problems())


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_screens_name_each_key_in_brackets(path):
    """A screen names a key as ``[KEY]``, and only a key the curriculum asks for."""
    phases = Curriculum.load(str(path)).phases
    keys = {"ARROWS"} | {k for p in phases for need in key_needs(p) for c in need.combos for k in c}
    texts = [p.text for p in phases if isinstance(p, Message)] + [
        p.instructions.text for p in phases if isinstance(p, GamePhase) and p.instructions]
    for text in texts:
        assert not BARE_KEY.search(text), f"write it [{BARE_KEY.search(text)[1]}]: {text!r}"
        unused = set(re.findall(r"\[([A-Z0-9_]+)\]", text)) - keys
        assert not unused, f"{sorted(unused)}: no phase asks for it: {text!r}"


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_config_names_known_backends(path):
    unknown = {phase_backend(g) for g in game_phases(path)} - set(BACKENDS) - {"p5"}
    assert not unknown, f"{unknown}: not a backend get_adapter knows"


def test_table_matches_get_adapter():
    """A backend added to ``get_adapter`` needs a row in ``tests/backends.py`` to be tested."""
    source = (REPO / "fmri_gym" / "adapters" / "__init__.py").read_text()
    registered = set(re.findall(r'backend == "(\w+)"', source))
    for group in re.findall(r"backend in \(([^)]*)\)", source):
        registered |= set(re.findall(r'"(\w+)"', group))
    registered -= {"p5"}      # an alias of aigamestore
    missing = registered - set(BACKENDS)
    extra = set(BACKENDS) - registered
    assert not missing, f"get_adapter knows {sorted(missing)}: add a Backend row in tests/backends.py"
    assert not extra, f"tests/backends.py lists {sorted(extra)}, which get_adapter does not know"


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_smoke_configs_stand_for_their_backend(backend):
    for rel in BACKENDS[backend].smoke:
        path = REPO / rel
        assert path.exists(), f"{rel}: smoke config missing"
        games = game_phases(path)
        assert games, f"{rel}: no game phase"
        assert phase_backend(games[0]) == backend, f"{rel}: first game phase is not {backend}'s"
