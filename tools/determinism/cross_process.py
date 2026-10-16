"""Protocol 1: cross-process determinism.

The reproducibility rule in ``AGENTS.md``, made executable: *"replay a
recorded block in a second process and compare the frames bit for bit"* --
not a same-interpreter comparison, which would never catch a hash-order bug
(``PYTHONHASHSEED`` is fixed for the life of one interpreter and only
randomizes between processes).

:func:`check_cross_process` samples one fixed action sequence from the
phase's own keymap, then replays it in three independent ``python``
subprocesses (``PYTHONHASHSEED`` left unset, so each gets its own random hash
seed, the default):

- twice with the *same* seed (``seed_a``) -- these two MUST come back
  bit-identical (frame hashes, rewards, ``terminated``/``truncated``,
  ``capture().variables``). Anything else is a real bug: the trajectory and
  the interpreter are identical except for a hash seed nothing in the game
  should depend on.
- once with a *different* seed (``seed_b``) -- expected to diverge if the
  game reads its seed at all; coming back identical is not automatically a
  bug (the game may have no seed-dependent randomness, or this trajectory
  never reached the branch that uses it), but is worth a human look before
  being called "fine".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any

import numpy as np


def load_phase(config_path: str) -> dict:
    """The first ``"game"`` phase of a curriculum config."""
    with open(config_path) as f:
        cfg = json.load(f)
    for p in cfg["curriculum"]:
        if p["type"] == "game":
            return p
    raise ValueError(f"{config_path}: no game phase in curriculum")


def sample_actions(phase: dict, steps: int, action_seed: int) -> list:
    """A fixed-length list of real actions, sampled from the phase's own keymap.

    Real actions the engine accepts (``keymap.resolve(combo)``), never
    ``action_space.sample()``, which can hand a ``MultiBinary`` adapter a
    combination its keymap never emits.
    """
    from fmri_gym.adapters import get_adapter

    adapter = get_adapter(phase.get("backend", "gym"), phase)
    try:
        pool = [adapter.keymap.resolve(combo) for combo in adapter.keymap.combos]
    finally:
        adapter.close()
    if not pool:
        raise ValueError("keymap has no combos to sample actions from")
    rng = np.random.default_rng(action_seed)
    idx = rng.integers(0, len(pool), size=steps)
    return [pool[i] for i in idx]


def _frame_hash(frame: Any) -> str:
    """A 16-hex-char SHA-1 of a rendered frame array (``"none"`` for ``None``)."""
    if frame is None:
        return "none"
    return hashlib.sha1(np.asarray(frame).tobytes()).hexdigest()[:16]


def _variables_repr(variables: dict) -> str:
    """A sorted-key JSON string of a ``capture().variables`` dict, for exact diffing."""
    try:
        return json.dumps(variables, sort_keys=True, default=lambda o: np.asarray(o).tolist())
    except Exception:
        return str(variables)


def _run_trajectory(phase: dict, seed: int, actions: list) -> dict:
    """Reset + step through ``actions`` once, recording a hashable trace.

    Mirrors real ``fmri_gym/run.py`` usage: ``capture()`` is only ever called
    with a STEP's ``(obs, info)``, never ``reset``'s (some adapters' reset
    info lacks fields step info has, e.g. crafter's ``"inventory"``).
    """
    from fmri_gym.adapters import get_adapter

    adapter = get_adapter(phase.get("backend", "gym"), phase)
    trace: dict = {"frame_hashes": [], "rewards": [], "term": [], "trunc": [], "variables": []}
    try:
        adapter.reset(seed)
        trace["frame_hashes"].append(_frame_hash(adapter.render()))
        for action in actions:
            obs, reward, terminated, truncated, info = adapter.step(action)
            trace["rewards"].append(float(reward))
            trace["term"].append(bool(terminated))
            trace["trunc"].append(bool(truncated))
            fs = adapter.capture(obs, info, want_blob=False)
            trace["variables"].append(_variables_repr(fs.variables))
            trace["frame_hashes"].append(_frame_hash(adapter.render()))
            if terminated or truncated:
                break
    finally:
        adapter.close()
    return trace


def diff_traces(a: dict, b: dict) -> dict:
    """Where two :func:`_run_trajectory` traces first disagree, and whether they're identical."""
    # frame_hashes: [reset, after step 0, after step 1, ...] (len = n_steps+1).
    # rewards/variables: [after step 0, after step 1, ...] (len = n_steps), so
    # index i there lines up with frame_hashes[i + 1], not frame_hashes[i].
    n = min(len(a["frame_hashes"]), len(b["frame_hashes"]))
    nr = min(len(a["rewards"]), len(b["rewards"]))
    nv = min(len(a["variables"]), len(b["variables"]))
    first_frame_diff = first_reward_diff = first_var_diff = None
    for i in range(n):
        if a["frame_hashes"][i] != b["frame_hashes"][i]:
            first_frame_diff = i
            break
    for i in range(nv):
        if a["variables"][i] != b["variables"][i]:
            first_var_diff = i
            break
    for i in range(nr):
        if a["rewards"][i] != b["rewards"][i]:
            first_reward_diff = i
            break
    return {
        "n_steps_a": len(a["rewards"]), "n_steps_b": len(b["rewards"]),
        "first_frame_diff": first_frame_diff, "first_reward_diff": first_reward_diff,
        "first_variables_diff": first_var_diff,
        "term_match": a["term"][:nr] == b["term"][:nr],
        "trunc_match": a["trunc"][:nr] == b["trunc"][:nr],
        "identical": (first_frame_diff is None and first_reward_diff is None
                      and first_var_diff is None and len(a["rewards"]) == len(b["rewards"])),
    }


def _spawn(config: str, seed: int, actions_path: str, out_path: str, env: dict) -> None:
    """Run one trajectory in a fresh ``python`` subprocess, via this module's own CLI."""
    subprocess.run(
        [sys.executable, "-m", "tools.determinism.cross_process", "_run_one",
         config, str(seed), actions_path, out_path],
        check=True, capture_output=True, text=True, env=env, cwd=_repo_root())


def _repo_root() -> str:
    """The repo root (two levels up from this file), so the subprocess's cwd has ``fmri_gym``."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_cross_process(
    config_path: str, *, steps: int = 60, action_seed: int = 42,
    seed_a: int = 1000, seed_b: int = 2000, tmpdir: str | None = None,
) -> dict:
    """Protocol 1: same-seed / diff-seed cross-process reproducibility.

    :param config_path: a ``configs/dbp_games/*.json`` file.
    :param steps: action-sequence length (shorter if the episode ends first).
    :param action_seed: seed for sampling the one fixed action sequence.
    :param seed_a: the "same seed" pair's env seed.
    :param seed_b: the "different seed" run's env seed.
    :param tmpdir: scratch dir for inter-process trace files (default: a fresh
        one, removed afterwards).
    :return: ``{"ok", "same_seed_cross_process", "diff_seed_cross_process",
        "episode_ended_early", ...}`` -- ``ok`` is False (with ``"error"``) if
        the subprocesses themselves failed to run, not if they merely
        disagreed (that is ``same_seed_cross_process["identical"]`` being
        False, which is the actual failure signal for a PR check).
    """
    own_tmpdir = tmpdir is None
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="fmri_gym_determinism_")
    report: dict[str, Any] = {"config": config_path, "protocol": "cross_process"}
    try:
        phase = load_phase(config_path)
        report["backend"] = phase.get("backend")
        report["game"] = phase.get("game")

        actions = sample_actions(phase, steps, action_seed)
        tag = uuid.uuid4().hex[:8]
        actions_path = os.path.join(tmpdir, f"actions_{tag}.json")
        out_a1 = os.path.join(tmpdir, f"a1_{tag}.json")
        out_a2 = os.path.join(tmpdir, f"a2_{tag}.json")
        out_b = os.path.join(tmpdir, f"b_{tag}.json")
        with open(actions_path, "w") as f:
            json.dump(actions, f, default=lambda o: __import__("numpy").asarray(o).tolist())

        # No PYTHONHASHSEED pin: each subprocess gets Python's own fresh random
        # hash seed (the default), so a same-seed divergence between them is
        # the hash-order class of bug AGENTS.md calls out, not an artifact of
        # one shared hash ordering throughout.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}

        _spawn(config_path, seed_a, actions_path, out_a1, env)
        _spawn(config_path, seed_a, actions_path, out_a2, env)
        _spawn(config_path, seed_b, actions_path, out_b, env)
        with open(out_a1) as f:
            trace_a1 = json.load(f)
        with open(out_a2) as f:
            trace_a2 = json.load(f)
        with open(out_b) as f:
            trace_b = json.load(f)

        report["same_seed_cross_process"] = diff_traces(trace_a1, trace_a2)
        report["diff_seed_cross_process"] = diff_traces(trace_a1, trace_b)
        report["episode_ended_early"] = len(trace_a1["rewards"]) < steps
        report["ok"] = True
    except subprocess.CalledProcessError as e:
        report["ok"] = False
        report["error"] = f"subprocess failed: {e}"
        report["stderr"] = (e.stderr or "")[-4000:]
    except Exception as e:
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
    finally:
        if own_tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    return report


def _cli_run_one() -> None:
    """Internal subcommand: run one trajectory, write its trace to a JSON file.

    Invoked by :func:`_spawn` as ``python -m tools.determinism.cross_process
    _run_one <config> <seed> <actions.json> <out.json>`` -- this is the thing
    that actually runs in the fresh subprocess.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("seed", type=int)
    ap.add_argument("actions")
    ap.add_argument("out")
    args = ap.parse_args(sys.argv[2:])
    phase = load_phase(args.config)
    with open(args.actions) as f:
        actions = json.load(f)
    trace = _run_trajectory(phase, args.seed, actions)
    with open(args.out, "w") as f:
        json.dump(trace, f)


def main() -> None:
    """``python -m tools.determinism.cross_process <config> [--steps N ...]``."""
    if len(sys.argv) > 1 and sys.argv[1] == "_run_one":
        return _cli_run_one()

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--action-seed", type=int, default=42)
    ap.add_argument("--seed-a", type=int, default=1000)
    ap.add_argument("--seed-b", type=int, default=2000)
    args = ap.parse_args()
    report = check_cross_process(
        args.config, steps=args.steps, action_seed=args.action_seed,
        seed_a=args.seed_a, seed_b=args.seed_b)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report.get("ok") and report["same_seed_cross_process"]["identical"] else 1)


if __name__ == "__main__":
    main()
