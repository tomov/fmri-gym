"""Protocol 2 & 3: replay fidelity against the REAL production pipeline.

Protocol 1 only tests the adapter directly, not whether a recorded block
replays correctly through the real path: ``fmri_gym.logging.Logger`` writes
``events.jsonl``/``frames.h5`` during a run, and
``fmri_gym.replay.reconstruct_episode`` reads them back later, on a
long-lived engine that's already played prior episodes -- the precondition
``reconstruct_episode``'s ``warm_up()`` hook exists for.

:func:`check_episode_reconstruction` (Protocol 2) writes a real multi-episode
block through the real :class:`~fmri_gym.logging.Logger`, then diffs
``reconstruct_episode``'s replayed frames against what was actually recorded.

:func:`check_savestate_restore` (Protocol 3): for a backend with an
in-memory savestate, capture mid-episode, restore into a fresh adapter, and
check the tail matches the ORIGINAL recorded tail, not just another live run.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import traceback
import zlib
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


class _Recorder:
    """Wraps a real adapter, recording what ``reconstruct_episode`` sees as it
    replays: forwards everything except ``step``/``capture``, which it also
    records through, in the same order (``step`` then ``capture``) real
    ``fmri_gym/run.py`` uses.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.frames: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.term: list[bool] = []
        self.trunc: list[bool] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def step(self, action: Any) -> tuple:
        obs, reward, terminated, truncated, info = self._inner.step(action)
        self.rewards.append(float(reward))
        self.term.append(bool(terminated))
        self.trunc.append(bool(truncated))
        return obs, reward, terminated, truncated, info

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> Any:
        fs = self._inner.capture(obs, info, want_blob=want_blob)
        self.frames.append(np.array(self._inner.render()))
        return fs


def _decode_state(line: dict) -> bytes | None:
    """Inverse of the writer's ``zlib.compress`` + base64 of a frame's ``"state"``."""
    raw = line.get("state")
    if raw is None:
        return None
    return zlib.decompress(base64.b64decode(raw))


def record_real_block(
    config_path: str, *, episode_steps: tuple[int, ...] = (6, 40),
    state_stride: int = 10, action_seed: int = 42, base_seed: int = 1000,
    outdir: str | None = None,
) -> dict:
    """Play a real multi-episode block through the real ``Logger``, on one adapter.

    One long-lived adapter plays every episode back to back, matching
    ``fmri_gym/run.py`` and the precondition the VizDoom episode-2+ bug needed.

    :param config_path: a ``configs/dbp_games/*.json`` file.
    :param episode_steps: step budget per episode, e.g. ``(6, 40)`` = a short
        throwaway episode 0, then a longer episode 1 to check reconstruction on.
    :param state_stride: how often ``capture(want_blob=True)`` is asked for
        (``ep_frame == 0`` always included, see ``docs/SAVESTATES.md``).
    :param action_seed: seed for sampling each episode's actions
        (``action_seed + episode_id``).
    :param base_seed: episode ``i``'s env seed is ``base_seed + i``, matching
        ``fmri_gym/run.py``.
    :param outdir: where to write the block (default: a fresh temp dir).
    :return: ``{"block_dir", "actions", "outdir", "own_outdir"}``; caller owns
        cleanup of ``outdir`` when ``own_outdir`` is True.
    """
    from fmri_gym.adapters import get_adapter
    from fmri_gym.logging import Logger

    own_outdir = outdir is None
    outdir = outdir or tempfile.mkdtemp(prefix="fmri_gym_replay_")

    phase = load_phase(config_path)
    # Every frame kept (frame_stride=1): this module diffs frame-for-frame, a
    # block recorded for real use might thin that out.
    rec_phase = {**phase, "frame_stride": 1}
    backend = rec_phase.get("backend", "gym")

    # Sampled before the recording adapter is built: some backends (AI
    # GameStore's Playwright sync API) can't have two instances open at once.
    actions_per_episode = [sample_actions(rec_phase, steps, action_seed + i)
                           for i, steps in enumerate(episode_steps)]

    adapter = get_adapter(backend, rec_phase)
    logger = Logger(outdir, subject="determinism-check", curriculum=[rec_phase], clock=None)
    try:
        block_name = logger.open_block(0, backend, rec_phase["game"], rec_phase, base_seed)
        for i, steps in enumerate(episode_steps):
            seed = base_seed + i
            actions = actions_per_episode[i]
            adapter.reset(seed)
            logger.log(type="episode_start", episode_id=i, seed=seed)
            ep_frame = 0
            for action in actions:
                obs, reward, terminated, truncated, info = adapter.step(action)
                save_state = ep_frame % state_stride == 0
                fs = adapter.capture(obs, info, want_blob=save_state)
                frame = adapter.render()
                logger.log_frame({
                    "episode_id": i, "ep_frame": ep_frame, "action": action,
                    "reward": float(reward), "terminated": bool(terminated),
                    "truncated": bool(truncated), "variables": fs.variables,
                }, frame=frame, state=fs.blob)
                ep_frame += 1
                if terminated or truncated:
                    break
            logger.log(type="episode_end", episode_id=i, terminated=bool(terminated),
                       truncated=bool(truncated))
        logger.close_block(n_episodes=len(episode_steps))
    finally:
        adapter.close()
        logger.close()  # joins the writer process: files are fully flushed and closed
    return {"block_dir": os.path.join(outdir, block_name), "actions": actions_per_episode,
            "outdir": outdir, "own_outdir": own_outdir}


def check_episode_reconstruction(
    config_path: str, *, episode_steps: tuple[int, ...] = (6, 40),
    state_stride: int = 10, action_seed: int = 42, base_seed: int = 1000,
) -> dict:
    """Protocol 2: does ``fmri_gym.replay.reconstruct_episode`` reproduce every
    episode of a real recorded block, frame for frame -- including episode 1+,
    on the engine's second (and later) ``reset()``?

    :return: ``{"ok", "episodes": {episode_id: {"n_frames", "n_mismatched",
        "first_mismatch", "rewards_match", "term_match", "trunc_match"}}}``.
        ``ok`` is False if the harness itself errored (see ``"error"``), not
        merely if an episode mismatched -- check each episode's own
        ``n_mismatched == 0`` for the actual pass/fail signal.
    """
    from fmri_gym.logging import read_events, read_frames
    from fmri_gym.replay import reconstruct_episode

    report: dict[str, Any] = {"config": config_path, "protocol": "episode_reconstruction"}
    rec = None
    try:
        rec = record_real_block(
            config_path, episode_steps=episode_steps, state_stride=state_stride,
            action_seed=action_seed, base_seed=base_seed)
        block_dir = rec["block_dir"]
        events = read_events(block_dir)
        phase = events[0]["phase"]
        backend = events[0]["backend"]
        frames, frame_index = read_frames(block_dir)
        row_of = {int(fi): row for row, fi in enumerate(frame_index)}

        episodes: dict[str, Any] = {}
        all_ok = True
        for episode_id in range(len(episode_steps)):
            ep_lines = [e for e in events if e.get("type") == "frame"
                       and e["episode_id"] == episode_id]
            original_frames = [frames[row_of[e["frame"]]] for e in ep_lines]
            original_rewards = [e["reward"] for e in ep_lines]
            original_term = [e["terminated"] for e in ep_lines]
            original_trunc = [e["truncated"] for e in ep_lines]

            from fmri_gym.adapters import get_adapter
            fresh = get_adapter(backend, phase)
            recorder = _Recorder(fresh)
            try:
                reconstruct_episode(block_dir, episode_id=episode_id, adapter=recorder)
            finally:
                recorder.close()

            n = min(len(original_frames), len(recorder.frames))
            first_mismatch = None
            n_mismatched = 0
            for i in range(n):
                if not np.array_equal(original_frames[i], recorder.frames[i]):
                    n_mismatched += 1
                    if first_mismatch is None:
                        first_mismatch = i
            ep_ok = (n_mismatched == 0 and len(original_frames) == len(recorder.frames)
                     and original_rewards[:n] == recorder.rewards[:n]
                     and original_term[:n] == recorder.term[:n]
                     and original_trunc[:n] == recorder.trunc[:n])
            episodes[str(episode_id)] = {
                "n_frames_original": len(original_frames), "n_frames_reconstructed": len(recorder.frames),
                "n_mismatched": n_mismatched, "first_mismatch": first_mismatch,
                "rewards_match": original_rewards[:n] == recorder.rewards[:n],
                "term_match": original_term[:n] == recorder.term[:n],
                "trunc_match": original_trunc[:n] == recorder.trunc[:n],
                "identical": ep_ok,
            }
            all_ok = all_ok and ep_ok
        report["episodes"] = episodes
        report["ok"] = all_ok
    except Exception as e:
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
    finally:
        if rec and rec.get("own_outdir"):
            shutil.rmtree(rec["outdir"], ignore_errors=True)
    return report


def check_savestate_restore(
    config_path: str, *, steps: int = 60, state_stride: int = 5,
    action_seed: int = 42, base_seed: int = 1000,
) -> dict:
    """Protocol 3: capture mid-episode, restore into a fresh adapter, compare
    the tail against the ORIGINAL recorded tail (not just another live run).

    :return: ``{"ok", "has_savestate", "restore_frame", "n_mismatched",
        "first_mismatch", "rewards_match"}``; ``has_savestate: False`` (most
        backends, see ``docs/SAVESTATES.md``) is a pass, not a skip -- there
        is nothing to restore and nothing to test.
    """
    from fmri_gym.adapters import get_adapter
    from fmri_gym.logging import read_events, read_frames

    report: dict[str, Any] = {"config": config_path, "protocol": "savestate_restore"}
    rec = None
    try:
        rec = record_real_block(
            config_path, episode_steps=(steps,), state_stride=state_stride,
            action_seed=action_seed, base_seed=base_seed)
        block_dir = rec["block_dir"]
        actions = rec["actions"][0]
        events = read_events(block_dir)
        phase = events[0]["phase"]
        backend = events[0]["backend"]
        seed = [e for e in events if e["type"] == "episode_start"][0]["seed"]
        frame_lines = [e for e in events if e.get("type") == "frame" and e["episode_id"] == 0]

        blob_frames = [e for e in frame_lines if e.get("state") is not None]
        if not blob_frames:
            report["ok"] = True
            report["has_savestate"] = False
            return report

        mid = blob_frames[len(blob_frames) // 2]
        k = mid["ep_frame"]  # single-episode block: ep_frame == position in `actions`
        blob = _decode_state(mid)

        frames, frame_index = read_frames(block_dir)
        row_of = {int(fi): row for row, fi in enumerate(frame_index)}
        tail_lines = [e for e in frame_lines if e["ep_frame"] >= k]
        original_tail_frames = [frames[row_of[e["frame"]]] for e in tail_lines]
        original_tail_rewards = [e["reward"] for e in tail_lines]

        adapter2 = get_adapter(backend, phase)
        try:
            adapter2.reset(seed)
            for action in actions[:k]:
                adapter2.step(action)
            adapter2.restore(blob)
            tail_frames = [np.array(adapter2.render())]
            tail_rewards: list[float] = []
            # The blob already reflects actions[k]; the next new one is k + 1.
            for action in actions[k + 1:]:
                obs, reward, terminated, truncated, info = adapter2.step(action)
                tail_rewards.append(float(reward))
                tail_frames.append(np.array(adapter2.render()))
                if terminated or truncated:
                    break
        finally:
            adapter2.close()

        n = min(len(original_tail_frames), len(tail_frames))
        first_mismatch = None
        n_mismatched = 0
        for i in range(n):
            if not np.array_equal(original_tail_frames[i], tail_frames[i]):
                n_mismatched += 1
                if first_mismatch is None:
                    first_mismatch = i
        report["ok"] = True
        report["has_savestate"] = True
        report["restore_frame"] = k
        report["n_frames_compared"] = n
        report["n_mismatched"] = n_mismatched
        report["first_mismatch"] = first_mismatch
        # original_tail_rewards[0] is the restore frame's own reward (already
        # in the blob); tail_rewards starts one frame later.
        report["rewards_match"] = (
            original_tail_rewards[1:1 + len(tail_rewards)] == tail_rewards)
        report["identical"] = (n_mismatched == 0 and report["rewards_match"]
                                and len(original_tail_frames) == len(tail_frames))
    except NotImplementedError:
        report["ok"] = True
        report["has_savestate"] = False
    except Exception as e:
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
    finally:
        if rec and rec.get("own_outdir"):
            shutil.rmtree(rec["outdir"], ignore_errors=True)
    return report


def main() -> None:
    """``python -m tools.determinism.replay <reconstruction|restore> <config>``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("subcommand", choices=["reconstruction", "restore"])
    ap.add_argument("config")
    args = ap.parse_args()
    func = (check_episode_reconstruction if args.subcommand == "reconstruction"
            else check_savestate_restore)
    report = func(args.config)
    print(json.dumps(report, default=str))
    sys.exit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
