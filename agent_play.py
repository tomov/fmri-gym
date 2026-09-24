"""agent_play.py -- play a curriculum's game blocks with a policy, not a person.

The comparison the rig is built for needs a model to play the worlds the subject
played and be logged the same way. So this shares everything that defines the
task -- the curriculum JSON, the EnvAdapter, the episode seeds, the Logger and
its npz schema -- and shares none of the run loop, which exists for a
scanner: trigger wait, fps pacing, a pygame window, a key event queue. A model
needs no window and no wall clock, and faking them would only slow the block to
human speed.

Episode seeds are the block's own (`seed` + episode index), which is what makes
this the same-worlds condition rather than a fresh sample of the game.

Usage:
    python agent_play.py --curriculum configs/dbp_games/crafter__crafter.json \
        --policy random --outdir data/model-random --max-frames 200
    python agent_play.py --curriculum configs/dbp_games/crafter__crafter.json \
        --policy vlm --model claude-sonnet-5 --max-frames 60
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

from fmri_gym import Clock, Logger, get_adapter
from fmri_gym.config import load_config, validate_config
from fmri_gym.policies import Policy, RandomPolicy, VLMPolicy


def build_policy(args, adapter) -> Policy:
    """Construct the policy named on the command line.

    Both policies are built from the adapter's own keyspec, so a model is
    offered exactly the keys the subject was taught and nothing more.

    :param args: parsed command-line arguments.
    :param adapter: the block's adapter, already constructed.
    :return: a ready :class:`~fmri_gym.policies.Policy`.
    """
    menu = adapter.keyspec.key_to_action_map()
    noop = adapter.keyspec.noop
    if args.policy == "random":
        return RandomPolicy(list(menu.values()) + [noop], seed=args.policy_seed)
    return VLMPolicy(menu, model=args.model, history=args.history, noop=noop)


def play_episode(adapter, policy: Policy, frames: dict, clock: Clock, *,
                 seed: int, episode_id: int, state_stride: int,
                 max_frames: int) -> int:
    """Run one episode under ``policy``, appending to ``frames``.

    Mirrors ``Run._episode`` field for field, minus everything that is
    about a person at a screen. Keeping the two in step by hand is the price of
    not bending the run loop around a case it was not written for.

    :param adapter: the block's adapter.
    :param policy: the policy choosing actions.
    :param frames: mutable frame-log dict; lists are appended in place.
    :param clock: the run's clock, for the same time columns humans get.
    :param seed: RNG seed for this episode's ``reset``.
    :param episode_id: index of this episode within the block.
    :param state_stride: save a full state blob every this many frames.
    :param max_frames: stop the episode after this many frames.
    :return: the number of frames played.
    """
    frames["episode_seeds"].append(seed)
    overlay = getattr(adapter, "overlay", lambda: None)
    policy.reset()
    obs, info = adapter.reset(seed)

    terminated = truncated = False
    ep_frame = 0
    while not (terminated or truncated) and ep_frame < max_frames:
        action = policy.act(adapter.render(), overlay())
        obs, reward, terminated, truncated, info = adapter.step(action)
        save_blob = (ep_frame % state_stride == 0)
        ep_frame += 1
        fs = adapter.capture(obs, info, want_blob=save_blob)

        frames["action"].append(action)
        frames["reward"].append(reward)
        frames["terminated"].append(bool(terminated))
        frames["truncated"].append(bool(truncated))
        frames["episode_id"].append(episode_id)
        frames["run_time"].append(clock.run_time())
        # A policy has no screen, so no frame of this block was ever flipped.
        # NaN, not the step time: the human column is a measured photon onset
        # and nothing should be able to average the two together by accident.
        frames["flip_time"].append(float("nan"))
        frames["wall_time"].append(clock.wall_time())
        frames["state_blob"].append(fs.blob)
        for k, v in fs.variables.items():
            frames["variables"][k].append(v)
    return ep_frame


def play_block(phase: dict, index: int, args, logger: Logger,
               clock: Clock) -> str:
    """Play every episode of one game block and write its npz.

    :param phase: the game-phase config, used exactly as a run uses it.
    :param index: phase index in the curriculum (names the output file).
    :param args: parsed command-line arguments.
    :param logger: the run's logger writing the block.
    :param clock: the run's clock.
    :return: path of the written npz.
    """
    backend = phase.get("backend", "gym")
    base_seed = phase.get("seed", 1000 + index)
    state_stride = max(1, int(phase.get("state_stride", 1)))
    # A policy has no ears. Whatever the block's config plays to a subject as a
    # sound, it reads as a line of text, so the two players are told the same
    # things; scanner configs keep that line off the screen to hold the gaze on
    # the frame. Adapters without cues ignore the key.
    adapter = get_adapter(backend, {**phase, "cue_overlay": True})
    policy = build_policy(args, adapter)

    frames = defaultdict(list)
    frames["variables"] = defaultdict(list)
    for episode_id in range(args.n_episodes):
        n = play_episode(adapter, policy, frames, clock,
                         seed=base_seed + episode_id, episode_id=episode_id,
                         state_stride=state_stride, max_frames=args.max_frames)
        print(f"  episode {episode_id} (seed {base_seed + episode_id}): "
              f"{n} frames")

    extra = getattr(adapter, "block_extra", lambda: None)() or {}
    # Which policy produced the block belongs in the block, not in a filename:
    # a model npz and a human npz are otherwise indistinguishable by design.
    extra["policy"] = args.policy
    extra["policy_model"] = args.model if args.policy == "vlm" else ""
    adapter.close()
    path = logger.save_game_block(index, backend, phase["game"], frames,
                                  extra=extra)
    logger.log_phase({
        "index": index, "type": "game", "backend": backend,
        "game": phase["game"], "policy": args.policy,
        "n_episodes": args.n_episodes, "n_frames": len(frames["action"]),
        "total_reward": sum(float(r) for r in frames["reward"]),
        "invalid_replies": policy.invalid, "dropped_calls": policy.dropped,
        "data_file": path.split("/")[-1]})
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Play a curriculum with a policy.")
    p.add_argument("--curriculum", required=True)
    p.add_argument("--subject", default="model", help="stored in the manifest")
    p.add_argument("--outdir")
    p.add_argument("--policy", default="random", choices=["random", "vlm"])
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--history", type=int, default=4,
                   help="frames (and past keys) a vlm policy sees")
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--policy-seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.curriculum)
    problems = validate_config(config)  # the same check fmri-play runs, so one file suits both
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))
    curriculum = config["curriculum"]
    outdir = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")
    clock = Clock()
    clock.trigger()
    logger = Logger(outdir, args.subject, curriculum, clock)
    logger.set_trigger_time()

    for index, phase in enumerate(curriculum):
        if phase.get("type") != "game":
            continue
        print(f"block {index}: {phase.get('game')} via {args.policy}")
        print(" ->", play_block(phase, index, args, logger, clock))
    print("Manifest:", logger.save_manifest())


if __name__ == "__main__":
    main()
