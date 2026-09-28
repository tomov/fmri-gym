"""Play a curriculum's game blocks with a policy, not a person.

The comparison the rig is built for needs a model to play the worlds the subject
played and be logged the same way. So this shares everything that defines the
task: the curriculum JSON, the EnvAdapter, the episode seeds, the Logger and its
log schema. It shares none of the run loop, which exists for a scanner: trigger
wait, fps pacing, a pygame window, a key event queue, a pause menu. A model needs
no window and no wall clock, and faking them would only slow the block to human
speed.

Episode seeds are the block's own (``seed`` + episode index), which is what makes
this the same-worlds condition rather than a fresh sample of the game.

Usage:
    python agents/agent_play.py --curriculum configs/dbp_games/crafter__crafter_L4.json \
        --policy random --outdir data/model-random --max-frames 200
    python agents/agent_play.py --curriculum configs/dbp_games/crafter__crafter_L4.json \
        --policy vlm --model claude-sonnet-5 --max-frames 60
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter

from policies import Policy, RandomPolicy, VLMPolicy

from fmri_gym import Clock, Logger, get_adapter, resume
from fmri_gym.config import load_config, validate_config


def no_key_action(adapter) -> tuple:
    """The action for "no key held", if the block's keys define one.

    A real-time phase must map ``""`` (see :mod:`fmri_gym.adapters.keymap`), so
    standing still is an action like any other. A turn-based phase has no such
    entry, on purpose: nothing happens until a key is pressed, and
    ``resolve(frozenset())`` raises rather than invent a step. A policy in that
    mode is allowed to press nothing too, which costs it the frame.

    :param adapter: the block's adapter.
    :return: ``(action, True)``, or ``(None, False)`` in a turn-based phase.
    """
    try:
        return adapter.keymap.resolve(frozenset()), True
    except ValueError:
        return None, False


def build_policy(args, adapter) -> Policy:
    """Construct the policy named on the command line.

    Both policies are built from the adapter's own keymap, so a model is offered
    exactly the keys the subject was taught and nothing more.

    :param args: parsed command-line arguments.
    :param adapter: the block's adapter, already constructed.
    :return: a ready :class:`~policies.Policy`.
    """
    keys = adapter.keymap.turn_actions()
    noop, has_noop = no_key_action(adapter)
    if args.policy == "random":
        # The no-key action is one of the draws where the phase has one: a
        # subject who presses nothing is playing, and the random baseline holds
        # still at the same rate as any other key.
        return RandomPolicy(list(keys.values()) + ([noop] if has_noop else []),
                            seed=args.policy_seed)
    return VLMPolicy(keys, model=args.model, history=args.history, noop=noop)


def play_episode(adapter, policy: Policy, logger: Logger, clock: Clock, *,
                 seed: int, episode_id: int, state_stride: int, fps: float,
                 turn_based: bool, max_frames: int,
                 carry: resume.Carry | None = None, slot: str | None = None,
                 resume_dir: str | None = None,
                 provenance: dict | None = None) -> tuple[dict, int]:
    """Run one episode under ``policy``, logging every frame.

    Mirrors ``Run._episode`` field for field, minus everything that is about a
    person at a screen. Keeping the two in step by hand is the price of not
    bending the run loop around a case it was not written for.

    :param adapter: the block's adapter.
    :param policy: the policy choosing actions.
    :param logger: the run's logger, with this block open.
    :param clock: the run's clock, for the same time columns humans get.
    :param seed: RNG seed for this episode's ``reset``.
    :param episode_id: index of this episode within the block.
    :param state_stride: save a full state blob every this many frames.
    :param fps: the phase's frames per second, for the HUD's countdown only.
    :param turn_based: the phase's, so the same stretches the run loop steps
        by itself are not put to the policy either
        (:meth:`~fmri_gym.adapters.base.EnvAdapter.autoplay`).
    :param max_frames: stop the episode after this many frames.
    :param carry: a world an earlier block left off in
        (:mod:`fmri_gym.resume`), restored over the one ``seed`` opens.
    :param slot: the slot this block hands its world on in, if the frame
        budget is what ends this episode.
    :param resume_dir: the folder those slots live in.
    :param provenance: what the slot's file records about where the world
        came from.
    :return: the episode's ``episode_end`` record, as logged, and the number
        of frames the policy pressed nothing on.
    """
    policy.reset()
    observation, info = adapter.reset(seed)
    if carry is not None:
        adapter.restore(carry.blob)
    logger.log(type="episode_start", episode_id=episode_id, seed=seed,
               resumed=carry is not None)

    terminated = truncated = False
    score = 0.0                         # the episode's cumulative reward
    ep_frame = skipped = 0
    auto = None                         # see EnvAdapter.autoplay
    while not (terminated or truncated) and ep_frame + skipped < max_frames:
        if auto is not None:
            # The env is in a state it takes no action in, so the policy is not
            # asked for one, exactly as the subject is not asked to press. The
            # frames still cost budget: they are engine steps, and the subject
            # pays for them in block time.
            action = auto
        else:
            # The subject's block ends on a wall clock; the model's ends on this
            # frame budget, so the countdown it reads is over its own budget at the
            # block's fps. The alternative, a real clock, would tell the model how
            # long its own inference took, which is not something the game shows.
            remaining = (max_frames - ep_frame - skipped) / fps
            status = list(adapter.hud(score, remaining) or [])
            over = adapter.overlay()
            action = policy.act(adapter.render(), status + list(over[0] if over else []))
            if action is None:
                # A turn-based phase with no key pressed: nothing steps, exactly as
                # in the run loop. It costs the frame, which is what keeps a policy
                # that never names a key from looping here forever.
                skipped += 1
                continue

        observation, reward, terminated, truncated, info = adapter.step(action)
        auto = adapter.autoplay(info) if turn_based else None
        score += float(reward)
        # A savestate at the episode's first frame (the replay anchor), then every stride.
        fs = adapter.capture(observation, info, want_blob=ep_frame % state_stride == 0)
        fields = {
            "episode_id": episode_id, "ep_frame": ep_frame, "action": action, "reward": reward,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "run_time": clock.run_time(),
            # A policy has no screen, so no frame of this block was ever flipped.
            # NaN, not the step time: the human column is a measured photon onset,
            # and nothing should be able to average the two together by accident.
            "flip_time": float("nan"), "wall_time": clock.wall_time(),
            "variables": fs.variables,
        }
        if isinstance(info, dict) and "env_action" in info:
            fields["env_action"] = info["env_action"]
        logger.log_frame(fields, frame=adapter.render(), state=fs.blob)
        ep_frame += 1

    # Same vocabulary the human blocks log, from the same method: "playing" is
    # the budget cutting the episode off, where a subject's block clock would.
    outcome, _ = adapter.outcome(terminated, truncated)
    end = {"episode_id": episode_id, "outcome": outcome, "terminated": bool(terminated),
           "truncated": bool(truncated), "score": score, "n_pacing_resets": 0}
    logger.log(type="episode_end", **end)
    # ... and the same rule about which world is still the player's: only the
    # one the budget interrupted carries on into the next block that names
    # this slot. A model chains among its own blocks, not into a subject's:
    # the folder is per run (--resume-dir), like the session folder is.
    if slot is not None and outcome == "playing" and ep_frame:
        resume.save(resume_dir, slot,
                    adapter.capture(observation, info, want_blob=True).blob,
                    {**(provenance or {}), "episode_id": episode_id, "n_frames": ep_frame,
                     "score": score, "run_time": clock.run_time(),
                     "wall_time": clock.wall_time()})
    return end, skipped


def play_block(phase: dict, index: int, args, logger: Logger,
               clock: Clock) -> str:
    """Play every episode of one game block and write its log.

    :param phase: the game-phase config, used exactly as a run uses it.
    :param index: phase index in the curriculum (names the block's folder).
    :param args: parsed command-line arguments.
    :param logger: the run's logger writing the block.
    :param clock: the run's clock.
    :return: the block's folder name.
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

    # The world this block continues, on the same terms the run loop resumes
    # one: the model plays the worlds the subject plays, so a block the config
    # says carries over carries over here too, or neither player's blocks mean
    # the same thing. Whether the backend can is known only now (built env).
    slot = resume.slot_of(phase)
    carry = None
    if slot is not None:
        if not resume.supported(adapter):
            raise ValueError(f'game phase {index}: "resume": {slot!r}, but the {backend} '
                             "adapter has no savestate to resume from (EnvAdapter.restore)")
        carry = resume.load(args.resume_dir, slot)
        print(f"  resume slot {slot!r}: "
              f"{'continuing ' + carry.path if carry else 'no world yet, starting one'}")
    resumed_from = resume.describe(carry)
    provenance = {"subject": args.subject, "backend": backend, "game": phase["game"],
                  "block": index, "policy": args.policy}

    data_dir = logger.open_block(index, backend, phase["game"], phase, base_seed)
    if carry is not None:
        # The block keeps its own copy of the world it was handed; the frame a
        # resumed episode opens on is in no `frame` line this block would
        # otherwise hold. Same line the run loop writes.
        logger.log(type="resume", slot=slot, source=carry.header, path=carry.path,
                   state=carry.blob)
    episodes: list[dict] = []           # each episode's episode_end record
    skipped = 0
    for episode_id in range(args.n_episodes):
        before = logger.n_frames
        end, n_skipped = play_episode(
            adapter, policy, logger, clock, seed=base_seed + episode_id,
            episode_id=episode_id, state_stride=state_stride,
            fps=phase["fps"], turn_based=bool(phase.get("turn_based", False)),
            max_frames=args.max_frames,
            # Only the first: the episodes after it are there because that
            # world ended, and an ended world is not resumed.
            carry=carry if episode_id == 0 else None, slot=slot,
            resume_dir=args.resume_dir, provenance=provenance)
        episodes.append(end)
        skipped += n_skipped
        print(f"  episode {episode_id} (seed {base_seed + episode_id}): "
              f"{logger.n_frames - before} frames, {end['outcome']}, score {end['score']:g}")

    extra = adapter.block_extra()
    adapter.close()
    summary = {
        "n_episodes": len(episodes), "n_frames": logger.n_frames,
        "outcomes": dict(Counter(e["outcome"] for e in episodes)),
        "total_reward": sum(e["score"] for e in episodes),
        # Which policy produced the block belongs in the block, not in a folder
        # name: a model's log and a human's are otherwise alike by design.
        "policy": args.policy, "policy_model": args.model if args.policy == "vlm" else "",
        # Three ways a model wastes a frame, kept apart: it named no key, it
        # named something that is not a key, or the call never came back.
        "skipped_frames": skipped, "invalid_replies": policy.invalid,
        "dropped_calls": policy.dropped,
        **({"resume": slot, "resumed_from": resumed_from} if slot else {}),
    }
    logger.close_block(**summary, audio={}, extra=extra)
    logger.log_phase({"index": index, "type": "game", "backend": backend,
                      "game": phase["game"], **summary, "data_dir": data_dir})
    return data_dir


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
    p.add_argument("--resume-dir",
                   help="where blocks that name a \"resume\" slot keep the world they left "
                        "off in (default: <outdir>/resume, so a curriculum's blocks continue "
                        "each other but two commands do not). Point two commands at the same "
                        "folder to chain them, as a session's runs are chained.")
    args = p.parse_args()

    config = load_config(args.curriculum)
    # The same check fmri_play runs, so one config file suits both players.
    problems = validate_config(config)
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))
    curriculum = config["curriculum"]
    outdir = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")
    args.resume_dir = args.resume_dir or os.path.join(outdir, "resume")
    clock = Clock()
    # No scanner here, so the run's zero is simply when it started: the time
    # columns stay the same shape as a human run's, measured from that.
    clock.trigger()
    logger = Logger(outdir, args.subject, curriculum, clock)
    logger.set_trigger_time()

    for index, phase in enumerate(curriculum):
        if phase.get("type") != "game":
            continue
        print(f"block {index}: {phase.get('game')} via {args.policy}")
        print(" ->", play_block(phase, index, args, logger, clock))
    print("Manifest:", logger.save_manifest())
    logger.close()


if __name__ == "__main__":
    main()
