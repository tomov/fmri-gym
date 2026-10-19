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
    python agents/agent_play.py --curriculum configs/curricula/crafter/L4.json \
        --policy random --outdir data/model-random --max-frames 200
    python agents/agent_play.py --curriculum configs/curricula/crafter/L4.json \
        --policy vlm --model claude-sonnet-5 --max-frames 60
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter

from policies import Policy, RandomPolicy, VLMPolicy

from fmri_gym import Clock, Logger, get_adapter
from fmri_gym.curriculum import GamePhase, load_curriculum, validate_curriculum
from fmri_gym.run import EpisodeResult


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
                 turn_based: bool, max_frames: int, n_chunks: int = 0,
                 ) -> tuple[EpisodeResult, int, int]:
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
    :param n_chunks: sound chunks the block has logged so far, so this
        episode's ``audio_chunk`` numbers carry on from them.
    :return: how the episode ended, as logged, the number of frames the
        policy pressed nothing on, and the block's chunk count after it.
    """
    policy.reset()
    observation, info = adapter.reset(seed)
    logger.log(type="episode_start", episode_id=episode_id, seed=seed)

    terminated = truncated = False
    score = 0.0                         # the episode's cumulative reward
    ep_frame = skipped = 0
    autoplay_action = None              # see EnvAdapter.autoplay
    chunks = n_chunks                   # sound chunks the block logged before this episode
    while not (terminated or truncated) and ep_frame + skipped < max_frames:
        if autoplay_action is not None:
            # The env is in a state it takes no action in, so the policy is not
            # asked for one, exactly as the subject is not asked to press. The
            # frames still cost budget: they are engine steps, and the subject
            # pays for them in block time.
            action = autoplay_action
        else:
            # The subject's block ends on a wall clock; the model's ends on this
            # frame budget, so the countdown it reads is over its own budget at the
            # block's fps. The alternative, a real clock, would tell the model how
            # long its own inference took, which is not something the game shows.
            remaining_s = (max_frames - ep_frame - skipped) / fps
            hud_lines = list(adapter.hud(score, remaining_s) or [])
            overlay = adapter.overlay()
            action = policy.act(adapter.render(),
                                hud_lines + list(overlay[0] if overlay else []))
            if action is None:
                # A turn-based phase with no key pressed: nothing steps, exactly as
                # in the run loop. It costs the frame, which is what keeps a policy
                # that never names a key from looping here forever.
                skipped += 1
                continue

        observation, reward, terminated, truncated, info = adapter.step(action)
        autoplay_action = adapter.autoplay(info) if turn_based else None
        score += float(reward)
        # A savestate at the episode's first frame (the replay anchor), then every stride.
        anchor_frame = ep_frame % state_stride == 0
        frame_state = adapter.capture(observation, info, want_blob=anchor_frame)
        frame_fields = {
            "episode_id": episode_id, "ep_frame": ep_frame, "action": action, "reward": reward,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "run_time": clock.run_time(),
            # A policy has no screen, so no frame of this block was ever flipped.
            # NaN, not the step time: the human column is a measured photon onset,
            # and nothing should be able to average the two together by accident.
            "flip_time": float("nan"), "wall_time": clock.wall_time(),
            "variables": frame_state.variables,
        }
        if isinstance(info, dict) and "env_action" in info:
            frame_fields["env_action"] = info["env_action"]
        # The engine's sound for this frame goes to the log as a subject's does
        # (audio.h5, the chunk's number in audio_chunk); nobody hears it here.
        sound = adapter.sound()
        frame_fields["audio_chunk"] = -1
        if sound is not None and len(sound.pcm):
            frame_fields["audio_chunk"] = chunks
            chunks += 1
        logger.log_frame(frame_fields, frame_rgb=adapter.render(), savestate=frame_state.blob)
        logger.log_audio(sound if frame_fields["audio_chunk"] >= 0 else None)
        ep_frame += 1

    # Same vocabulary the human blocks log, from the same method: "playing" is
    # the budget cutting the episode off, where a subject's block clock would.
    outcome, _ = adapter.outcome(terminated, truncated)
    episode_result = EpisodeResult(episode_id, outcome, bool(terminated), bool(truncated),
                                   score, 0)
    logger.log(type="episode_end", **episode_result.record())
    return episode_result, skipped, chunks


def play_block(phase: dict, phase_index: int, args, logger: Logger,
               clock: Clock) -> str:
    """Play every episode of one game block and write its log.

    :param phase: the game-phase config, used exactly as a run uses it.
    :param phase_index: the phase's index in the curriculum (names the block's folder).
    :param args: parsed command-line arguments.
    :param logger: the run's logger writing the block.
    :param clock: the run's clock.
    :return: the block's folder name.
    """
    game_phase = GamePhase.from_dict(phase, phase_index)
    # A policy has no ears. Whatever the block's curriculum plays to a subject
    # as a sound, it reads as a line of text, so the two players are told the
    # same things; scanner curricula keep that line off the screen to hold the
    # gaze on the frame. Adapters without cues ignore the key.
    adapter = get_adapter(game_phase.backend, {**phase, "cue_overlay": True})
    policy = build_policy(args, adapter)

    block_folder_name = logger.open_block(phase_index, game_phase.backend, game_phase.game,
                                          phase, game_phase.seed)
    episodes: list[EpisodeResult] = []
    skipped = chunks = 0
    for episode_id in range(args.n_episodes):
        n_frames_before = logger.n_frames
        episode_result, n_skipped, chunks = play_episode(
            adapter, policy, logger, clock, seed=game_phase.seed + episode_id,
            episode_id=episode_id, state_stride=game_phase.state_stride,
            fps=game_phase.fps, turn_based=game_phase.is_turn_based,
            max_frames=args.max_frames, n_chunks=chunks)
        episodes.append(episode_result)
        skipped += n_skipped
        print(f"  episode {episode_id} (seed {game_phase.seed + episode_id}): "
              f"{logger.n_frames - n_frames_before} frames, {episode_result.outcome}, "
              f"score {episode_result.score:g}")

    block_extra = adapter.block_extra()
    adapter.close()
    summary = {
        "n_episodes": len(episodes), "n_frames": logger.n_frames,
        "outcomes": dict(Counter(episode.outcome for episode in episodes)),
        "total_reward": sum(episode.score for episode in episodes),
        # Which policy produced the block belongs in the block, not in a folder
        # name: a model's log and a human's are otherwise alike by design.
        "policy": args.policy, "policy_model": args.model if args.policy == "vlm" else "",
        # Three ways a model wastes a frame, kept apart: it named no key, it
        # named something that is not a key, or the call never came back.
        "skipped_frames": skipped, "invalid_replies": policy.invalid,
        "dropped_calls": policy.dropped,
    }
    logger.close_block(**summary, extra=block_extra)
    logger.log_phase({"index": phase_index, "type": "game", "backend": game_phase.backend,
                      "game": game_phase.game, **summary, "data_dir": block_folder_name})
    return block_folder_name


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

    curriculum = load_curriculum(args.curriculum)
    # The same check fmri_play runs, so one curriculum file suits both players.
    problems = validate_curriculum(curriculum)
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))
    phases = curriculum["curriculum"]
    run_folder = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")
    clock = Clock()
    # No scanner here, so the run's zero is simply when it started: the time
    # columns stay the same shape as a human run's, measured from that.
    clock.trigger()
    logger = Logger(run_folder, args.subject, phases, clock)
    logger.set_trigger_time()

    for phase_index, phase in enumerate(phases):
        if phase.get("type") != "game":
            continue
        print(f"block {phase_index}: {phase.get('game')} via {args.policy}")
        print(" ->", play_block(phase, phase_index, args, logger, clock))
    print("Manifest:", logger.save_manifest())
    logger.close()


if __name__ == "__main__":
    main()
