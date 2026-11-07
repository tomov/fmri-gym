"""Show a VLM a logged block's frames one at a time and print its notes; nothing plays the game.

Usage:
    python agents/agent_replay.py --block data/pilot/.../block-00_aigamestore_game4 \
        --backend transformers --model Qwen/Qwen3.5-35B-A3B --every 8 --max-steps 57
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from fmri_gym.logging import read_events, read_frame, read_frames
from vlm.conversation import Conversation
from vlm.observation import frame_content

SYSTEM = (
    "You are watching a video game being played; you do not control it.\n\n"
    "Every turn, reply with the JSON this turn's message asks for, and nothing else. "
    "The rationale you write is the only memory you keep between turns: past frames "
    "are not shown again, only what you wrote about them. A sentence or two is enough "
    "-- note what's happening and what you expect next, not a running essay.")

REPLY_INSTRUCTION = 'Reply with exactly this JSON: {"rationale": "<brief note>"}'


def watch(block: str, backend: Any, token_budget: int | None = 8000, every: int = 1,
         max_steps: int | None = None) -> None:
    """Send a block's frames to ``backend`` one per turn, keeping its notes as a scratchpad.

    :param block: a logged block's folder (``frames.h5`` + ``events.jsonl``).
    :param backend: a backend that can ``generate`` and read images.
    :param token_budget: the scratchpad's cap in estimated tokens; ``None`` or ``0`` for none.
    :param every: use every this-th recorded frame.
    :param max_steps: stop after this many frames; ``None`` for the whole block.
    """
    episodes = {e["frame"]: e["episode_id"] for e in read_events(block) if e["type"] == "frame"}
    frame_ids = [int(i) for i in read_frames(block)[1]][::every]
    if max_steps is not None:
        frame_ids = frame_ids[:max_steps]
    conversation = Conversation(SYSTEM, token_budget=token_budget)
    seen_episode = None
    for frame_id in frame_ids:
        episode = episodes.get(frame_id)
        boundary = None
        if conversation.has_turns and episode != seen_episode:
            boundary = "(the previous episode just ended; this is a new one)"
        seen_episode = episode
        content = frame_content(read_frame(block, frame_id), [], state="",
                                reply_instruction=REPLY_INSTRUCTION)
        if boundary:
            content = [{"type": "text", "text": boundary}, *content]
        reply = backend.generate(conversation.chat(content))
        rationale = _parse_rationale(reply.text)
        conversation.add(content, json.dumps({"rationale": rationale}))
        if not rationale:
            print(f"frame {frame_id} (episode {episode}): reply did not parse "
                  f"({reply.output_tokens} tokens out; cut off by the token cap?)")
            continue
        print(f"frame {frame_id} (episode {episode}): {rationale}")


def _parse_rationale(text: str) -> str:
    """The reply's ``rationale``, or ``""`` if it holds no JSON object.

    A reply cut off mid-thought is not kept as text: it would fill the scratchpad.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        return ""
    try:
        reply = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return ""
    if not isinstance(reply, dict):
        return ""
    return str(reply.get("rationale", ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", required=True, help="a logged block (frames.h5 + events.jsonl)")
    p.add_argument("--backend", required=True, choices=["transformers", "openrouter"])
    p.add_argument("--model", required=True)
    p.add_argument("--every", type=int, default=1, help="narrate every this-th frame")
    p.add_argument("--max-steps", type=int, help="stop after this many narrated frames")
    p.add_argument("--token-budget", type=int, default=8000,
                   help="the scratchpad's size cap, in estimated tokens; 0 keeps it all")
    args = p.parse_args()
    from vlm.backends import get_backend
    backend = get_backend(args.backend, args.model)
    watch(args.block, backend, token_budget=args.token_budget, every=args.every,
         max_steps=args.max_steps)


if __name__ == "__main__":
    main()
