"""Read a transcript through a language model and save its internal state at every turn.

The second half of the pipeline: :mod:`agents.lrm.transcript` turned a
subject's block into turns; this reads them, one episode at a time and in
overlapping windows (:func:`~agents.lrm.prompting.windows`), and writes the
model's hidden state at each turn to ``states.h5``
(:mod:`agents.lrm.internal_state`). It needs the model, not the game.

Usage::

    python -m agents.lrm.encode --transcript <block>/lrm/transcript_stride1.jsonl \\
        --backend transformers --model Qwen/Qwen3.5-9B --layers all --window 64
    torchrun --nproc-per-node 8 -m agents.lrm.encode --transcript ... \\
        --backend deepseek_v32 --model /ckpt/DeepSeek-V3.2-mp8 --option max_seq_len=32768

A backend's own settings go in as ``--option key=value`` (the value read as
JSON when it parses, else as a string). Under ``torchrun`` every rank reads
every window, and rank 0 writes the file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import groupby

from .backends import BACKENDS, LRMBackend, get_backend
from .internal_state import StateConfig, StateWriter
from .prompting import messages, system_prompt, windows
from .transcript import Turn, read_transcript


def encode_episode(backend: LRMBackend, writer: StateWriter | None, system: str,
                   turns: list[Turn], layers: list[int], window: int) -> None:
    """Read one episode's turns window by window, writing each window's reported states.

    :param backend: the model.
    :param writer: the open states file; ``None`` on a rank that does not write.
    :param system: the system prompt.
    :param turns: the episode's turns, in order.
    :param layers: the layers to keep.
    :param window: turns per window.
    """
    for w in windows(len(turns), window):
        chat = messages(system, turns[w.start:w.end])
        # Turn i of the window's assistant message is chat[2 + 2 * i].
        targets = [2 + 2 * (i - w.start) for i in range(w.first_target, w.end)]
        hidden = backend.encode(chat, targets, layers)
        if writer is not None:
            writer.append(turns[w.first_target:w.end], hidden, w.start)


def parse_layers(text: str, n_layers: int) -> list[int]:
    """``"all"``, or comma-separated indices where ``-1`` is the last block."""
    if text == "all":
        return list(range(n_layers))
    return sorted({int(x) % n_layers for x in text.split(",")})


def parse_options(pairs: list[str]) -> dict:
    """``["device_map=cpu", "max_seq_len=32768"]`` -> ``{"device_map": "cpu", ...}``.

    The values parse as JSON where they can, so numbers and booleans arrive typed.
    """
    options = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--option {pair!r}: write it as key=value")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def main() -> None:
    p = argparse.ArgumentParser(description="Save an LRM's internal state over a transcript.")
    p.add_argument("--transcript", required=True)
    p.add_argument("--backend", default="transformers", choices=BACKENDS)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                   help="the backend's own setting, e.g. device_map=cpu; repeatable")
    p.add_argument("--layers", default="all", help='"all", or indices like "0,16,-1"')
    p.add_argument("--window", type=int, default=64, help="turns per forward pass")
    p.add_argument("--out", help="default: beside the transcript, states_<transcript>_<model>.h5")
    args = p.parse_args()

    header, turns = read_transcript(args.transcript)
    backend = get_backend(args.backend, args.model, **parse_options(args.option))
    if not backend.can_encode:
        raise SystemExit(f"the {args.backend} backend returns text, not activations; "
                         "read internal states with transformers, deepseek_v32 or deepseek_v4")
    layers = parse_layers(args.layers, backend.n_layers)
    system = system_prompt(header)
    stem = os.path.splitext(os.path.basename(args.transcript))[0]
    # "Qwen/Qwen3.5-9B" -> "Qwen3.5-9B"; a local checkpoint's folder name.
    model_slug = os.path.basename(args.model.rstrip("/"))
    out = args.out or os.path.join(os.path.dirname(args.transcript),
                                   f"states_{stem}_{model_slug}.h5")
    config = StateConfig(backend.record(), tuple(layers), args.transcript, args.window, system)
    writer = StateWriter(out, config, header) if backend.is_writer else None
    try:
        for episode_id, episode in groupby(turns, key=lambda t: t.episode_id):
            began = time.time()
            episode = list(episode)
            encode_episode(backend, writer, system, episode, layers, args.window)
            if writer is not None:
                print(f"episode {episode_id}: {len(episode)} turns in {time.time() - began:.1f}s")
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        print(f"{len(turns)} turns x {len(layers)} layers -> {out}")


if __name__ == "__main__":
    main()
