"""A VLM's hidden states over a logged block: one turn per recorded frame, read from the log.

The block holds everything most readers need -- the frames (``frames.h5``), each frame's
action, reward and logged ``variables`` -- so most backends need no adapter and no replay.
A reader whose game costs real per-frame engine work for its state (:attr:`.symbolic.base
.SymbolicReader.wants_engine_state`; vizdoom and coom, for the actors and player variables
``capture()`` never logs) gets it from a fresh adapter replayed alongside the log instead,
read from the adapter's own game object each frame (:func:`_engine_state`). Each recorded
frame is one turn: the frame, the score, and
the game's state as a symbolic reader says it (:mod:`.symbolic`), all as they stood before
that frame's action; the action is the assistant's reply, and the state kept is the one at
its header token. The states go to ``<block>/vlm_states_<model>.h5`` (:class:`StateWriter`),
one row per turn, beside the turn's ``frame``, ``episode_id``, ``ep_frame`` and ``run_time``:
the scan's alignment keys.

Usage:
    python agents/vlm/extract_features.py --block data/<run>/block-01_crafter_L4 \
        --backend transformers --model Qwen/Qwen3.5-9B --layers 0 16 -1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from backends import get_backend
from observation import frame_content
from symbolic import get_reader

from fmri_gym.logging import read_events, read_frames, read_state

SYSTEM = "You are playing a video game, one key press per turn."
#: Per-turn columns of the states file, beside ``hidden``.
COLUMNS = {"frame": "int64", "episode_id": "int64", "ep_frame": "int64", "run_time": "float64"}


@dataclass(frozen=True)
class StateConfig:
    """How a states file's rows were made; its attributes."""

    block: str
    backend: dict
    layers: tuple[int, ...]
    system: str

    def record(self) -> dict:
        return {"block": self.block, "model": self.backend["model"],
                "backend": json.dumps(self.backend), "layers": json.dumps(list(self.layers)),
                "system_prompt": self.system}


class StateWriter:
    """Appends turns' states and their alignment columns to an HDF5 file, flushing each."""

    def __init__(self, path: str, config: StateConfig) -> None:
        self.file = h5py.File(path, "w")
        self.file.attrs.update(config.record())
        for name, dtype in COLUMNS.items():
            self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
        self.file.create_dataset("action", shape=(0,), maxshape=(None,),
                                 dtype=h5py.string_dtype())

    def append(self, event: dict, action: str, hidden: np.ndarray) -> None:
        """Add one turn: its frame record's columns, its action name and ``(layers, width)``."""
        if "hidden" not in self.file:
            self.file.create_dataset("hidden", shape=(0, *hidden.shape),
                                     maxshape=(None, *hidden.shape), dtype="float16",
                                     chunks=(1, *hidden.shape))
        n = self.file["hidden"].shape[0]
        row = {**{k: event[k] for k in COLUMNS}, "action": action, "hidden": hidden}
        for name, value in row.items():
            self.file[name].resize(n + 1, axis=0)
            self.file[name][n] = value
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def extract(block: str, backend_name: str, model: str, layers: list[int] | None = None,
            **backend_options) -> str:
    """Write the backend's state at every recorded frame of ``block``.

    :param block: the block's folder.
    :param backend_name: an :mod:`.backends` name that can encode and see images.
    :param model: the backend's model id.
    :param layers: which layers, negative from the last; ``None`` for all.
    :param backend_options: the backend's own constructor arguments.
    :return: the states file written.
    """
    events = read_events(block)
    start = events[0]
    images, frame_index = read_frames(block)
    image_at = {int(f): pos for pos, f in enumerate(frame_index)}
    backend = get_backend(backend_name, model, **backend_options)
    if not (backend.can_encode and backend.can_see_images):
        raise ValueError(f"the {backend_name} backend ({model}) cannot both encode and see "
                         "images, which reading a block's frames needs")
    n = backend.n_layers
    layers = list(range(n)) if layers is None else sorted({layer % n for layer in layers})
    keys = start["phase"]["keys"]
    names = _name_for_action(keys, (e["action"] for e in events if e["type"] == "frame"))
    system = SYSTEM + "\nKeys you may press: " + ", ".join(sorted(keys))
    slug = os.path.basename(model.rstrip("/"))
    out = os.path.join(block, f"vlm_states_{slug}.h5")
    writer = StateWriter(out, StateConfig(block, backend.record(), tuple(layers), system))
    try:
        _read_block(events, start["backend"], start["phase"], images, image_at, names, system,
                    backend, layers, writer)
    finally:
        writer.close()
    return out


def _read_block(events: list[dict], game: str, phase: dict, images: Any, image_at: dict,
                names: dict, system: str, backend: Any, layers: list[int],
                writer: StateWriter) -> None:
    """One turn per recorded frame, each seeing the state before its own action."""
    reader, score = get_reader(game), 0.0
    adapter = _replay_adapter(game, phase) if reader.wants_engine_state else None
    try:
        for e in events:
            if e["type"] == "episode_start":
                reader.reset()
                score = 0.0
                if adapter is not None:
                    adapter.reset(e["seed"])
                    if e.get("initial_state") is not None:
                        adapter.restore(read_state(e["initial_state"]))
            if e["type"] != "frame":
                continue
            variables = e.get("variables", {})
            if adapter is not None:
                adapter.step(e["action"])
                variables = {**variables, **_engine_state(game, adapter)}
            if e["frame"] in image_at:
                content = frame_content(images[image_at[e["frame"]]], [f"Score: {score:g}"],
                                        reader.observation().text())
                action = names.get(_action_key(e["action"]), "?")
                chat = [{"role": "system", "content": system},
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": action}]
                writer.append(e, action, backend.encode(chat, targets=[2], layers=layers)[0])
            reader.see(variables)
            score += float(e["reward"])
    finally:
        if adapter is not None:
            adapter.close()


def _replay_adapter(backend: str, phase: dict) -> Any:
    """A fresh adapter to drive alongside the log, for a reader needing engine state.

    Vizdoom's ``objects``/``labels`` buffers (what :func:`_engine_state` reads) are set
    only at construction, and no curriculum turns them on since nothing on the live path
    needs them, so this is the one adapter built with them forced on; coom needs no
    override -- its own env forces them on regardless.

    :param backend: the block's ``backend`` (an adapter's ``name``).
    :param phase: the block's ``phase``, as logged on ``block_start``.
    :return: the adapter, reset by the caller before its first ``step``.
    """
    from fmri_gym.adapters import get_adapter
    env_kwargs = phase.get("env_kwargs", {})
    if backend == "vizdoom":
        from symbolic.doom import ENGINE_OUTPUTS
        env_kwargs = {**env_kwargs, **ENGINE_OUTPUTS}
    return get_adapter(backend, {**phase, "env_kwargs": env_kwargs})


def _engine_state(backend: str, adapter: Any) -> dict:
    """This frame's engine-only state: the player's variables, every actor, and labels.

    Not in the log -- no adapter's ``capture()`` logs it, so this reads it from the
    engine underneath a fresh adapter stepped alongside the log (see
    :func:`_replay_adapter`), reaching into the adapter instance the way a game's own
    engine is otherwise private to it.

    :param backend: the block's ``backend`` (an adapter's ``name``).
    :param adapter: the fresh adapter, already stepped for this frame.
    :return: ``{"player_variables": ..., "objects": ..., "labels": ...}``.
    """
    from symbolic.doom import labels, objects, player_variables
    game = adapter.env.unwrapped.game if backend == "vizdoom" else adapter._game
    state = game.get_state()
    return {"player_variables": player_variables(game),
            "objects": objects(state), "labels": labels(state)}


def _name_for_action(keys: dict, logged_actions: Any) -> dict:
    """Every logged action's name, from ``keys`` (name -> curriculum value).

    A MultiBinary action is the vector of buttons held, so it is named by its buttons'
    keys joined with ``+`` (``"LEFT+SPACE"``; ``"NOOP"`` for none), each button by the
    first key the phase gives it.
    """
    logged = list(logged_actions)
    if not any(isinstance(a, list) for a in logged):
        return {_action_key(value): name for name, value in keys.items()}
    button: dict[int, str] = {}
    for name, index in keys.items():
        button.setdefault(index, name)
    return {_action_key(a): "+".join(button.get(i, f"button{i}") for i, held in enumerate(a)
                                     if held) or "NOOP" for a in logged}


def _action_key(action: Any) -> Any:
    """A hashable form of a logged action (a MultiBinary one is a list)."""
    return tuple(action) if isinstance(action, list) else action


def _parse_options(options: list[str]) -> dict:
    """``["device_map=cpu"]`` -> ``{"device_map": "cpu"}``, values read as JSON where they parse."""
    parsed = {}
    for option in options:
        key, _, value = option.partition("=")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def main() -> None:
    p = argparse.ArgumentParser(description="A VLM's hidden states over a logged block.")
    p.add_argument("--block", required=True)
    p.add_argument("--backend", required=True, help="an agents.vlm.backends name")
    p.add_argument("--model", required=True)
    p.add_argument("--layers", type=int, nargs="+", help="omit for every layer")
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                   help="a backend constructor argument (e.g. device_map=cpu); repeatable")
    args = p.parse_args()
    print("States:", extract(args.block, args.backend, args.model, args.layers,
                             **_parse_options(args.option)))


if __name__ == "__main__":
    main()
