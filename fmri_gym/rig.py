"""The rig files, and the rig keys: the controller's own buttons.

A **rig** is one setup a run is played on -- a monitor, a scanner, a response
device, a trigger line -- and a site can have several. Each has its own file,
outside the checkout: ``~/.config/fmri-gym/rigs/<name>.json``
(``$XDG_CONFIG_HOME`` moves it). It holds everything the run configs share on
that rig, so a run config says only what is played and plays the same on
every rig. ``fmri-play --rig <name>`` picks one (:func:`choose`): a machine
with one rig file needs no ``--rig``; with none, or several and no ``--rig``,
nothing starts.

A rig file says what the rig is -- the fields a rig check files with its
results (:data:`fmri_gym.checks.RIG_FIELDS`), its ``"rig"`` being the file's
name -- and, each optional (:data:`SETTINGS` when left out):

- ``"screen"``: the window, ``{"size": "1024x768", "fullscreen": false,
  "monitor": 0, "vsync": true}`` (``monitor`` by index, 0 the first);
- ``"pad"``: read a plugged-in game controller (:mod:`fmri_gym.pad`);
- ``"audio"``: play game audio (``false`` mutes every block, and the manifest's
  curriculum shows ``"audio": false``);
- ``"data_root"``: where the BIDS tree goes, ``<root>/sub-XX/ses-NNN/beh/<run>/``;
- ``"keys"``: typed key -> rig key, for a device that types (``{"1": "LEFT"}``
  on a button box), over :data:`TYPED_KEYS`; a typed key is named as pygame
  names it (``"1"``, ``"space"``, ``"left shift"``, ``"[1]"`` for keypad 1);
- ``"controls"``: the rig keys that device has, which a rig check asks for one
  by one (all of them when it is left out: the controller has them all).

Every game phase maps the **rig keys** -- the buttons of the Current Designs
controller, :data:`CONTROLS` -- to its actions. The controller presses them
under their own names; a device that types keys instead (the keyboard, a button
box) is read through :data:`TYPED_KEYS` and the rig file's ``keys``.

A rig file can be written by hand, or in the form a rig check opens when there
is none (``python -m fmri_gym.checks rig --rig <name>`` opens it alone).
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: The rig keys: the controller's stick (four directions) and its six buttons.
CONTROLS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "X", "Y", "LT", "RT")
#: The keyboard's rig keys, when the rig file says nothing of a key: the arrows,
#: the letters named like the buttons, and the shifts for the triggers. Typed keys
#: are named as pygame names them (``pygame.key.name``), in any case.
TYPED_KEYS: dict[str, str] = {
    "up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT",
    "a": "A", "b": "B", "x": "X", "y": "Y", "left shift": "LT", "right shift": "RT",
}
#: What a rig file leaves out: a window at the desk, the pad and the audio on.
SETTINGS: dict[str, Any] = {
    "screen": {"size": "1024x768", "fullscreen": False, "monitor": 0, "vsync": True},
    "pad": True, "audio": True, "data_root": "data",
}
#: The rig picked for this process (:func:`use`); ``None`` until one is.
_chosen: str | None = None


def folder() -> Path:
    """Where the rig files are: ``~/.config/fmri-gym/rigs/``."""
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "fmri-gym" / "rigs"


def path(name: str) -> Path:
    """The file of the rig ``name`` (it need not exist)."""
    return folder() / f"{name}.json"


def names() -> list[str]:
    """The rigs this machine has a file for, by name."""
    return sorted(p.stem for p in folder().glob("*.json"))


def choose(name: str | None) -> str:
    """The rig to play on: ``name``, or this machine's only one.

    :raises ValueError: when ``name`` has no file, or none is given and there
        is not exactly one; the message says how to pick or make one.
    """
    found = names()
    make = ("make one with the rig's long rig check, which opens a form for it the first time: "
            "fmri-play --curriculum configs/rig-check-long.json --subject sub-rig --ses 1 --run 1 "
            "--rig <name> (python -m fmri_gym.checks rig --rig <name> opens the form alone)")
    if name is not None:
        if name in found:
            return name
        have = f"this machine has {', '.join(found)}" if found else "this machine has none"
        raise ValueError(f"--rig {name}: no {path(name)} ({have}); {make}")
    if len(found) == 1:
        return found[0]
    if found:
        raise ValueError(f"{len(found)} rigs in {folder()} ({', '.join(found)}): "
                         "pick one with --rig <name>")
    # where the one rig file was kept before there was one per rig
    old = next((p for p in (folder().parent / "rig.json", Path("rig.json")) if p.exists()), None)
    if old:
        raise ValueError(f"no rig file in {folder()}: move {old} there as <name>.json, its "
                         '"rig" field as the name, or ' + make)
    raise ValueError(f"no rig file in {folder()}: {make}")


def load(name: str) -> dict:
    """The rig file of ``name`` as written.

    :raises ValueError: a file that is not a JSON object.
    """
    where = path(name)
    try:
        with where.open() as f:
            rig = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: not JSON ({exc})") from None
    if not isinstance(rig, dict):
        raise ValueError(f"{where}: expected a JSON object")
    return rig


def use(name: str | None) -> dict:
    """Pick the rig this process plays on (:func:`choose`), and return its file.

    :raises ValueError: see :func:`choose` and :func:`load`.
    """
    global _chosen
    _chosen = choose(name)
    return load(_chosen)


def read() -> dict | None:
    """The rig picked (:func:`use`) -- or, before one is, this machine's only rig --
    as written; ``None`` when there is none to read.

    :raises ValueError: a file that is not a JSON object.
    """
    name = _chosen
    if name is None:
        found = names()
        name = found[0] if len(found) == 1 else None
    return load(name) if name is not None else None


def current() -> str:
    """The rig :func:`read` reads, for messages."""
    try:
        return str(path(_chosen or choose(None)))
    except ValueError:
        return str(folder())


def settings(rig: dict | None) -> dict[str, Any]:
    """The rig's screen, pad, audio and data root, :data:`SETTINGS` where it says nothing."""
    rig = rig or {}
    return {**SETTINGS, **{k: rig[k] for k in SETTINGS if k in rig},
            "screen": {**SETTINGS["screen"], **(rig.get("screen") or {})}}


def setting_problems(rig: dict) -> list[str]:
    """What is wrong with the rig file's ``screen``, ``pad``, ``audio`` and ``data_root``.

    :return: one line per problem; empty when they are fine.
    """
    screen = rig.get("screen", {})
    if not isinstance(screen, dict):
        return [(f'"screen": expected {{"size", "fullscreen", "monitor", "vsync"}}, '
                 f"got {screen!r}")]
    out = [f'"screen": unknown key {k!r}' for k in screen if k not in SETTINGS["screen"]]
    size = screen.get("size", SETTINGS["screen"]["size"])
    if not (isinstance(size, str) and re.fullmatch(r"[1-9]\d*x[1-9]\d*", size)):
        out.append(f'"screen": "size" must be <width>x<height> such as 1024x768, got {size!r}')
    monitor = screen.get("monitor", 0)
    if not isinstance(monitor, int) or isinstance(monitor, bool) or monitor < 0:
        out.append(f'"screen": "monitor" must be a monitor\'s index (0: the first), '
                   f"got {monitor!r}")
    out += [f'"{where}" must be true or false, got {value!r}' for where, value in
            (("screen.fullscreen", screen.get("fullscreen", False)),
             ("screen.vsync", screen.get("vsync", True)),
             ("pad", rig.get("pad", True)), ("audio", rig.get("audio", True)))
            if not isinstance(value, bool)]
    root = rig.get("data_root", "data")
    if not isinstance(root, str) or not root:
        out.append(f'"data_root" must be a folder, got {root!r}')
    return out


def run_args(rig: dict) -> dict[str, Any]:
    """The rig's settings under the names ``fmri_play``'s run reads them by."""
    s = settings(rig)
    return {"size": s["screen"]["size"], "fullscreen": s["screen"]["fullscreen"],
            "monitor": s["screen"]["monitor"], "no_vsync": not s["screen"]["vsync"],
            "no_pad": not s["pad"], "no_audio": not s["audio"], "data_root": s["data_root"]}


def key_problems(rig: dict, is_key: Callable[[str], bool]) -> list[str]:
    """What is wrong with the rig file's ``keys`` and ``controls`` (both optional).

    :param rig: the rig file.
    :param is_key: whether a name is a key that can be typed (:func:`fmri_gym.keys.is_key`).
    :return: one line per problem; empty when they are fine.
    """
    out = []
    keys = rig.get("keys", {})
    if not isinstance(keys, dict):
        out.append(f'"keys": expected {{typed key: rig key}}, got {keys!r}')
    else:
        out += [f'"keys": {k!r} is not a key name (pygame\'s names: "1", "space", '
                f'"left shift", "[1]" for keypad 1, ...)' for k in keys if not is_key(k)]
        out += [f'"keys": {k!r}: {v!r} is not a rig key ({", ".join(CONTROLS)})'
                for k, v in keys.items() if v not in CONTROLS]
    controls = rig.get("controls", list(CONTROLS))
    if not isinstance(controls, list) or not controls:
        out.append(f'"controls": expected a list of rig keys, got {controls!r}')
    else:
        out += [f'"controls": {c!r} is not a rig key ({", ".join(CONTROLS)})'
                for c in controls if c not in CONTROLS]
    return out


def controls(rig: dict | None) -> list[str]:
    """The rig keys this rig's participant can press (all of them, unless the file says)."""
    return list((rig or {}).get("controls") or CONTROLS)

