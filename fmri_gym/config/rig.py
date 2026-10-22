"""The rig config: how one rig plays every run, whatever its curriculum.

A **rig** is one setup a run is played on -- a monitor, a scanner, a response
device, a trigger line -- and a site can have several. Each has its own rig
file, kept in the repo as ``configs/rigs/<site>/<rig>.json``
(``configs/rigs/README.md``), so every site's rigs are reviewed with the code. A
curriculum says what is played and plays the same on every rig; the rig file
says how this rig plays it, and every run played on the rig reads it.

``fmri-play --rig <file>`` names the rig file, as ``--curriculum`` names the
curriculum's (:meth:`RigConfig.open`); it is required, since which window,
controller and data folder a session gets is not something to guess.

Every key of a rig file is optional: an empty ``{}`` plays on the defaults. The
rig check files its results under ``site`` and ``rig``, the file's folder and
name; written in the file they must say the same. It may describe the rig for
people -- :data:`FIELDS`, notes
nothing plays from, filed with every run and pooled by the rig check -- and
say how its runs are played (:data:`DEFAULTS` when left out), under the field
names of :class:`RigConfig` and :class:`ScreenConfig`, so the file and the code
say the same thing:

- ``"screen"``: ``{"size": "1024x768", "is_fullscreen": false, "monitor": 0,
  "has_vsync": true}``, the monitor by index (0: the first); fullscreen takes
  the monitor's resolution;
- ``"has_pad"``: read a plugged-in game controller (:mod:`fmri_gym.pad`);
- ``"bindings"``: which keyboard key each of the controller's controls presses
  (:data:`fmri_gym.pad.CONTROLS` -> a key name, as curricula write keys),
  ``{"A": "SPACE", "B": "Z", "UP": "UP", ...}``: one map for every game played
  on the rig, so a curriculum's keys stay keyboard keys. A control left out
  presses nothing; with none at all a plugged-in controller is warned about
  (a controller that types keys itself needs none);
- ``"has_audio"``: play game audio; ``false`` mutes every block;
- ``"data_root"``: where the BIDS tree goes, ``<root>/sub-XX/ses-NNN/beh/<run>/``;
- ``"triggers"``: the scanner's start sync and the trigger codes, over which
  line (:class:`~fmri_gym.triggers.TriggerSettings`); left out, the run waits
  for ``=`` and sends no codes, and says so on the experimenter screen and in
  the manifest (``defaulted``).

It is written by hand or in the rig form (``python -m fmri_gym.checks rig --rig
<file>``), which a rig check also opens when its rig has no valid file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..keys import KEY_NAMES
from ..pad import CONTROLS
from ..triggers import TriggerError, TriggerSettings

MODALITIES = ("fmri", "meg", "eeg", "behav")
#: What the rig is, for people: notes nothing plays from, each optional, filed with
#: every run and pooled by the rig check (``site`` and ``rig`` are the file's place).
FIELDS = {
    "site": "the site, and the rig file's folder: letters, digits, _ and -",
    "rig": "this rig at the site, and the rig file's name: letters, digits, _ and -",
    "pi": "the PI's initials, in capitals (Momchil Tomov: MT)",
    "modality": f"one of {', '.join(MODALITIES)}",
    "monitor": "monitor or projector, make and model, and its settings that matter",
    "photodiode": "photodiode make/model and where it is plugged in",
    "audio_path": "from the sound card to the ear: amplifier, tubes, headphones",
    "trigger_hardware": "trigger box / port / scanner interface",
    "notes": "anything else (may be empty)",
}
#: What each of :data:`FIELDS` looks like, for the form's placeholders.
EXAMPLES = {
    "site": "mri_3T", "rig": "stimpc1", "pi": "MT", "modality": "fmri",
    "monitor": "BOLDscreen 32 LCD, 120 Hz, 1920x1080",
    "photodiode": "BioSemi photodiode on sound card input 0",
    "audio_path": "RME Babyface -> Sensimetrics S14 insert earphones",
    "trigger_hardware": "fORP 932 USB, '=' at each volume",
}
#: How a run is played where the rig file says nothing: a window at the desk,
#: the pad and the audio on.
DEFAULTS: dict[str, Any] = {
    "screen": {"size": "1024x768", "is_fullscreen": False, "monitor": 0, "has_vsync": True},
    "has_pad": True, "has_audio": True, "data_root": "data", "triggers": {}, "bindings": {},
}
#: Trigger presets the editor and the rig check offer. Each names ``sync.mode``
#: and ``backend`` outright: a section that leaves them out runs, but is
#: reported as NOT SET.
TRIGGER_PRESETS: dict[str, dict] = {
    "fMRI (wait for '=', no codes)": {
        "sync": {"mode": "wait", "key": "="}, "backend": "null"},
    # MEG and EEG alike: the stimulus PC starts the recording and marks it, over serial.
    "MEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    # A Mac has no parallel port: a USB DAQ's digital port wired to the trigger input instead.
    "MEG (start from the trigger line, USB-1208FS codes over HID)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "hid", "hid_device": "mcc-1208fs",
        "port": "A"},
    "EEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    "Behavioural (no scanner)": {
        "sync": {"mode": "none"}, "backend": "null"},
}
#: Site and rig: values in the TSV, never in a file name, so BIDS's letters-and-digits
#: rule does not bind them; still no spaces, as rows are grouped by them when pooled.
_LABEL = re.compile(r"[A-Za-z0-9_-]+")
_INITIALS = re.compile(r"[A-Z]{2,4}")
_SIZE = re.compile(r"([1-9]\d*)x([1-9]\d*)")
_MAKE = ("make it with the rig's long rig check, which opens a form for it: fmri-play "
         "--subject sub-rig --ses 1 --run 1 --rig <file> --curriculum "
         "configs/curricula/checks/rig-check-long.json (python -m fmri_gym.checks rig --rig "
         "<file> opens the form alone), or start from configs/rigs/example/empty.json")
#: The rig files kept in the repo, which the editor offers.
REPO_RIGS = Path(__file__).resolve().parents[2] / "configs" / "rigs"


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    """The window a rig's runs are shown in: a rig file's ``"screen"``."""

    #: the window, ``(width, height)`` in pixels, when not fullscreen.
    size: tuple[int, int]
    is_fullscreen: bool
    #: which monitor the window opens on, by index.
    monitor: int
    #: lock flips to the monitor refresh.
    has_vsync: bool

    @classmethod
    def from_dict(cls, screen: dict) -> ScreenConfig:
        """Read a checked ``"screen"`` (:func:`problems`), ``DEFAULTS["screen"]`` where it
        says nothing; ``size`` is written ``<width>x<height>``."""
        screen = {**DEFAULTS["screen"], **screen}
        width, height = (int(n) for n in _SIZE.fullmatch(screen["size"]).groups())
        return cls(size=(width, height), is_fullscreen=screen["is_fullscreen"],
                   monitor=screen["monitor"], has_vsync=screen["has_vsync"])

    def record(self) -> dict[str, Any]:
        """As a rig file writes it."""
        return {"size": f"{self.size[0]}x{self.size[1]}", "is_fullscreen": self.is_fullscreen,
                "monitor": self.monitor, "has_vsync": self.has_vsync}


@dataclass(frozen=True, slots=True)
class RigConfig:
    """One rig file, checked: what the rig is, and how every run on it is played."""

    #: the rig file, as ``--rig`` named it.
    path: str
    #: what the rig is, :data:`FIELDS` as written.
    fields: dict[str, str]
    #: the window.
    screen: ScreenConfig
    #: read a plugged-in game controller as key presses.
    has_pad: bool
    #: the controller's control -> the key it presses
    bindings: dict[str, str]
    #: play game audio; ``False`` mutes every block and opens no output.
    has_audio: bool
    #: the BIDS tree's root.
    data_root: str
    #: the scanner's start sync and the trigger codes, over which line.
    triggers: TriggerSettings

    @classmethod
    def from_dict(cls, rig: dict, path: str) -> RigConfig:
        """Read a rig file's JSON, :data:`DEFAULTS` where it says nothing.

        :param rig: the file's JSON.
        :param path: the file; its folder and name are the rig's ``site`` and ``rig``,
            which a ``site`` or ``rig`` it writes must match.
        :raises ValueError: naming each field that is unknown or invalid, and how to fix it.
        """
        found = problems(rig)
        site, rig_name = Path(path).parent.name, Path(path).stem
        found += [f'"{k}" is {rig[k]!r}, but the file is {path}: its folder is the site, '
                  f"its name the rig" for k, own in (("site", site), ("rig", rig_name))
                  if rig.get(k, own) != own]
        if found:
            raise ValueError(f"{path}: " + "; ".join(found) + ". Fix it by hand, or in the "
                             f"form: python -m fmri_gym.checks rig --rig {path}")
        settings = {**DEFAULTS, **rig}
        fields = {"site": site, "rig": rig_name,
                  **{k: rig[k] for k in FIELDS if k in rig and k not in ("site", "rig")}}
        return cls(path=path, fields=fields,
                   screen=ScreenConfig.from_dict(rig.get("screen", {})),
                   has_pad=settings["has_pad"], bindings=dict(settings["bindings"]),
                   has_audio=settings["has_audio"],
                   data_root=settings["data_root"],
                   triggers=TriggerSettings.from_dict(settings["triggers"]))

    @classmethod
    def open(cls, path: str) -> RigConfig:
        """The rig file at ``path``, checked.

        :raises ValueError: a file that is missing, not a JSON object, or that
            :meth:`from_dict` refuses; the message says how to make one.
        """
        if not os.path.isfile(path):
            raise ValueError(f"--rig {path}: no such file; {_MAKE}")
        return cls.from_dict(load(path), path)

    @property
    def has_trigger_setup(self) -> bool:
        """Whether the file says how the run starts or where codes go (not both defaulted)."""
        return set(self.triggers.defaulted) != {"sync.mode", "backend"}

    def without_trigger_line(self) -> RigConfig:
        """This rig with no trigger line (``null`` backend): what a rig check plays on when
        the line does not open, so its other checks still run. ``send`` becomes ``none``,
        since it needs a line; the file is untouched."""
        sync = self.triggers.sync
        if sync.mode == "send":
            sync = replace(sync, mode="none")
        return replace(self, triggers=replace(self.triggers, backend="null", sync=sync))

    def record(self) -> dict[str, Any]:
        """The rig file as the run played it, every default written out: the manifest's."""
        return {"path": self.path, **self.fields, "screen": self.screen.record(), "has_pad": self.has_pad,
                "bindings": self.bindings, "has_audio": self.has_audio, "data_root": self.data_root,
                "triggers": asdict(self.triggers)}

    def status(self) -> str:
        """One line for the console: which file, and what it settles."""
        screen = self.screen
        window = ("fullscreen" if screen.is_fullscreen
                  else f"{screen.size[0]}x{screen.size[1]} window")
        return (f"{self.path} | {window} on monitor {screen.monitor}, vsync "
                f"{'on' if screen.has_vsync else 'off'} | pad {'on' if self.has_pad else 'off'}"
                f" | audio {'on' if self.has_audio else 'off'} | data in {self.data_root}"
                f" | triggers {self.triggers.sync.mode}, codes over {self.triggers.backend}")


def rig_files() -> list[str]:
    """The rig files kept in the repo (``configs/rigs/<site>/<rig>.json``), as paths
    from the working directory where they can be, for the editor to offer."""
    files = sorted(REPO_RIGS.glob("*/*.json"))
    return [os.path.relpath(f) if not os.path.relpath(f).startswith("..") else str(f)
            for f in files]


def load(path: str) -> dict:
    """The rig file at ``path`` as written, unchecked (the form edits an invalid one).

    :raises ValueError: a file that is not a JSON object.
    """
    where = path
    try:
        with open(where) as f:
            rig = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: not JSON ({exc})") from None
    if not isinstance(rig, dict):
        raise ValueError(f"{where}: expected a JSON object")
    return rig


def draft(path: str) -> dict:
    """What the rig form starts from for the rig file at ``path``: the file as written;
    nothing when it does not exist yet or is not JSON."""
    try:
        return load(path) if os.path.isfile(path) else {}
    except ValueError:
        return {}


def save(rig: dict, path: str) -> None:
    """Write ``rig`` as the rig file at ``path``, its folder made if need be."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rig, f, indent=2)
        f.write("\n")


def problems(rig: dict) -> list[str]:
    """What keeps ``rig`` from being a rig file: :data:`FIELDS` and :data:`DEFAULTS` of
    the right type where given, no other key.

    :return: one line per problem; empty when it is valid.
    """
    out = [f"unknown key {k!r}" for k in rig if k not in FIELDS and k not in DEFAULTS]
    fields = {k: v for k, v in rig.items() if k in FIELDS}
    out += [f"{k!r} must be a string" for k, v in fields.items() if not isinstance(v, str)]
    out += [f"{k!r} is empty" for k, v in fields.items() if k != "notes" and v == ""]
    for k in ("site", "rig"):
        if isinstance(rig.get(k), str) and rig[k] and not _LABEL.fullmatch(rig[k]):
            out.append(f"{k!r} must be letters, digits, _ or -, got {rig[k]!r}")
    if isinstance(rig.get("pi"), str) and rig["pi"] and not _INITIALS.fullmatch(rig["pi"]):
        out.append(f"'pi' must be 2 to 4 capital letters (initials), got {rig['pi']!r}")
    if rig.get("modality") not in (*MODALITIES, "", None):
        out.append(f"'modality' must be one of {MODALITIES}, got {rig['modality']!r}")
    return out + _setting_problems(rig)


def _setting_problems(rig: dict) -> list[str]:
    """What is wrong with the file's ``screen``, ``has_pad``, ``has_audio`` and ``data_root``."""
    screen = rig.get("screen", {})
    if not isinstance(screen, dict):
        return [f'"screen": expected {{"size", "is_fullscreen", "monitor", "has_vsync"}}, '
                f"got {screen!r}"]
    out = [f'"screen": unknown key {k!r}' for k in screen if k not in DEFAULTS["screen"]]
    size = screen.get("size", DEFAULTS["screen"]["size"])
    if not (isinstance(size, str) and _SIZE.fullmatch(size)):
        out.append(f'"screen": "size" must be <width>x<height> such as 1024x768, got {size!r}')
    monitor = screen.get("monitor", DEFAULTS["screen"]["monitor"])
    if not isinstance(monitor, int) or isinstance(monitor, bool) or monitor < 0:
        out.append(f"\"screen\": \"monitor\" must be a monitor's index (0: the first), "
                   f"got {monitor!r}")
    flags = (("screen.is_fullscreen", screen.get("is_fullscreen", False)),
             ("screen.has_vsync", screen.get("has_vsync", True)),
             ("has_pad", rig.get("has_pad", True)), ("has_audio", rig.get("has_audio", True)))
    out += [f'"{k}" must be true or false, got {v!r}' for k, v in flags if not isinstance(v, bool)]
    root = rig.get("data_root", DEFAULTS["data_root"])
    if not isinstance(root, str) or not root:
        out.append(f'"data_root" must be a folder, got {root!r}')
    return out + trigger_problems(rig.get("triggers", {})) + _binding_problems(rig)


def _binding_problems(rig: dict) -> list[str]:
    """``bindings``: the controller's controls, each to a key name a curriculum can bind."""
    bindings = rig.get("bindings", {})
    if not isinstance(bindings, dict):
        return [f'"bindings": expected {{control: key}}, got {bindings!r}']
    return ([f'"bindings": {c!r} is not a control of the controller ({", ".join(CONTROLS)})'
             for c in bindings if c not in CONTROLS]
            + [f'"bindings": {c!r}: {k!r} is not a key name (as in fmri_gym/keys.py)'
               for c, k in bindings.items() if k not in KEY_NAMES])


def trigger_problems(section: Any) -> list[str]:
    """What :class:`~fmri_gym.triggers.TriggerSettings` refuses in a ``triggers`` section, as text.

    The port is the one check added here: the settings accept a missing port
    and the backend refuses it on opening, which a check before the run never does.

    :return: at most one problem (the first the settings raise).
    """
    if not isinstance(section, dict):
        return [f'"triggers": expected an object, got {section!r}']
    try:
        s = TriggerSettings.from_dict(section)
    except (TypeError, ValueError, TriggerError) as exc:
        return [str(exc)]
    if s.backend in ("serial", "parallel", "hid") and not s.port:
        return [f"triggers: backend {s.backend!r} needs a port"]
    return []
