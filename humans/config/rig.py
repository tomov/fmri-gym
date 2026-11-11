"""The rig config: how one rig plays every run, whatever its curriculum.

A **rig** is one setup a run is played on -- a monitor, a scanner, a response
device, a trigger line -- and a team can have several. Each has its own rig
file, kept in the repo as ``configs/<team>/rigs/<rig>.json`` (the examples in
``configs/examples/rigs/``), so every team's rigs are reviewed with the code. A
curriculum says what is played and plays the same on every rig; the rig file
says how this rig plays it, and every run played on the rig reads it.

A run names its rig file (:meth:`RigConfig.open`) as it names its curriculum;
there is no default, since which window, controller and data folder a session
gets is not something to guess.

Every key of a rig file is optional: an empty ``{}`` plays on the defaults. A
rig's ``site`` and ``rig`` are the team's folder (:func:`site_of`) and the
file's name; written in the file they must say the same. It may describe the
rig for people -- :data:`FIELDS`, notes nothing plays from, filed with every
run -- and say how its runs are played (:data:`DEFAULTS` when left out), under the field
names of :class:`RigConfig` and :class:`ScreenConfig`, so the file and the code
say the same thing:

- ``"screen"``: ``{"size": "1024x768", "is_fullscreen": false, "monitor": 0,
  "has_vsync": true}``, the monitor by index (0: the first); fullscreen scales
  ``size`` to the monitor, aspect kept, at the monitor's own resolution;
- ``"inputs"``: the rig's input devices, the keyboard among them, by the names of
  :data:`humans.inputs.INPUT_DEVICE_CLASSES_BY_NAME` (``["keyboard",
  "current_designs_gamepad3"]``); each controller must be plugged in when a run starts;
- ``"response_keys"``: the keys a response device types by itself (a button
  box, a fORP in a key mode), each with what it stands for, ``{"1": "index
  finger", ...}``;
- ``"has_audio"``: play game audio; ``false`` mutes every block;
- ``"data_root"``: where the BIDS tree goes, ``<root>/sub-XX/ses-NNN/beh/<run>/``;
- ``"start_fixation"``, ``"end_fixation"``: seconds of fixation cross every run
  opens on after the trigger and closes on, the modality's baseline (fMRI: 10
  each, for the haemodynamic lag); left out, the run shows none;
- ``"triggers"``: the scanner's start sync and the trigger codes, over which
  line (:class:`~humans.sync.triggers.TriggerSettings`); left out, the run waits
  for ``=`` and sends no codes, and the settings mark it (``defaulted``);
- ``"rig_measurement"``: the equipment the rig's timing is measured with
  (:class:`RigMeasurementConfig`): the photodiode, a microphone, the recording's rate,
  more frame rates.

**Controls and response keys**, never mixed. A curriculum's **controls** (a game
phase's ``"controls"``) are control names of the rig's input devices
(:mod:`humans.inputs`), mapped to the game's actions. A rig's **response
keys** are keyboard keys a device types itself; a curriculum binds them like
any key.

What was measured on the rig is kept beside its file (:func:`measured_latency_files`).
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..inputs import INPUT_DEVICE_CLASSES_BY_NAME, Keyboard
from ..sync.triggers import TriggerError, TriggerSettings

MODALITIES = ("fmri", "meg", "eeg", "behav")
#: The modalities that record the brain: their runs need a fixation baseline at each end.
NEUROIMAGING = ("fmri", "meg", "eeg")
#: What the rig is, for people: notes nothing plays from, each optional, filed with
#: every run (``site`` and ``rig`` are the file's place).
FIELDS = {
    "site": "the team, the folder its rigs are in (configs/<team>/rigs/): letters, digits, "
            "_ and -",
    "rig": "this rig of the team, the rig file's name, its place included (mgh-prisma): "
           "letters, digits, _ and -",
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
    "site": "MT", "rig": "mgh-prisma", "pi": "MT", "modality": "fmri",
    "monitor": "BOLDscreen 32 LCD, 120 Hz, 1920x1080",
    "photodiode": "BioSemi photodiode on sound card input 0",
    "audio_path": "RME Babyface -> Sensimetrics S14 insert earphones",
    "trigger_hardware": "fORP 932 USB, '=' at each volume",
}
#: How a run is played where the rig file says nothing: a window at the desk,
#: the keyboard alone, the audio on, no fixation at either end.
DEFAULTS: dict[str, Any] = {
    "screen": {"size": "1024x768", "is_fullscreen": False, "monitor": 0, "has_vsync": True},
    "inputs": ["keyboard"], "has_audio": True, "data_root": "data", "start_fixation": None,
    "end_fixation": None, "triggers": {}, "response_keys": {}, "rig_measurement": {},
}
#: Keys refused with a pointer to ``"inputs"``, which says what they would mean.
_GONE = {"has_pad", "bindings"}
#: How the rig's photodiode can be read: not at all, on this PC's sound card
#: (input 0), or in the MEG/EEG recording (matched offline).
PHOTODIODE_READOUTS = ("none", "soundcard", "recording")
#: Where the flashing patch the photodiode reads is.
CORNERS = ("br", "bl", "tr", "tl")
#: A rig file's ``"rig_measurement"`` where it says nothing (:class:`RigMeasurementConfig`).
RIG_MEASUREMENT_DEFAULTS: dict[str, Any] = {
    "photodiode": "none", "input_device": None, "has_mic": False, "corner": "br",
    "patch_px": 120, "recording_hz": None, "frame_rates": [],
}
#: Trigger presets the rig form offers. Each names ``sync.mode``
#: and ``backend`` outright: a section that leaves them out runs, but is
#: reported as NOT SET.
TRIGGER_PRESETS: dict[str, dict] = {
    "fMRI (wait for EQUALS, no codes)": {
        "sync": {"mode": "wait", "key": "EQUALS"}, "backend": "null"},
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
LABEL = re.compile(r"[A-Za-z0-9_-]+")
_INITIALS = re.compile(r"[A-Z]{2,4}")
_SIZE = re.compile(r"([1-9]\d*)x([1-9]\d*)")
#: The repo's ``configs/``: a folder per team, each with its ``rigs/``, and the examples.
CONFIGS = Path(__file__).resolve().parents[2] / "configs"
#: The folder a team keeps its rig files in.
RIGS_DIRNAME = "rigs"
#: The folder beside a rig file that holds what was measured on the rig.
MEASURED_DIRNAME = "measured"


@dataclass(frozen=True, slots=True)
class RigMeasurementConfig:
    """The equipment the rig's timing is measured with: a rig file's ``"rig_measurement"``."""

    #: one of :data:`PHOTODIODE_READOUTS`; ``none``: the rig has no photodiode
    photodiode: str
    #: the sound-card input the diode (and microphone) are on: index or name; ``None``: default
    input_device: int | str | None
    #: a microphone at the ear on the sound card's input 1 times the sound as heard
    has_mic: bool
    #: where the flashing patch is, under the diode: one of :data:`CORNERS`
    corner: str
    patch_px: int
    #: the MEG/EEG recording's sampling rate: each frame code must last two samples there
    recording_hz: float | None
    #: frame rates this rig's games use, measured besides the usual ones
    frame_rates: tuple[float, ...]

    @classmethod
    def from_dict(cls, section: dict) -> RigMeasurementConfig:
        """Read a checked ``"rig_measurement"`` (:func:`problems`), defaults where it says
        nothing."""
        d = {**RIG_MEASUREMENT_DEFAULTS, **section}
        device = d["input_device"]
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        return cls(photodiode=d["photodiode"], input_device=device, has_mic=d["has_mic"],
                   corner=d["corner"], patch_px=d["patch_px"], recording_hz=d["recording_hz"],
                   frame_rates=tuple(d["frame_rates"]))

    def record(self) -> dict[str, Any]:
        """As a rig file writes it."""
        return {**asdict(self), "frame_rates": list(self.frame_rates)}


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
    #: the input devices, the keyboard among them, by
    #: :data:`humans.inputs.INPUT_DEVICE_CLASSES_BY_NAME` name.
    inputs: tuple[str, ...]
    #: the keys a response device types itself -> what each stands for
    response_keys: dict[str, str]
    #: play game audio; ``False`` mutes every block and opens no output.
    has_audio: bool
    #: the BIDS tree's root.
    data_root: str
    #: seconds of fixation every run opens on after the trigger; ``None``: none.
    start_fixation: float | None
    #: seconds of fixation every run closes on; ``None``: none.
    end_fixation: float | None
    #: the scanner's start sync and the trigger codes, over which line.
    triggers: TriggerSettings
    #: the equipment the rig's timing is measured with.
    rig_measurement: RigMeasurementConfig

    @classmethod
    def from_dict(cls, rig: dict, path: str) -> RigConfig:
        """Read a rig file's JSON, :data:`DEFAULTS` where it says nothing.

        :param rig: the file's JSON.
        :param path: the file; its folder and name are the rig's ``site`` and ``rig``,
            which a ``site`` or ``rig`` it writes must match.
        :raises ValueError: naming each field that is unknown or invalid, and how to fix it.
        """
        found = problems(rig)
        site, rig_name = site_of(path), Path(path).stem
        found += [f'"{k}" is {rig[k]!r}, but the file is {path}: the site is its team\'s '
                  f"folder, the rig its name" for k, own in (("site", site), ("rig", rig_name))
                  if rig.get(k, own) != own]
        if found:
            raise ValueError(f"{path}: " + "; ".join(found))
        settings = {**DEFAULTS, **rig}
        fields = {"site": site, "rig": rig_name,
                  **{k: rig[k] for k in FIELDS if k in rig and k not in ("site", "rig")}}
        return cls(path=path, fields=fields,
                   screen=ScreenConfig.from_dict(rig.get("screen", {})),
                   inputs=tuple(settings["inputs"]),
                   response_keys=dict(settings["response_keys"]),
                   has_audio=settings["has_audio"],
                   data_root=settings["data_root"],
                   start_fixation=settings["start_fixation"],
                   end_fixation=settings["end_fixation"],
                   triggers=TriggerSettings.from_dict(settings["triggers"]),
                   rig_measurement=RigMeasurementConfig.from_dict(settings["rig_measurement"]))

    @classmethod
    def open(cls, path: str) -> RigConfig:
        """The rig file at ``path``, checked.

        :raises ValueError: a file that is missing, not a JSON object, or that
            :meth:`from_dict` refuses.
        """
        if not os.path.isfile(path):
            raise ValueError(f"{path}: no such rig file")
        return cls.from_dict(load(path), path)

    @property
    def has_trigger_setup(self) -> bool:
        """Whether the file says how the run starts or where codes go (not both defaulted)."""
        return set(self.triggers.defaulted) != {"sync.mode", "backend"}

    def without_trigger_line(self) -> RigConfig:
        """This rig with no trigger line (``null`` backend), for measuring the rest of it when
        the line does not open. ``send`` becomes ``none``, since it needs a line; the file
        is untouched."""
        sync = self.triggers.sync
        if sync.mode == "send":
            sync = replace(sync, mode="none")
        return replace(self, triggers=replace(self.triggers, backend="null", sync=sync))

    def record(self) -> dict[str, Any]:
        """The rig file as the run played it, every default written out: the manifest's."""
        return {"path": self.path, **self.fields, "screen": self.screen.record(),
                "inputs": list(self.inputs), "response_keys": self.response_keys,
                "has_audio": self.has_audio, "data_root": self.data_root,
                "start_fixation": self.start_fixation, "end_fixation": self.end_fixation,
                "triggers": asdict(self.triggers), "rig_measurement": self.rig_measurement.record()}

    def status(self) -> str:
        """One line for the console: which file, and what it settles."""
        screen = self.screen
        window = (f"{screen.size[0]}x{screen.size[1]} "
                  f"{'fullscreen' if screen.is_fullscreen else 'window'}")
        fixations = "".join(f" | {s:g} s fixation at the {end}" for end, s in
                            (("start", self.start_fixation), ("end", self.end_fixation))
                            if s is not None)
        return (f"{self.path} | {window} on monitor {screen.monitor}, vsync "
                f"{'on' if screen.has_vsync else 'off'} | inputs "
                f"{', '.join(self.inputs)}"
                f" | audio {'on' if self.has_audio else 'off'} | data in {self.data_root}"
                f"{fixations}"
                f" | triggers {self.triggers.sync.mode}, codes over {self.triggers.backend}")

    @property
    def fixation_warning(self) -> str | None:
        """What is odd about the fixations for the rig's modality: a brain recording with
        no baseline at an end; ``None`` when nothing is."""
        return fixation_warning(self.fields.get("modality"), self.start_fixation,
                                self.end_fixation)

    @property
    def measured_latency_files(self) -> list[str]:
        """What was measured on this rig, oldest first (:func:`measured_latency_files`)."""
        return measured_latency_files(self.path)

    @property
    def measured_latency_warning(self) -> str | None:
        """That nothing was ever measured on this rig; ``None`` once something was."""
        if self.measured_latency_files:
            return None
        return (f"{self.path} has no measured latencies beside it: when a frame's light and "
                "sound reach the participant on this rig is not known")


def fixation_warning(modality: str | None, start: float | None,
                     end: float | None) -> str | None:
    """A neuroimaging rig whose runs have no fixation at one end, or both, in words.

    :return: the warning; ``None`` for another modality, or fixations at both ends.
    """
    if modality not in NEUROIMAGING:
        return None
    missing = [name for name, s in (("start_fixation", start), ("end_fixation", end)) if not s]
    if not missing:
        return None
    return (f"the rig's modality is {modality}, but it has no {' and no '.join(missing)}: its "
            "runs have no fixation baseline there (10 s each is usual for fMRI)")


def measured_latency_files(rig_path: str) -> list[str]:
    """What was measured on the rig file at ``rig_path``, in the order it was filed:
    ``<rig>-<n>.json`` in :data:`MEASURED_DIRNAME` beside it, by ``n``."""
    folder = os.path.join(os.path.dirname(rig_path), MEASURED_DIRNAME)
    rig = Path(rig_path).stem
    numbered = {}
    for path in Path(folder).glob(f"{glob.escape(rig)}-*.json"):
        n = re.fullmatch(rf"{re.escape(rig)}-(\d+)\.json", path.name)
        if n:
            numbered[int(n[1])] = str(path)
    return [numbered[n] for n in sorted(numbered)]


def next_measured_latency_file(rig_path: str) -> str:
    """Where the next measurement of the rig file at ``rig_path`` is filed: one more than
    the last :func:`measured_latency_files`."""
    rig = Path(rig_path).stem
    return os.path.join(os.path.dirname(rig_path), MEASURED_DIRNAME,
                        f"{rig}-{len(measured_latency_files(rig_path)) + 1}.json")


def site_of(path: str) -> str:
    """The site a rig file is filed under: the folder holding its ``rigs/`` folder
    (``configs/<team>/rigs/<rig>.json``: the team), or else its own folder."""
    folder = Path(os.path.abspath(path)).parent
    return folder.parent.name if folder.name == RIGS_DIRNAME else folder.name


def repo_rig_files() -> list[Path]:
    """The rig files kept in the repo: every team's (``configs/<team>/rigs/``), examples
    included."""
    return sorted(CONFIGS.glob(f"*/{RIGS_DIRNAME}/*.json"))


def rig_files() -> list[str]:
    """The rig files kept in the repo (:func:`repo_rig_files`), as paths from the working
    directory where they can be, for the editor to offer."""
    files = repo_rig_files()
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
    out = [f"unknown key {k!r}" + (': "bindings" and "has_pad" are gone: list the controller '
                                   'under "inputs" and name its controls in the curriculum'
                                   if k in _GONE else "")
           for k in rig if k not in FIELDS and k not in DEFAULTS]
    fields = {k: v for k, v in rig.items() if k in FIELDS}
    out += [f"{k!r} must be a string" for k, v in fields.items() if not isinstance(v, str)]
    out += [f"{k!r} is empty" for k, v in fields.items() if k != "notes" and v == ""]
    for k in ("site", "rig"):
        if isinstance(rig.get(k), str) and rig[k] and not LABEL.fullmatch(rig[k]):
            out.append(f"{k!r} must be letters, digits, _ or -, got {rig[k]!r}")
    if isinstance(rig.get("pi"), str) and rig["pi"] and not _INITIALS.fullmatch(rig["pi"]):
        out.append(f"'pi' must be 2 to 4 capital letters (initials), got {rig['pi']!r}")
    if rig.get("modality") not in (*MODALITIES, "", None):
        out.append(f"'modality' must be one of {MODALITIES}, got {rig['modality']!r}")
    return out + _setting_problems(rig)


def _setting_problems(rig: dict) -> list[str]:
    """What is wrong with the file's ``screen``, ``inputs``, ``has_audio``, ``data_root``,
    fixations, triggers, response keys and ``rig_measurement``."""
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
             ("has_audio", rig.get("has_audio", True)))
    out += [f'"{k}" must be true or false, got {v!r}' for k, v in flags if not isinstance(v, bool)]
    root = rig.get("data_root", DEFAULTS["data_root"])
    if not isinstance(root, str) or not root:
        out.append(f'"data_root" must be a folder, got {root!r}')
    out += [f'"{k}" must be seconds, 0 or more, or null for none, got {rig[k]!r}'
            for k in ("start_fixation", "end_fixation")
            if rig.get(k) is not None and not _is_seconds(rig[k])]
    return (out + trigger_problems(rig.get("triggers", {})) + _input_device_problems(rig)
            + _response_key_problems(rig)
            + _rig_measurement_problems(rig.get("rig_measurement", {})))


def _input_device_problems(rig: dict) -> list[str]:
    """``inputs``: distinct names of the kinds of input device the rig has."""
    names = rig.get("inputs", DEFAULTS["inputs"])
    known = INPUT_DEVICE_CLASSES_BY_NAME
    if (isinstance(names, list) and all(isinstance(n, str) and n in known for n in names)
            and len(set(names)) == len(names)):
        return []
    return [f'"inputs": expected a list of distinct names from {sorted(known)}, got {names!r}']


def _response_key_problems(rig: dict) -> list[str]:
    """``response_keys``: keys a device types, each to a short text of what it stands for."""
    keys = rig.get("response_keys", {})
    if not isinstance(keys, dict):
        return [f'"response_keys": expected {{key: what it stands for}}, got {keys!r}']
    return ([f'"response_keys": {k!r} is not a keyboard key name (as in '
             "humans/inputs/keyboard.py)" for k in keys if k not in Keyboard.CONTROLS]
            + [f'"response_keys": {k!r}: say what it stands for, as text' for k, v in keys.items()
               if not isinstance(v, str)])


def _rig_measurement_problems(section: Any) -> list[str]:
    """``rig_measurement``: :data:`RIG_MEASUREMENT_DEFAULTS`' keys, each of the right kind."""
    if not isinstance(section, dict):
        return [f'"rig_measurement": expected an object, got {section!r}']
    d = {**RIG_MEASUREMENT_DEFAULTS, **section}
    out = [f'"rig_measurement": unknown key {k!r}'
           for k in section if k not in RIG_MEASUREMENT_DEFAULTS]
    if d["photodiode"] not in PHOTODIODE_READOUTS:
        out.append(f'"rig_measurement": "photodiode" must be one of {PHOTODIODE_READOUTS}, '
                   f'got {d["photodiode"]!r}')
    if not isinstance(d["has_mic"], bool):
        out.append(f'"rig_measurement": "has_mic" must be true or false, got {d["has_mic"]!r}')
    if d["has_mic"] and d["photodiode"] != "soundcard":
        out.append('"rig_measurement": "has_mic" needs "photodiode": "soundcard" -- the microphone '
                   "is read on the sound card, next to the diode")
    if d["corner"] not in CORNERS:
        out.append(f'"rig_measurement": "corner" must be one of {CORNERS}, got {d["corner"]!r}')
    if not _is_positive(d["patch_px"], int):
        out.append(f'"rig_measurement": "patch_px" must be a whole number of pixels, got '
                   f'{d["patch_px"]!r}')
    if d["recording_hz"] is not None and not _is_positive(d["recording_hz"]):
        out.append(f'"rig_measurement": "recording_hz" must be a rate in Hz, or null, got '
                   f'{d["recording_hz"]!r}')
    rates = d["frame_rates"]
    if not (isinstance(rates, list) and all(_is_positive(r) for r in rates)):
        out.append(f'"rig_measurement": "frame_rates" must be a list of frames per second, got '
                   f"{rates!r}")
    if not (d["input_device"] is None or isinstance(d["input_device"], (int, str))):
        out.append(f'"rig_measurement": "input_device" must be an index or a name, got '
                   f'{d["input_device"]!r}')
    return out


def _is_positive(v: Any, kind: type | tuple[type, ...] = (int, float)) -> bool:
    return isinstance(v, kind) and not isinstance(v, bool) and v > 0


def _is_seconds(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def trigger_problems(section: Any) -> list[str]:
    """What :class:`~humans.sync.triggers.TriggerSettings` refuses in a ``triggers`` section,
    as text.

    Two checks are added here: the settings accept a missing port, which the
    backend refuses only on opening, and a scanner key that is no keyboard key,
    which a run would wait for forever.

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
    if s.sync.mode == "wait" and s.sync.key not in Keyboard.CONTROLS:
        return [f"triggers: sync.key {s.sync.key!r} is not a keyboard key name (as in "
                "humans/inputs/keyboard.py): name the key the trigger box types"]
    return []
