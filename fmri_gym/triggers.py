"""Recording-device triggers: run-start sync and per-event trigger codes.

fMRI needs one thing from the stimulus PC: to *wait* for the scanner's first
volume (the ``=`` key from the trigger box) and anchor the session clock on
it. MEG/EEG need the opposite direction: the stimulus PC *sends* trigger codes
onto a trigger line -- often also to start the acquisition -- and the analyst
realigns the recording to the frame log by those codes. Both live under one
optional ``"triggers"`` section of the config so a single curriculum runs at
either scanner; without the section, behaviour is the fMRI one as before.

One section, two concerns:

* ``sync`` (:class:`SyncSettings`) -- what gates the start of the session:
  ``wait`` for a key, ``send`` a start code over the trigger backend (some MEG
  systems start recording on it), or ``none``.
* the rest (:class:`TriggerSettings`) -- which codes go out, over what.
  Backends: ``null`` (default, sends nothing), ``lsl``, ``serial``,
  ``parallel``. A backend that cannot be opened raises :class:`TriggerError`
  when :class:`Triggers` is built -- in ``fmri_play.py`` that is before the
  window even opens: better a clear stop at the desk than a silent recording
  without triggers.

Code scheme -- disjoint bit fields, so coincident triggers stay decodable
-----------------------------------------------------------------------
A parallel port (and most MEG trigger inputs) is a set of lines read as one
integer, and two codes written within one sample of each other are seen
OR'd together. The scheme therefore never shares bits between events:

* **frames** cycle through the low ``frame_bits`` bits (``1..2**bits-1``,
  never 0, so "no frame" is distinguishable from "frame 0") -- one code per
  stepped frame, thinned by ``frame_every``. The absolute frame index is in
  the block's npz; the stream only has to show frame boundaries.
* **lifecycle** events (``task_start``, ``task_stop``, ``episode_start``,
  ``scanner_start``) each own one higher bit. A lifecycle value is OR'd with
  the frame level in force, so ``task_stop`` landing on frame code 5 reads
  as ``16 | 5`` and both survive. :meth:`Codes.validate` refuses any scheme
  where two codes overlap.

Action triggers are deliberately absent: every action is already logged per
frame with its session time, which is enough to realign it to the frame
triggers and the emulator state.

Backend deps (pylsl, pyserial, pyparallel) are imported lazily, in the backend
that needs them, so the core stays dependency-free.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .run import Clock

log = logging.getLogger(__name__)

LIFECYCLE_EVENTS = ("task_start", "task_stop", "episode_start", "scanner_start")
SYNC_MODES = ("wait", "send", "none")
TRIGGER_BACKENDS = ("null", "lsl", "serial", "parallel")


class TriggerError(RuntimeError):
    """A trigger setting or backend that cannot be used; raised before the session starts."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Codes:
    """The trigger code scheme; see the module docstring for the bit layout."""

    frame_bits: int = 3
    task_start: int = 1 << 3
    task_stop: int = 1 << 4
    episode_start: int = 1 << 5
    scanner_start: int = 1 << 6

    @property
    def frame_mask(self) -> int:
        """Bit mask of the frame field (also the number of distinct frame codes)."""
        return (1 << self.frame_bits) - 1

    def frame(self, n_sent: int) -> int:
        """Frame code for the ``n_sent``-th frame trigger of a block (cycles ``1..mask``).

        :param n_sent: how many frame triggers this block has sent so far.
        :return: a value in ``1..frame_mask``, never 0.
        """
        return 1 + n_sent % self.frame_mask

    def lifecycle(self, name: str) -> int:
        """Code of a lifecycle event.

        :param name: one of :data:`LIFECYCLE_EVENTS`.
        :return: its code (a single bit by default).
        """
        return int(getattr(self, name))

    def validate(self) -> None:
        """Refuse any scheme in which two triggers could not be told apart.

        :raises ValueError: if a code leaves one byte, overlaps the frame
            field, or overlaps another lifecycle code.
        """
        if not 1 <= self.frame_bits <= 7:
            raise ValueError(f"triggers: frame_bits must be in 1..7, got {self.frame_bits}")
        codes = {name: self.lifecycle(name) for name in LIFECYCLE_EVENTS}
        for name, code in codes.items():
            if not 1 <= code <= 255:
                raise ValueError(f"triggers: code {name}={code} must be in 1..255")
            if code & self.frame_mask:
                raise ValueError(f"triggers: code {name}={code} overlaps the frame bits "
                                 f"(mask {self.frame_mask}); use a higher bit")
        names = list(codes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if codes[a] & codes[b]:
                    raise ValueError(f"triggers: codes {a}={codes[a]} and {b}={codes[b]} "
                                     "share a bit; coincident triggers could not be decoded")


@dataclass(frozen=True)
class SyncSettings:
    """The ``triggers.sync`` section: what gates the start of the session."""

    #: ``wait`` for ``key`` (fMRI trigger box), ``send`` the ``scanner_start``
    #: code over the trigger backend (MEG started from the trigger line), or
    #: ``none`` (start on the experimenter's key alone).
    mode: str = "wait"
    key: str = "="
    #: ``send`` only: seconds between the start code and the clock anchor,
    #: for acquisitions that need a moment to come up.
    delay: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> SyncSettings:
        """Build from the config section.

        :param d: the ``sync`` dict (empty for the fMRI default).
        :return: validated settings.
        :raises ValueError: on an unknown mode.
        """
        s = cls(**d)
        if s.mode not in SYNC_MODES:
            raise ValueError(f"triggers: unknown sync mode {s.mode!r}; "
                             f"expected one of {SYNC_MODES}")
        return s


@dataclass(frozen=True)
class TriggerSettings:
    """The ``triggers`` section: the start sync plus what goes out, over what."""

    sync: SyncSettings = field(default_factory=SyncSettings)
    backend: str = "null"
    port: str | None = None
    lsl_stream_name: str = "fmri_gym"
    #: parallel only: how long a lifecycle code is held when no frame level
    #: is in force (between blocks); inside a block the next frame clears it.
    pulse_ms: float = 10.0
    on_frame: bool = True
    frame_every: int = 1
    on_episode_start: bool = True
    codes: Codes = field(default_factory=Codes)
    #: The two settings that decide what the recording gets and that the
    #: config left unset (``"sync.mode"``, ``"backend"``). Defaulting them is
    #: allowed, silently is not: the session shows them on the experimenter
    #: screen and the manifest keeps them.
    defaulted: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict | None) -> TriggerSettings:
        """Build from the config section, filling defaults and validating.

        :param d: the ``triggers`` dict, or ``None`` for all defaults.
        :return: validated settings.
        :raises ValueError: on an unknown backend or an ambiguous code scheme.
        :raises TriggerError: if ``sync.mode`` is ``send`` with no backend to
            send on.
        """
        d = dict(d or {})
        sync_d = d.pop("sync", {})
        defaulted = tuple(key for key, given in (("sync.mode", "mode" in sync_d),
                                                 ("backend", "backend" in d)) if not given)
        sync = SyncSettings.from_dict(sync_d)
        codes = Codes(**d.pop("codes", {}))
        codes.validate()
        s = cls(sync=sync, codes=codes, defaulted=defaulted, **d)
        if s.backend not in TRIGGER_BACKENDS:
            raise ValueError(f"triggers: unknown backend {s.backend!r}; "
                             f"expected one of {TRIGGER_BACKENDS}")
        if s.frame_every < 1:
            raise ValueError("triggers: frame_every must be >= 1")
        if sync.mode == "send" and s.backend == "null":
            raise TriggerError('triggers: sync.mode "send" sends scanner_start on the trigger '
                               'line, so triggers.backend must not be "null"')
        return s


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _Backend(Protocol):
    #: True when the transport holds the last value on its lines (parallel),
    #: so a lifecycle code sent between blocks must be cleared after a pulse.
    holds_level: bool

    def send(self, value: int) -> None: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


class _Null:
    holds_level = False

    def send(self, value: int) -> None:
        pass

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass


class _LSL:
    holds_level = False

    def __init__(self, stream_name: str) -> None:
        import pylsl
        info = pylsl.StreamInfo(stream_name, "Markers", 1, 0, "int32", f"{stream_name}_markers")
        self._outlet = pylsl.StreamOutlet(info)

    def send(self, value: int) -> None:
        self._outlet.push_sample([int(value)])

    def clear(self) -> None:
        pass

    def close(self) -> None:
        self._outlet = None


class _Serial:
    holds_level = False

    def __init__(self, port: str) -> None:
        import serial
        self._port = serial.serial_for_url(port, baudrate=115200, timeout=0)

    def send(self, value: int) -> None:
        self._port.write(bytes([value & 0xFF]))

    def clear(self) -> None:
        pass

    def close(self) -> None:
        self._port.close()


class _Parallel:
    holds_level = True

    def __init__(self, port: str) -> None:
        import parallel
        self._port = parallel.Parallel(port)
        self._port.setData(0)

    def send(self, value: int) -> None:
        self._port.setData(value & 0xFF)

    def clear(self) -> None:
        self._port.setData(0)

    def close(self) -> None:
        self.clear()


def _open_backend(s: TriggerSettings) -> tuple[_Backend, str]:
    """Open the configured transport.

    :param s: trigger settings.
    :return: ``(backend, description)``; the description names what is open,
        for the manifest and the console.
    :raises TriggerError: if the backend's package is missing, no port is
        configured, or the device cannot be opened -- with the fix in the
        message.
    """
    if s.backend == "null":
        return _Null(), "null"
    hint = 'set triggers.backend to "null" to run without trigger codes'
    target = s.lsl_stream_name if s.backend == "lsl" else s.port
    if s.backend != "lsl" and not s.port:
        raise TriggerError(f"triggers: backend {s.backend!r} needs a port "
                           f"(e.g. /dev/ttyUSB0 or /dev/parport0); {hint}")
    try:
        if s.backend == "lsl":
            return _LSL(s.lsl_stream_name), f"lsl:{target}"
        if s.backend == "serial":
            return _Serial(s.port), f"serial:{target}"
        return _Parallel(s.port), f"parallel:{target}"
    except ImportError as exc:
        pkg = {"lsl": "pylsl", "serial": "pyserial", "parallel": "pyparallel"}[s.backend]
        raise TriggerError(f"triggers: the {s.backend} backend needs the {pkg} "
                           f"package (pip install {pkg}); {hint}") from exc
    except Exception as exc:  # noqa: BLE001 -- any device failure, reworded with the fix
        raise TriggerError(f"triggers: cannot open {s.backend} triggers on {target}: {exc}; "
                           f"check the device and permissions, or {hint}") from exc


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


class Triggers:
    """Sends trigger codes for one session and remembers the lifecycle ones.

    Built once per run, next to the :class:`~fmri_gym.display.Display` and
    the :class:`~fmri_gym.audio.Audio`, and passed to the
    :class:`~fmri_gym.run.Run`; :meth:`from_config` on ``None`` gives
    the ``null`` no-op, so the session loop calls these methods
    unconditionally. Events are stamped with ``perf_counter``;
    :meth:`describe` converts them to session time once given the clock.

    :ivar sync: the start-sync settings, for the session's trigger screen.
    :ivar enabled: ``False`` for the ``null`` backend -- the session then skips
        logging a per-frame ``trigger`` column.
    :ivar events: every lifecycle trigger sent, with its value and times.
    """

    def __init__(self, settings: TriggerSettings) -> None:
        """Open the backend.

        :param settings: validated trigger settings.
        :raises TriggerError: if the backend cannot be opened.
        """
        self.settings = settings
        self.sync = settings.sync
        self.codes = settings.codes
        self._backend, self.active = _open_backend(settings)
        self.enabled = settings.backend != "null"
        self.events: list[dict[str, Any]] = []
        #: value sent by the latest :meth:`frame` call (0 if none), for the log.
        self.last_frame = 0
        self._level = 0     # frame code currently on the lines (0 between blocks)
        self._n_sent = 0    # frame triggers sent in the current block

    @classmethod
    def from_config(cls, section: dict | None) -> Triggers:
        """Build from the ``triggers`` config section.

        :param section: the dict, or ``None`` for the defaults (``null``
            backend, wait for the scanner key).
        :return: an open :class:`Triggers`.
        :raises TriggerError: if the backend cannot be opened.
        :raises ValueError: on an invalid section.
        """
        return cls(TriggerSettings.from_dict(section))

    def lifecycle(self, name: str) -> int:
        """Send a lifecycle trigger, OR'd with the frame level in force.

        Between blocks on a level-holding transport the code is pulsed for
        ``pulse_ms`` then cleared; inside a block the next frame clears it.

        :param name: one of :data:`LIFECYCLE_EVENTS`.
        :return: the value sent.
        """
        value = self.codes.lifecycle(name) | self._level
        self._backend.send(value)
        self.events.append({"name": name, "value": value,
                            "wall_time": time.time(), "perf_time": time.perf_counter()})
        if self._backend.holds_level and self._level == 0:
            time.sleep(self.settings.pulse_ms / 1000.0)
            self._backend.clear()
        return value

    def frame(self) -> int:
        """Send the next frame trigger of the block, if this frame gets one.

        :return: the value sent, or 0 when this frame is thinned out or frame
            triggers are off (the level then stays as it was).
        """
        s = self.settings
        n = self._n_sent
        self._n_sent += 1
        self.last_frame = 0
        if not s.on_frame or n % s.frame_every:
            return 0
        self._level = self.codes.frame(n // s.frame_every)
        self._backend.send(self._level)
        self.last_frame = self._level
        return self._level

    def episode_start(self) -> int:
        """Send ``episode_start`` if enabled.

        :return: the value sent, or 0.
        """
        if not self.settings.on_episode_start:
            return 0
        return self.lifecycle("episode_start")

    def block_end(self) -> None:
        """Drop the frame level at the end of a game block."""
        self._level = 0
        self._n_sent = 0
        self._backend.clear()

    def close(self) -> None:
        """Close the transport."""
        self._backend.close()

    def pulse(self, value: int, hold_s: float) -> float:
        """Put an arbitrary ``value`` on the line for ``hold_s``, then clear it.

        For testing the lines outside a session; a session sends only through
        :meth:`frame` and :meth:`lifecycle`, which keep the code scheme.

        :param value: the byte to send.
        :param hold_s: seconds before the lines are cleared.
        :return: seconds the send call itself took.
        """
        t0 = time.perf_counter()
        self._backend.send(value)
        took = time.perf_counter() - t0
        time.sleep(hold_s)
        self._backend.clear()
        return took

    def status(self) -> str:
        """One line saying what this run will do, for the experimenter screen and the console.

        Names the start sync, the transport (or that nothing is sent) and any
        setting the config left to its default -- the run should never end
        with someone discovering the recording got no triggers.

        :return: e.g. ``"wait for scanner key '=' | no trigger codes sent
            (backend null) | NOT SET in config: sync.mode, backend"``.
        """
        sync = self.sync
        start = {"wait": f"wait for scanner key {sync.key!r}",
                 "send": f"send scanner_start, then wait {sync.delay:g} s",
                 "none": "no scanner sync (start on SPACE)"}[sync.mode]
        out = (f"codes over {self.active}" if self.enabled
               else "no trigger codes sent (backend null)")
        parts = [start, out]
        if self.settings.defaulted:
            parts.append("NOT SET in config: " + ", ".join(self.settings.defaulted))
        return " | ".join(parts)

    def describe(self, clock: Clock | None = None) -> dict[str, Any]:
        """Settings, the open transport, and the lifecycle events sent.

        :param clock: a triggered session clock; each event then also gets a
            ``run_time`` (negative for ``scanner_start``, which precedes
            the anchor).
        :return: a JSON-serializable dict for the manifest.
        """
        events = [dict(e) for e in self.events]
        if clock is not None and clock.t0_perf is not None:
            for e in events:
                e["run_time"] = clock.from_perf(e["perf_time"])
        settings = asdict(self.settings)
        defaulted = settings.pop("defaulted")
        return {"settings": settings, "defaulted": defaulted, "active": self.active,
                "events": events}
