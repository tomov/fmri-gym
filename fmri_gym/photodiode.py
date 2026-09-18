"""Photodiode calibration: measure the flip-to-photon offset of this rig.

``flip_time`` (and the trigger sent on the flip) says when the flip returned;
photons leave the panel some constant time later -- swap-chain depth, panel
response, any compositor. Software cannot see that offset; a photodiode on
the screen can. This task, run once per rig with the same :class:`Display`
and :class:`Triggers` a session uses, flashes a patch and records the flips,
so the offset can be measured and then subtracted in analysis::

    python -m fmri_gym.photodiode --fullscreen --config configs/demo_meg.json

The patch (default 120 px, bottom-right) goes white ``n`` times for
``--on-ms`` and black again, with a randomized gap in between so nothing
aliases with the refresh. On the flip that turns it white the frame trigger is
sent (:meth:`Triggers.frame`, via :meth:`Display.call_on_flip`), and the flip
time is logged; ``task_start`` / ``task_stop`` bracket the run.

Two ways to read the diode, because how it is plugged in varies by lab:

* **Into the recording system** (MEG ADC / EEG aux channel), next to the
  trigger line. The offset is then trigger-to-edge *inside that recording*,
  no PC clock involved: extract the trigger times and the diode edges there
  (MNE or the vendor tool) and hand both to :func:`match_edges`.
* **Into this PC's sound card** (``--audio``): a photodiode on a mic/line
  input is a standard trick, and ``sounddevice`` is already a dependency.
  The task records the input during the flashes, maps PortAudio's ADC
  timestamps onto ``perf_counter`` (the flip-time clock), detects the first
  deviation from baseline after each white flip (:func:`detect_edges`) and
  prints the offset distribution on the spot. Audio inputs are AC-coupled,
  so a step arrives as a sharp pulse -- fine for an onset. The sound card's
  own input latency is inside the number; PortAudio reports the ADC time of
  each buffer, which compensates it as far as the driver allows.

**Sound** (``--audio-click``): the same run also measures when sound is
heard. On every white flip a 10 ms tone burst is queued through
:class:`~fmri_gym.audio.Audio` exactly as a game frame's sound is (same
output, same delay chosen at start-up), and its DAC onset is logged. A microphone at the ear -- at the end of the scanner's
sound tube, or inside the headphone -- gives the acoustic edge. Recorded on
the second input channel with ``--audio`` (diode on channel 0, microphone on
channel 1), the task prints flip-to-sound and photon-to-sound offsets on the
spot; recorded on a MEG/EEG channel, :func:`match_edges` does the same
offline against the triggers. Expect flip-to-sound = the printed delay plus the
speaker/tube delay, with a spread of about a millisecond on a low-latency
device.

Output: ``data/photodiode_<timestamp>/`` with ``photodiode.npz`` (per-flash
``flip_on``, ``flip_off``, ``trigger``, ``click_dac``, and the audio signal if
recorded) and ``summary.json`` (display, trigger and audio settings, offset
statistics).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pygame

from .adapters.base import Sound
from .display import Display
from .triggers import TriggerSettings, Triggers

if TYPE_CHECKING:
    from .audio import Audio

CORNERS = ("br", "bl", "tr", "tl")


# ---------------------------------------------------------------------------
# Stimulus
# ---------------------------------------------------------------------------


def _patch_rect(size: tuple[int, int], corner: str, px: int) -> pygame.Rect:
    """The patch rectangle in a corner of the screen."""
    x = size[0] - px if corner[1] == "r" else 0
    y = size[1] - px if corner[0] == "b" else 0
    return pygame.Rect(x, y, px, px)


def _show_patch(display: Display, on: bool, corner: str, px: int) -> float:
    """Draw the patch white or black on a black canvas and present it.

    :return: ``perf_counter`` of the flip.
    """
    display.canvas.fill((0, 0, 0))
    if on:
        display.canvas.fill((255, 255, 255), _patch_rect(display.size, corner, px))
    return display.redraw()


def _hold(display: Display, seconds: float) -> None:
    """Keep the current screen up for ``seconds`` (re-presenting when locked).

    :raises KeyboardInterrupt: on ESC or window close.
    """
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise KeyboardInterrupt
        display.idle(end)


def run_flashes(
    display: Display,
    triggers: Triggers,
    n: int,
    on_ms: float,
    gap_ms: tuple[float, float],
    rng: random.Random,
    corner: str = "br",
    px: int = 120,
    click: Callable[[float], None] | None = None,
) -> dict[str, np.ndarray]:
    """Flash the patch ``n`` times; a frame trigger goes out on each white flip.

    :param display: the display to draw on.
    :param triggers: trigger sender (``task_start``/``task_stop`` bracket the run).
    :param n: number of flashes.
    :param on_ms: how long the patch stays white.
    :param gap_ms: ``(min, max)`` black gap before each flash, drawn uniformly.
    :param rng: random source for the gaps (seeded for a reproducible schedule).
    :param corner: ``"br"``, ``"bl"``, ``"tr"`` or ``"tl"``.
    :param px: patch side in pixels.
    :param click: called with each white flip's time, to queue a sound on it.
    :return: ``flip_on``, ``flip_off`` (``perf_counter``), ``trigger`` (value sent).
    """
    flip_on, flip_off, values = [], [], []
    _show_patch(display, False, corner, px)
    _hold(display, 1.0)                       # let the diode and the chain settle
    triggers.lifecycle("task_start")
    for _ in range(n):
        _hold(display, rng.uniform(*gap_ms) / 1000.0)
        display.call_on_flip(triggers.frame)
        flip_on.append(_show_patch(display, True, corner, px))
        if click is not None:
            click(flip_on[-1])
        values.append(triggers.last_frame)
        _hold(display, on_ms / 1000.0)
        flip_off.append(_show_patch(display, False, corner, px))
    _hold(display, 0.5)
    triggers.block_end()
    triggers.lifecycle("task_stop")
    return {"flip_on": np.asarray(flip_on), "flip_off": np.asarray(flip_off),
            "trigger": np.asarray(values, dtype=np.int16)}


# ---------------------------------------------------------------------------
# Readout: sound-card recording and edge detection
# ---------------------------------------------------------------------------


class AudioRecorder:
    """Record one input channel, with every sample mapped onto ``perf_counter``.

    PortAudio stamps each buffer with the ADC time of its first sample on the
    stream clock; :meth:`sync` samples that clock against ``perf_counter`` so
    the recording and the flip times share a time base.
    """

    def __init__(
        self, device: int | str | None = None, samplerate: int = 48000, channels: int = 1
    ) -> None:
        import sounddevice as sd
        self.fs = samplerate
        self._chunks: list[tuple[float, np.ndarray]] = []
        self._pairs: list[tuple[float, float]] = []
        self._stream = sd.InputStream(device=device, channels=channels, samplerate=samplerate,
                                      callback=self._callback)

    def _callback(self, indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
        self._chunks.append((float(t.inputBufferAdcTime), indata.copy()))

    def sync(self) -> None:
        """Record a (stream clock, perf_counter) pair."""
        self._pairs.append((float(self._stream.time), time.perf_counter()))

    def start(self) -> None:
        self._stream.start()
        time.sleep(0.2)
        self.sync()

    def stop(self) -> None:
        self.sync()
        self._stream.stop()
        self._stream.close()

    def signal(self) -> tuple[np.ndarray, np.ndarray]:
        """The recording as ``(perf_counter times, samples shaped (n, channels))``."""
        offset = float(np.median([p - s for s, p in self._pairs]))
        times, values = [], []
        for adc_t, x in self._chunks:
            times.append(adc_t + offset + np.arange(len(x)) / self.fs)
            values.append(x)
        if not values:
            return np.zeros(0), np.zeros((0, self._stream.channels))
        return np.concatenate(times), np.concatenate(values)


def tone_burst(samplerate: float, ms: float = 10.0, hz: float = 2000.0) -> Sound:
    """A short stereo tone burst: a sharp, unambiguous acoustic onset.

    :param samplerate: output sample rate.
    :param ms: duration.
    :param hz: tone frequency.
    :return: int16 PCM at half scale.
    """
    t = np.arange(int(samplerate * ms / 1000)) / samplerate
    mono = (np.sin(2 * np.pi * hz * t) * 16384).astype(np.int16)
    return Sound(np.column_stack([mono, mono]), samplerate)


def detect_edges(
    t: np.ndarray, x: np.ndarray, onsets: np.ndarray, window: float = 0.2, k: float = 8.0
) -> np.ndarray:
    """First deviation from baseline after each onset -- the diode's edge.

    Baseline and noise are the median and MAD of the whole signal, so the
    polarity of the diode and the AC coupling of an audio input do not
    matter: any excursion beyond ``k`` MADs counts.

    :param t: sample times (same clock as ``onsets``).
    :param x: samples.
    :param onsets: times to search after (the white flips).
    :param window: seconds after each onset to look in.
    :param k: threshold in MADs.
    :return: one edge time per onset, ``nan`` where none was found.
    """
    base = float(np.median(x))
    mad = float(np.median(np.abs(x - base))) * 1.4826 + 1e-12
    hot = np.abs(x - base) > k * mad
    edges = np.full(len(onsets), np.nan)
    for i, t0 in enumerate(onsets):
        sel = np.flatnonzero((t >= t0) & (t < t0 + window) & hot)
        if sel.size:
            edges[i] = t[sel[0]]
    return edges


def match_edges(onsets: np.ndarray, edges: np.ndarray, window: float = 0.2) -> np.ndarray:
    """Offset from each onset to the first edge that follows it.

    For the offline path: ``onsets`` are the trigger times and ``edges`` the
    diode edges, both extracted from the same recording.

    :param onsets: trigger (or flip) times.
    :param edges: detected edge times, any order.
    :param window: seconds after an onset within which an edge counts.
    :return: ``edge - onset`` per onset in seconds, ``nan`` where unmatched.
    """
    edges = np.sort(np.asarray(edges, dtype=float))
    out = np.full(len(onsets), np.nan)
    for i, t0 in enumerate(onsets):
        j = int(np.searchsorted(edges, t0))
        if j < len(edges) and edges[j] - t0 < window:
            out[i] = edges[j] - t0
    return out


def summarize(offsets: np.ndarray) -> dict[str, float]:
    """Median / SD / min / max of the matched offsets, in ms."""
    ok = offsets[~np.isnan(offsets)] * 1000.0
    if ok.size == 0:
        return {"n": int(len(offsets)), "n_matched": 0}
    return {"n": int(len(offsets)), "n_matched": int(ok.size),
            "median_ms": float(np.median(ok)), "sd_ms": float(np.std(ok)),
            "min_ms": float(ok.min()), "max_ms": float(ok.max())}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _trigger_settings(args: argparse.Namespace) -> TriggerSettings:
    """Trigger settings from ``--config`` (its ``triggers`` section), overridden by flags."""
    section: dict = {}
    if args.config:
        with open(args.config) as f:
            section = dict(json.load(f).get("triggers", {}))
        section.pop("sync", None)     # the calibration has no run start to sync
    if args.trigger_backend:
        section["backend"] = args.trigger_backend
    if args.trigger_port:
        section["port"] = args.trigger_port
    return TriggerSettings.from_dict(section)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flash a patch to measure flip-to-photon offset.")
    p.add_argument("--n", type=int, default=100, help="number of flashes")
    p.add_argument("--on-ms", type=float, default=100.0, help="white duration per flash")
    p.add_argument("--gap-ms", type=float, nargs=2, default=(400.0, 800.0),
                   metavar=("MIN", "MAX"), help="black gap before each flash, uniform")
    p.add_argument("--corner", choices=CORNERS, default="br")
    p.add_argument("--patch-px", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--config", help="curriculum/config JSON whose triggers section to use")
    p.add_argument("--trigger-backend", choices=("null", "lsl", "serial", "parallel"))
    p.add_argument("--trigger-port")
    p.add_argument("--audio", action="store_true",
                   help="record the photodiode on the sound-card input and compute offsets")
    p.add_argument("--audio-device", help="sounddevice input device (index or name substring)")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--audio-click", action="store_true",
                   help="queue a tone burst on each white flip through the session's audio "
                        "output; with --audio, record a microphone "
                        "on input channel 1")
    p.add_argument("--list-audio-devices", action="store_true")
    p.add_argument("--outdir")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_audio_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    outdir = args.outdir or os.path.join("data", f"photodiode_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    w, h = (int(x) for x in args.size.lower().split("x"))
    triggers = Triggers(_trigger_settings(args))    # before the window: fails at the desk
    audio = _audio_output() if args.audio_click else None
    display = Display((w, h), fullscreen=args.fullscreen, vsync=not args.no_vsync)
    click, chunks = _clicker(audio) if audio is not None else (None, [])
    recorder = None
    if args.audio:
        # Asked for the audio readout: if the input cannot open, stop here
        # rather than flash for a minute and report "no edges found".
        device = args.audio_device
        if (device or "").isdigit():
            device = int(device)
        recorder = AudioRecorder(device, args.samplerate, channels=2 if args.audio_click else 1)
        recorder.start()
    summary: dict[str, Any] = {"display": display.describe(), "triggers": triggers.describe(),
                               "n": args.n, "on_ms": args.on_ms, "gap_ms": list(args.gap_ms),
                               "corner": args.corner, "patch_px": args.patch_px}
    arrays: dict[str, np.ndarray] = {}
    try:
        arrays = run_flashes(display, triggers, args.n, args.on_ms, tuple(args.gap_ms),
                             random.Random(args.seed), args.corner, args.patch_px, click)
    except KeyboardInterrupt:
        print("photodiode: interrupted", file=sys.stderr)
    finally:
        if recorder is not None:
            recorder.stop()
        if audio is not None:
            _log_clicks(audio, chunks, arrays, summary)
        triggers.close()
        display.close()
    if recorder is not None and "flip_on" in arrays:
        _readout(*recorder.signal(), arrays, summary)
    summary["triggers"] = triggers.describe()
    np.savez_compressed(os.path.join(outdir, "photodiode.npz"), **arrays)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _report(summary, arrays, outdir)


def _audio_output() -> Audio:
    """The session's audio output, built as a session builds it."""
    from .audio import Audio
    audio = Audio()
    print(f"audio: {audio.status()}", file=sys.stderr)
    return audio


def _log_clicks(audio: Audio, chunks: list[int], arrays: dict, summary: dict) -> None:
    """Close the output and log when each click reached the DAC."""
    log = audio.block_log(chunks)
    audio.close()
    summary["audio_out"] = audio.describe()
    if log and "flip_on" in arrays:
        arrays["click_dac"] = log["audio_onset"]
        summary["click_dac_offset"] = summarize(log["audio_onset"] - arrays["flip_on"])


def _clicker(audio: Audio) -> tuple[Callable[[float], None], list[int]]:
    """A ``click`` for :func:`run_flashes`, and the chunk each click was queued as.

    :param audio: the output, as a session builds it.
    :return: ``(click, chunks)``; ``chunks`` fills as the flashes run.
    """
    burst = tone_burst(audio.samplerate)
    chunks: list[int] = []
    audio.start(frame_period=None, flip_period=None)    # clicks are not game steps

    def click(flip_t: float) -> None:
        audio.play(burst, flip_t)
        chunks.append(audio.last_chunk)

    return click, chunks


def _readout(t: np.ndarray, x: np.ndarray, arrays: dict, summary: dict) -> None:
    """Offsets from the sound-card recording: diode on channel 0, mic on 1."""
    arrays["audio_time"], arrays["audio"] = t, x
    edges = detect_edges(t, x[:, 0], arrays["flip_on"])
    arrays["edge_time"] = edges
    arrays["offset_s"] = edges - arrays["flip_on"]
    summary["offset"] = summarize(arrays["offset_s"])
    if "click_dac" not in arrays:
        return
    # From the click's DAC time, not the flip: the delay between them is room
    # noise that would pass for an edge.
    sound = detect_edges(t, x[:, 1], arrays["click_dac"] - 0.010)
    arrays["sound_time"] = sound
    arrays["sound_offset_s"] = sound - arrays["flip_on"]
    arrays["av_offset_s"] = sound - edges
    summary["sound_offset"] = summarize(arrays["sound_offset_s"])
    summary["av_offset"] = summarize(arrays["av_offset_s"])


def _report(summary: dict, arrays: dict, outdir: str) -> None:
    d = summary["display"]
    print(f"display: {d['size']} vsync={d['vsync']} refresh={d['refresh_rate']} Hz; "
          f"triggers: {summary['triggers']['active']}")
    if "flip_on" in arrays and len(arrays["flip_on"]) > 1:
        on = arrays["flip_on"]
        held = (arrays["flip_off"] - on) * 1000
        print(f"flashes: {len(on)}  white held {np.median(held):.1f} ms (median)")
    if "offset" in summary:
        o = summary["offset"]
        if o.get("n_matched"):
            print("flip -> photodiode offset: median {median_ms:.2f} ms  sd {sd_ms:.2f}  "
                  "min {min_ms:.2f}  max {max_ms:.2f}  (matched {n_matched}/{n} flashes; "
                  "a low fraction or a wide spread means noise, not the diode)".format(**o))
        else:
            print("flip -> photodiode offset: no edges found -- check the diode, its "
                  "input gain, and that it sits on the patch")
    else:
        print("no audio readout: match the trigger-channel codes to the diode edges in "
              "your recording with fmri_gym.photodiode.match_edges()")
    for key, label in (("click_dac_offset", "flip -> click at the DAC"),
                       ("sound_offset", "flip -> sound at the microphone"),
                       ("av_offset", "photodiode -> sound (audio behind video)")):
        o = summary.get(key, {})
        if o.get("n_matched"):
            print(f"{label}: median {o['median_ms']:.2f} ms  sd {o['sd_ms']:.2f}  "
                  f"min {o['min_ms']:.2f}  max {o['max_ms']:.2f}  "
                  f"(matched {o['n_matched']}/{o['n']})")
    print(f"saved: {outdir}")


if __name__ == "__main__":
    main()
