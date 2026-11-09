"""Filing a rig measurement, and its reports.

**What is filed.** A folder in the rig's data root, under
:data:`~fmri_gym.standalone.rig_setup.measure.SUBJECT` and its next session: ``manifest.json``,
one ``.npz`` per measurement, ``report.html`` for the people who run the rig (a verdict per
test, the numbers, charts, this rig's previous rig measurements; one offline file),
``report.md``, and ``rigmeasurement.tsv`` -- one row of fixed columns (:data:`COLUMNS`) with its
BIDS sidecar. The data root's ``rigmeasurements.tsv`` holds every rig measurement filed there,
read from their manifests.

**What the rig measured.** Every rig measurement that measured a latency also files it beside
the rig file (:func:`file_measured`, :func:`fmri_gym.config.rig.measured_latency_files`):
flip to light at the screen, flip to the sound card, sound card to the ear, with
every measurement's summary, the verdicts, the rig file as played, the software and
the machine (:func:`machine`: processor, memory, displays, GPUs, sound and input
devices, what is plugged into USB). The rig file is the rig's settings, shared; these
are one machine's measurements, and stay on it (``.gitignore``).
"""

from __future__ import annotations

import datetime
import glob
import html
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import pygame

from ...config import rig as rig_file
from ...inputs import ControlNames
from .measure import (LONG, QUICK, SUBJECT, TESTS, RigMeasurement, detect_outputs, input_devices,
                      port_name)

# ---------------------------------------------------------------------------
# The pooling row
# ---------------------------------------------------------------------------


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

#: The pooled table: ``(column, path into the manifest, description, units)``.
COLUMNS: list[tuple[str, tuple[str, ...], str, str | None]] = [
    ("site", ("rig", "site"), "Site, from the rig file", None),
    ("rig", ("rig", "rig"), "Rig at the site, from the rig file", None),
    ("pi", ("rig", "pi"), "PI's initials, from the rig file", None),
    ("modality", ("rig", "modality"), "Recording modality of the rig", None),
    ("date", ("start_time",), "When the rig measurement started, ISO 8601 with UTC offset", None),
    ("task", ("task",), f"{QUICK.task} (before a session) or {LONG.task} (measuring the "
     "rig): the BIDS task label", None),
    ("subject", ("subject",), "BIDS subject the rig measurement was filed under", None),
    ("session", ("session",), "BIDS session number", None),
    ("label", ("run", "label"), "BIDS run label of the rig measurement", None),
    ("attempt", ("run", "attempt"), "1, or 2, 3... for one filed again under its label",
     None),
    ("status", ("status",), "pass, or fail", None),
    ("failed_tests", ("failed_tests",), "The tests that failed, comma-separated", None),
    ("failure", ("failure",), "Why each failed, test: reason; ...", None),
    *[(f"{t}_status", ("tests", t, "status"), f"The {t} test: pass, fail, offline (read "
       "in the recording) or not run", None) for t in TESTS],
    ("fmri_gym_version", ("versions", "fmri_gym"), "fmri-gym package version", None),
    ("git_commit", ("versions", "git"), "git describe of the checkout (-dirty: edited)", None),
    ("os", ("platform", "system"), "Operating system", None),
    ("python", ("versions", "python"), "Python version", None),
    ("cpu", ("machine", "cpu"), "Processor", None),
    ("cores", ("machine", "cores"), "Processor cores", None),
    ("memory_gb", ("machine", "memory_gb"), "Memory", "GB"),
    ("gpus", ("machine", "gpus"), "Graphics cards, separated by ;", None),
    ("monitor", ("rig", "monitor"), "Monitor or projector, from the rig file", None),
    ("photodiode", ("rig", "photodiode"), "Photodiode, from the rig file", None),
    ("audio_path", ("rig", "audio_path"), "Sound path to the ear, from the rig file", None),
    ("trigger_hardware", ("rig", "trigger_hardware"), "Trigger hardware, from the rig file",
     None),
    ("display_driver", ("summaries", "display", "display", "driver"), "SDL video driver", None),
    ("display_size", ("summaries", "display", "display", "size"), "Window size obtained", "px"),
    ("fullscreen", ("summaries", "display", "display", "fullscreen"), "Fullscreen window", None),
    ("refresh_hz", ("summaries", "display", "display", "refresh_rate"), "Refresh rate reported", "Hz"),
    ("vsync", ("summaries", "display", "display", "vsync"), "Flips wait for the vertical blank", None),
    ("flip_n", ("summaries", "display", "flips", "n"), "Flip intervals measured", None),
    ("flip_mean_ms", ("summaries", "display", "flips", "mean_ms"), "Mean flip interval", "ms"),
    ("flip_sd_ms", ("summaries", "display", "flips", "sd_ms"), "SD of the flip intervals", "ms"),
    ("flip_max_ms", ("summaries", "display", "flips", "max_ms"), "Longest flip interval", "ms"),
    ("flip_late", ("summaries", "display", "flips", "late"), "Intervals over 1.5 periods (missed refresh)",
     None),
    ("display_locked", ("summaries", "display", "locked"), "Flips locked to the refresh", None),
    ("trigger_backend", ("summaries", "triggers", "triggers", "settings", "backend"),
     "Trigger output of the config: null, lsl, serial, parallel", None),
    ("sync_mode", ("summaries", "triggers", "triggers", "settings", "sync", "mode"),
     "Run start: wait (for the scanner), send, none", None),
    ("ports_writable", ("summaries", "triggers", "outputs", "n_writable"),
     "Serial/parallel ports with hardware that this user may write", None),
    ("lsl_probe", ("summaries", "triggers", "outputs", "lsl"), "LSL stream found and read back: OK or why not",
     None),
    ("codes_sent", ("summaries", "triggers", "sent", "n"), "Trigger codes sent", None),
    ("send_median_ms", ("summaries", "triggers", "sent", "send_median_ms"), "Median send call duration",
     "ms"),
    ("send_max_ms", ("summaries", "triggers", "sent", "send_max_ms"), "Longest send call", "ms"),
    ("lsl_readback", ("summaries", "triggers", "sent", "lsl_readback"),
     "LSL codes read back as sent: ok, differs, n/a (not LSL)", None),
    ("pulses_n", ("summaries", "triggers", "pulses", "n"), "Scanner pulses counted", None),
    ("tr_median_s", ("summaries", "triggers", "pulses", "tr_s"), "Median pulse interval", "s"),
    ("tr_train_s", ("summaries", "triggers", "pulses", "tr_train_s"),
     "Mean pulse interval over the train, on this PC's clock", "s"),
    ("pulse_sd_ms", ("summaries", "triggers", "pulses", "sd_ms"), "SD of the pulse intervals", "ms"),
    ("pulses_missed", ("summaries", "triggers", "pulses", "missed"), "Intervals over 1.5 TR", None),
    ("pulses_doubled", ("summaries", "triggers", "pulses", "doubles"), "Intervals under 0.5 TR", None),
    ("controls_keys", ("summaries", "controls", "n_keys"), "Keys asked for", None),
    ("controls_pressed", ("summaries", "controls", "n_pressed"), "Keys pressed when asked",
     None),
    ("controls_latency_ms", ("summaries", "controls", "median_latency_ms"),
     "Median time from the prompt to the press (a person's, not the device's)", "ms"),
    ("pad_device", ("summaries", "controls", "pad", "device"),
     "The controller found, whose buttons are read as keys", None),
    ("pad_axis_on", ("summaries", "controls", "pad", "axes", ControlNames.CONTROLLER_AXIS_LEFTX, "on"),
     "Fraction of the left stick's horizontal travel that counts as pushed", None),
    ("pad_corrections", ("summaries", "controls", "pad", "corrections"),
     "Wrong controls pressed, and asked again, while the calibration confirmed each", None),
    ("input_devices", ("summaries", "controls", "devices"),
     "Keyboards, button boxes and joysticks plugged in", None),
    ("frames_blocks", ("summaries", "frames", "played"),
     "Frame-rate blocks played, as fps/load", None),
    ("frames_late", ("summaries", "frames", "late_total"),
     "Frames shown a refresh later than their rate allows, all blocks", None),
    ("frames_lost", ("summaries", "frames", "lost_total"), "Frames not played, all blocks",
     None),
    ("frames_pacing_resets", ("summaries", "frames", "pacing_resets_total"),
     "Stalls that restarted the frame schedule, all blocks", None),
    ("frames_triggers_ok", ("summaries", "frames", "triggers_ok"),
     "Every block's frame triggers marked every frame, in their cycle", None),
    ("diode_readout", ("summaries", "photodiode", "readout"),
     "Where the diode was read: soundcard (here) or recording (offline)", None),
    ("flashes_n", ("summaries", "photodiode", "n"), "Flashes shown", None),
    ("photon_matched", ("summaries", "photodiode", "offset", "n_matched"), "Flashes the diode saw", None),
    ("photon_median_ms", ("summaries", "photodiode", "offset", "median_ms"), "Flip to photodiode, median",
     "ms"),
    ("photon_sd_ms", ("summaries", "photodiode", "offset", "sd_ms"), "Flip to photodiode, SD", "ms"),
    ("photon_min_ms", ("summaries", "photodiode", "offset", "min_ms"), "Flip to photodiode, min", "ms"),
    ("photon_max_ms", ("summaries", "photodiode", "offset", "max_ms"), "Flip to photodiode, max", "ms"),
    ("photon_drift_ms_per_min", ("summaries", "photodiode", "offset", "drift_ms_per_min"),
     "Flip to photodiode, trend over the run (runs of a minute or more)", "ms/min"),
    ("audio_device", ("summaries", "photodiode", "audio_out", "device"), "Sound output device", None),
    ("audio_delay_ms", ("summaries", "photodiode", "audio_out", "delay_ms"),
     "Delay from flip to DAC the audio output chose", "ms"),
    ("click_dac_median_ms", ("summaries", "photodiode", "click_dac_offset", "median_ms"),
     "Flip to click at the DAC, median", "ms"),
    ("sound_matched", ("summaries", "photodiode", "sound_offset", "n_matched"), "Clicks the microphone heard",
     None),
    ("sound_median_ms", ("summaries", "photodiode", "sound_offset", "median_ms"),
     "Flip to sound at the microphone, median", "ms"),
    ("sound_sd_ms", ("summaries", "photodiode", "sound_offset", "sd_ms"), "Flip to sound, SD", "ms"),
    ("sound_min_ms", ("summaries", "photodiode", "sound_offset", "min_ms"), "Flip to sound, min", "ms"),
    ("sound_max_ms", ("summaries", "photodiode", "sound_offset", "max_ms"), "Flip to sound, max", "ms"),
    ("sound_drift_ms_per_min", ("summaries", "photodiode", "sound_offset", "drift_ms_per_min"),
     "Flip to sound, trend over the run (runs of a minute or more)", "ms/min"),
    ("acoustic_median_ms", ("summaries", "photodiode", "acoustic_offset", "median_ms"),
     "DAC to sound at the microphone (amplifier, tubes, earphones), median", "ms"),
    ("acoustic_sd_ms", ("summaries", "photodiode", "acoustic_offset", "sd_ms"),
     "DAC to sound, SD", "ms"),
    ("av_median_ms", ("summaries", "photodiode", "av_offset", "median_ms"),
     "Photodiode to sound (audio behind video), median", "ms"),
    ("av_sd_ms", ("summaries", "photodiode", "av_offset", "sd_ms"), "Photodiode to sound, SD", "ms"),
]
#: The table's column names, in order.
NAMES = tuple(c[0] for c in COLUMNS)


def finish(rig_measurement: RigMeasurement) -> bool:
    """File the rig measurement: its manifest, the report, the pooling row, the rig's data
    root's ``rigmeasurements.tsv`` rebuilt, and what it measured beside the rig file.

    :param rig_measurement: the rig measurement, its measurements taken (or stopped).
    :return: ``True`` if a test failed or did not finish (the report lists them).
    """
    tests = {test: {"status": "fail", "why": "did not finish (stopped before it)"}
             for test in TESTS}
    for entry in rig_measurement.measurements.values():
        tests.update(entry["tests"])
    m = {"subject": SUBJECT, "session": rig_measurement.session,
         "task": rig_measurement.settings.task,
         "run": {"label": rig_measurement.label, "attempt": rig_measurement.attempt},
         "start_time": datetime.datetime.fromtimestamp(rig_measurement.start_epoch).astimezone()
         .isoformat(timespec="seconds"),
         "rig": rig_measurement.rig.record(), "settings": rig_measurement.settings.record(),
         "trigger_error": rig_measurement.trigger_error,
         "measurements": rig_measurement.measurements,
         "summaries": {name: e["summary"] for name, e in rig_measurement.measurements.items()},
         "tests": tests, "versions": versions(), "platform": platform_info(),
         "machine": machine()}
    data_root = rig_measurement.rig.data_root
    write_outputs(rig_measurement.folder, m, data_root)
    file_measured(m, rig_measurement.folder, rig_measurement.rig.path)
    verdict = "PASS" if m["status"] == "pass" else f"FAIL ({m['failed_tests']})"
    print(f"rig measurement {verdict}: {os.path.join(rig_measurement.folder, 'report.html')} "
          f"(all rig measurements: {os.path.join(data_root, 'rigmeasurements.tsv')})",
          file=sys.stderr)
    return m["status"] != "pass"


# ---------------------------------------------------------------------------
# Filing
# ---------------------------------------------------------------------------


def write_outputs(folder: str, m: dict, data_root: str | None = None) -> None:
    """The verdict from ``m["tests"]``; then the manifest, the TSV row, its sidecar, the
    reports (``report.md``, ``report.html``), and the data root's integrated table.

    :param m: the rig measurement's manifest with ``tests``, ``rig`` and ``summaries``; the
        overall ``status``, ``failed_tests`` and ``failure`` are set here.
    :param data_root: the BIDS tree the rig measurement was filed in: its ``rigmeasurements.tsv``
        (every rig measurement under it, one row each) is rebuilt, and the HTML report
        compares this rig measurement with the rig's previous ones.
    """
    bad = [t for t, v in m["tests"].items() if v["status"] == "fail"]
    m["failed_tests"] = ",".join(bad) or None
    m["failure"] = "; ".join(f"{t}: {m['tests'][t]['why']}" for t in bad) or None
    m["status"] = "fail" if bad else "pass"
    _write_json(os.path.join(folder, "manifest.json"), m)
    with open(os.path.join(folder, "rigmeasurement.tsv"), "w") as f:
        f.write("\t".join(c[0] for c in COLUMNS) + "\n" + "\t".join(row(m)) + "\n")
    _write_json(os.path.join(folder, "rigmeasurement.json"), sidecar())
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write(report(m))
    history = rows([data_root]) if data_root else [dict(zip(NAMES, row(m)))]
    if data_root:
        write_table(os.path.join(data_root, "rigmeasurements.tsv"), history)
    with open(os.path.join(folder, "report.html"), "w") as f:
        f.write(render_html(m, folder, history))


def write_table(path: str, table: list[dict[str, str]]) -> None:
    """A TSV of rows (:func:`rows`) with its BIDS sidecar (``.json``) beside it."""
    with open(path, "w") as f:
        f.write("\n".join("\t".join(r) for r in _lines(table)) + "\n")
    _write_json(os.path.splitext(path)[0] + ".json", sidecar())


def _lines(table: list[dict[str, str]]) -> list[list[str]]:
    return [list(NAMES), *([r[n] for n in NAMES] for r in table)]


def rows(roots: list[str]) -> list[dict[str, str]]:
    """Every rig measurement filed under ``roots``, one row each, oldest first, read from each
    one's ``manifest.json``."""
    out = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "manifest.json"), recursive=True)):
            with open(path) as f:
                m = json.load(f)
            if "summaries" in m:
                out.append(dict(zip(NAMES, row(m))))
    return sorted(out, key=lambda r: r["date"])


def row(manifest: dict) -> list[str]:
    """The manifest's values for :data:`COLUMNS`, as TSV cells."""
    return [_cell(_get(manifest, path)) for _, path, _, _ in COLUMNS]


def sidecar() -> dict[str, dict[str, str]]:
    """The BIDS-style description of the TSV's columns."""
    out = {}
    for name, _, description, units in COLUMNS:
        out[name] = {"Description": description, **({"Units": units} if units else {})}
    return out


def pool(roots: list[str]) -> list[str]:
    """Every rig measurement under ``roots`` (:func:`rows`), as the lines of one TSV."""
    return ["\t".join(r) for r in _lines(rows(roots))]


def _get(d: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _cell(v: Any) -> str:
    """A TSV cell: ``n/a`` for nothing, BIDS-style; no tab or newline inside."""
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return format(v, ".6g")
    if isinstance(v, list):  # a size reads 1024x768; names read a; b
        numbers = all(isinstance(x, (int, float)) for x in v)
        return ("x" if numbers else "; ").join(map(str, v))
    return re.sub(r"\s+", " ", str(v))


def versions() -> dict[str, str]:
    """What ran the rig measurement: fmri-gym, its git checkout, and the libraries timing
    depends on."""
    from importlib.metadata import version

    import numpy
    import pygame
    try:
        git = subprocess.run(["git", "-C", _REPO, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        git = ""
    return {"fmri_gym": version("fmri-gym"), "git": git or "n/a",
            "python": platform.python_version(), "numpy": numpy.__version__,
            "pygame": pygame.version.ver, "sdl": ".".join(map(str, pygame.get_sdl_version()))}


def platform_info() -> dict[str, str]:
    """The machine: OS, architecture, host name."""
    return {"system": platform.platform(), "machine": platform.machine(),
            "hostname": socket.gethostname()}


def machine() -> dict[str, Any]:
    """The machine a rig measurement ran on, and what is plugged into it: for provenance, and to
    tell two rigs (or one rig before and after a change) apart when their numbers differ.

    Each part is read from the operating system without new dependencies, and is
    ``None`` (or empty) where the system does not say.
    """
    displays, inputs = _pygame_devices()
    return {**platform_info(), "python": platform.python_version(), "cpu": _cpu(),
            "cores": os.cpu_count(), "memory_gb": _memory_gb(), "displays": displays,
            "gpus": _gpus(), "audio_devices": _audio_devices(),
            "input_devices": inputs, "usb_devices": _usb_devices(),
            "trigger_ports": detect_outputs()}


def _pygame_devices() -> tuple[list[dict[str, Any]], list[str]]:
    """The monitors and the input devices, read with pygame's display up (it opens no
    window); pygame's display is left as it was found."""
    was_up = pygame.display.get_init()
    try:
        pygame.display.init()
        displays = _displays()
        pygame.joystick.init()
        settled = time.monotonic() + 0.5  # controllers arrive as events, ~0.1 s after init
        while time.monotonic() < settled:
            pygame.event.pump()
            time.sleep(0.05)
    except pygame.error:  # no screen here: no monitor to list
        displays = []
    try:
        return displays, input_devices()
    finally:
        if not was_up:
            pygame.joystick.quit()
            pygame.display.quit()


def _run(*command: str) -> str:
    """A system tool's output; empty when it is missing, fails or hangs."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout if out.returncode == 0 else ""


def _cpu() -> str | None:
    if sys.platform == "darwin":
        return _run("sysctl", "-n", "machdep.cpu.brand_string").strip() or None
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            model = re.search(r"^model name\s*:\s*(.+)$", f.read(), re.MULTILINE)
        if model:
            return model[1].strip()
    return platform.processor() or None


def _memory_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _displays() -> list[dict[str, Any]]:
    """Every monitor the system has, its desktop size and refresh rate."""
    sizes = pygame.display.get_desktop_sizes()
    rates = (pygame.display.get_desktop_refresh_rates()
             if hasattr(pygame.display, "get_desktop_refresh_rates") else [None] * len(sizes))
    return [{"size": f"{w}x{h}", "refresh_hz": rate} for (w, h), rate in zip(sizes, rates)]


def _gpus() -> list[str]:
    if sys.platform == "darwin":
        out = _run("system_profiler", "SPDisplaysDataType", "-json")
        try:
            return [g.get("sppci_model", g.get("_name", "?"))
                    for g in json.loads(out)["SPDisplaysDataType"]] if out else []
        except (ValueError, KeyError, TypeError):
            return []
    lines = _run("lspci", "-mm").splitlines()
    return [line for line in lines if re.search(r'"(VGA|3D|Display)', line)]


def _audio_devices() -> list[dict[str, Any]]:
    """Every sound device, with its channels and default rate, the defaults marked."""
    try:
        import sounddevice
        devices = sounddevice.query_devices()
        default_in, default_out = sounddevice.default.device
    except Exception:  # noqa: BLE001 -- no PortAudio: nothing to list
        return []
    return [{"name": d["name"], "hostapi": sounddevice.query_hostapis(d["hostapi"])["name"],
             "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
             "default_samplerate": d["default_samplerate"],
             "default": [k for k, i in (("input", default_in), ("output", default_out))
                         if i == d["index"]]}
            for d in devices]


def _usb_devices() -> list[str]:
    """What is plugged into USB: maker, product and vendor:product ids."""
    if os.path.isdir("/sys/bus/usb/devices"):
        devices = sorted(glob.glob("/sys/bus/usb/devices/*"))
        return [f"{_sysfs(d, 'manufacturer')} {_sysfs(d, 'product')} "
                f"({_sysfs(d, 'idVendor')}:{_sysfs(d, 'idProduct')})".strip()
                for d in devices if _sysfs(d, "product")]
    if sys.platform == "darwin":
        # macOS 15 moved the USB tree to SPUSBHostDataType; earlier ones have SPUSBDataType.
        for kind in ("SPUSBHostDataType", "SPUSBDataType"):
            out = _run("system_profiler", kind, "-json")
            try:
                devices = _usb_items(json.loads(out)[kind]) if out else []
            except (ValueError, KeyError, TypeError):
                devices = []
            if devices:
                return devices
    return []


def _sysfs(device: str, name: str) -> str:
    """One attribute of a Linux USB device; empty when it has none."""
    try:
        with open(os.path.join(device, name)) as f:
            return f.read().strip()
    except OSError:
        return ""


def _usb_items(items: list[dict]) -> list[str]:
    """The devices in macOS's USB tree, depth first (either report's keys)."""
    out = []
    for item in items:
        vendor = item.get("vendor_id", item.get("USBDeviceKeyVendorID"))
        product = item.get("product_id", item.get("USBDeviceKeyProductID"))
        if vendor or product:
            maker = item.get("manufacturer", item.get("USBDeviceKeyVendorName", ""))
            out.append(f"{maker} {item.get('_name', '?')} ({vendor}:{product})".strip())
        out += _usb_items(item.get("_items", []))
    return out


def measured(m: dict, folder: str) -> dict[str, Any]:
    """What a rig measurement measured on its rig, as filed beside the rig file.

    ``latency`` holds when what a frame shows reaches the participant, from its
    flip: each the rig measurement's summary of its offsets (median, spread, count), and
    ``None`` when the rig measurement could not measure it (no photodiode, no microphone,
    a diode read in the MEG/EEG recording) or its test did not pass. The rest is
    how it was measured and on what: the rig measurement, its verdicts, every
    measurement's summary, the rig file as it played, the machine and the software.

    :param m: the rig measurement's manifest, filed (:func:`write_outputs`).
    :param folder: the rig measurement's folder.
    """
    photodiode = m["summaries"].get("photodiode", {})

    def latency(test: str, key: str) -> dict | None:
        summary = photodiode.get(key)
        passed = m["tests"][test]["status"] == "pass"
        return summary if passed and summary and "median_ms" in summary else None

    return {"date": m["start_time"],
            "rig_measurement": {"folder": folder, "label": m["run"]["label"],
                                "task": m["task"], "status": m["status"],
                                "failed_tests": m["failed_tests"], "tests": m["tests"]},
            "latency": {"flip_to_photon": latency("photodiode", "offset"),
                        "flip_to_dac": latency("audio", "click_dac_offset"),
                        "dac_to_ear": latency("audio", "acoustic_offset")},
            "summaries": m["summaries"], "rig": m["rig"], "machine": m["machine"],
            "versions": m["versions"]}


def file_measured(m: dict, folder: str, rig_path: str) -> str | None:
    """Write what the rig measurement measured beside the rig file
    (:func:`fmri_gym.config.rig.next_measured_latency_file`); nothing when it measured
    no latency.

    :param m: the rig measurement's manifest, filed.
    :param folder: the rig measurement's folder.
    :param rig_path: the rig file it set up.
    :return: the file written, or ``None``.
    """
    values = measured(m, folder)
    if not any(values["latency"].values()):
        print("rig measurement: no latency measured (no photodiode or microphone read, or their "
              "tests failed): nothing filed beside the rig", file=sys.stderr)
        return None
    path = rig_file.next_measured_latency_file(rig_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_json(path, values)
    print(f"rig measurement: what it measured on the rig: {path}", file=sys.stderr)
    return path


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(m: dict) -> str:
    """``report.md``: the verdicts, then each test's numbers, for a person to read."""
    rig = m["rig"]
    verdict = "PASS" if m["status"] == "pass" else f"FAIL ({m['failed_tests']})"
    lines = [f"# Rig measurement {rig['site']}/{rig['rig']}: {verdict}", "",
             f"{m['start_time']} · `{m['run']['label']}` "
             f"(attempt {m['run']['attempt']}) · fmri-gym {m['versions']['fmri_gym']} "
             f"(`{m['versions']['git']}`)", "",
             "| test | verdict | why |", "|---|---|---|"]
    for test, v in m["tests"].items():
        status = f"**{v['status'].upper()}**" if v["status"] == "fail" else v["status"]
        lines.append(f"| {test} | {status} | {_cell(v['why']) if v['why'] else ''} |")
    lines += ["", "## Rig", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {_cell(v)} |" for k, v in rig.items()]
    for test, section in (("display", _md_display), ("frames", _md_frames),
                          ("triggers", _md_triggers), ("controls", _md_controls),
                          ("photodiode", _md_photodiode), ("audio", _md_audio)):
        status = m["tests"][test]["status"]
        lines += ["", f"## {test.capitalize()}: {status}", ""]
        measured = m["summaries"].get(TESTS[test])
        if measured:
            lines += section(measured)
    lines += ["", "## This computer"]
    for title, head, table in computer(m):
        lines += ["", f"### {title}", "", "| " + " | ".join(head) + " |",
                  "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in table]
    return "\n".join(lines) + "\n"


def computer(m: dict) -> list[tuple[str, list[str], list[list[Any]]]]:
    """The computer a rig measurement ran on and what was plugged into it, as titled tables.

    :param m: the rig measurement's manifest.
    :return: ``(title, column names, rows)`` per table; a table with nothing found says so.
    """
    mc, p = m["machine"], m["platform"]
    none = [["none found"]]
    return [
        ("Computer", ["", ""], [[k, v] for k, v in (
            ("system", p.get("system")), ("architecture", p.get("machine")),
            ("host", p.get("hostname")), ("processor", mc.get("cpu")), ("cores", mc.get("cores")),
            ("memory (GB)", mc.get("memory_gb")), ("graphics", mc.get("gpus")))]),
        ("Screens", ["screen", "size", "refresh (Hz)"],
         [[i, d.get("size"), d.get("refresh_hz")] for i, d in enumerate(mc.get("displays", []))]
         or none),
        ("Sound devices", ["name", "system", "inputs", "outputs", "rate (Hz)", "default for"],
         [[d.get("name"), d.get("hostapi"), d.get("inputs"), d.get("outputs"),
           d.get("default_samplerate"), d.get("default")] for d in mc.get("audio_devices", [])]
         or none),
        ("Response devices (keyboards, button boxes, controllers)", ["name"],
         [[n] for n in mc.get("input_devices", [])] or none),
        ("Trigger ports", ["kind", "port", "what", "writable"],
         [[o.get("kind"), port_name(o), o.get("what"), o.get("writable")]
          for o in mc.get("trigger_ports", [])] or none),
        ("USB devices", ["device"], [[d] for d in mc.get("usb_devices", [])] or none),
        ("Software", ["", "version"], [[k, v] for k, v in m["versions"].items()]),
    ]


def _md_display(d: dict) -> list[str]:
    s, f = d["display"], d["flips"]
    return [f"- {s['driver']}, {_cell(s['size'])} px, fullscreen {_cell(s['fullscreen'])}, "
            f"monitor {s['monitor']}, {s['refresh_rate']} Hz, vsync {_cell(s['vsync'])}",
            f"- {f['n']} flips: mean {f['mean_ms']:.2f} ms, SD {f['sd_ms']:.2f}, "
            f"max {f['max_ms']:.2f}; missed a refresh: {f['late']}",
            f"- **{'locked to the refresh' if d['locked'] else 'NOT locked to the refresh'}**"]


def _md_triggers(t: dict) -> list[str]:
    o = t["outputs"]
    lines = [f"- ports: {', '.join(map(port_name, o['ports'])) or 'none with hardware'} "
             f"({o['n_writable']} writable); LSL: {o['lsl']}"]
    if not (o["n_writable"] or o["lsl"] == "OK"):
        lines.append("- **warning: no usable trigger output on this PC**")
    if "triggers" not in t:
        return lines
    s = t["triggers"]["settings"]
    lines.append(f"- config: backend {s['backend']} ({t['triggers']['active']}), "
                 f"start {s['sync']['mode']}")
    if "sent" in t:
        x = t["sent"]
        check = ("read back as sent" if x["lsl_readback"] == "ok"
                 else "**check them in the recording**")
        lines.append(f"- sent {x['n']} codes, send call median {x['send_median_ms']:.3f} ms, "
                     f"max {x['send_max_ms']:.3f} ms: {check}")
    if "pulses" in t:
        p = t["pulses"]
        lines.append(f"- scanner pulses: {p['n']}, TR {p['tr_s']:.3f} s (median), "
                     f"{p['tr_train_s']:.6f} s over the train, SD {p['sd_ms']:.1f} ms, "
                     f"missed {p['missed']}, doubled {p['doubles']}")
    return lines


def _md_frames(f: dict) -> list[str]:
    lines = [f"- the test pattern played at each rate, under each load, paced as a game; "
             f"refresh {f['refresh_rate']} Hz"
             + (f", recording at {f['recording_hz']:g} Hz" if f.get("recording_hz") else "")]
    if not f.get("vsync"):
        lines.append("- **vsync off: frames were paced by the clock alone, not the refresh "
                     "(see the display test); late frames cannot be told from jitter**")
    lines += ["", "| fps | load | frames | lost | late | pacing resets "
                  "| interval median / SD / max ms | frame triggers |",
              "|---|---|---|---|---|---|---|---|"]
    for b in f["blocks"]:
        t = b["triggers"]
        marks = ("off" if t is None else f"{t['marked']}/{t['expected']}, "
                 f"{t['code_ms']:.1f} ms each" + ("" if t["ok"] else f" **{t['why']}**"))
        fits = " (does not divide the refresh)" if b["divides_refresh"] is False else ""
        ms = " / ".join("n/a" if b[k] is None else f"{b[k]:.2f}"
                        for k in ("interval_median_ms", "interval_sd_ms", "interval_max_ms"))
        lines.append(f"| {b['fps']:g}{fits} | {b['load']} | {b['n']}/{b['expected']} | "
                     f"{b['lost']} | {b['late']} | {b['pacing_resets']} | {ms} | {marks} |")
    return lines


def _md_controls(c: dict) -> list[str]:
    lines = [f"- input devices: {', '.join(c['devices']) or 'none listed'}",
             f"- {_pad_line(c.get('pad') or {})}"]
    for key, auto in c["automatic"].items():
        p = c["prompted"].get(key)
        asked = ("skipped" if p is None else
                 f"pressed {p['latency_s']:.2f} s after asked, held "
                 f"{p['held_s'] * 1000:.0f} ms" 
                 if p["pressed"] else "**never pressed**")
        strays = f" (other keys: {', '.join(p['strays'])})" if p and p["strays"] else ""
        lines.append(f"- `{key}`: {'read back' if auto else '**not read back**'}; "
                     f"{asked}{strays}")
    return lines


def _diode_text(readout: str) -> str:
    """How the photodiode was read, in words."""
    return {"soundcard": "the diode read on the sound card",
            "recording": "the diode read in the recording (match offline)"}.get(
                readout, "no photodiode on this rig")


def _md_photodiode(p: dict) -> list[str]:
    lines = [f"- {p['n']} flashes of {p['on_ms']:g} ms, {_diode_text(p['readout'])}"]
    return lines + _md_offsets(p, (("offset", "flip → photodiode"),))


def _md_audio(p: dict) -> list[str]:
    if "audio_out" not in p:
        return ["- the audio output did not open"]
    a = p["audio_out"]
    lines = [f"- a tone burst on every other flash through {a.get('device')}, "
             f"delay chosen {a.get('delay_ms', 0):.1f} ms; microphone: "
             f"{'on input 1' if p.get('mic') else 'none'}"]
    return lines + _md_offsets(p, (("click_dac_offset", "flip → click at the DAC"),
                                ("sound_offset", "flip → sound at the microphone"),
                                ("acoustic_offset", "DAC → sound at the microphone"),
                                ("av_offset", "photodiode → sound")))


def _md_offsets(p: dict, keys: tuple[tuple[str, str], ...]) -> list[str]:
    """A table of the offsets under ``keys`` that were measured."""
    rows = [(label, p[key]) for key, label in keys if key in p]
    if not rows:
        return []
    lines = ["", "| offset | median ms | SD | min | max | drift ms/min | seen |",
             "|---|---|---|---|---|---|---|"]
    for label, o in rows:
        if not o.get("n_matched"):
            lines.append(f"| {label} | none seen | | | | | 0/{o['n']} |")
            continue
        lines.append(f"| {label} | {o['median_ms']:.2f} | {o['sd_ms']:.2f} | "
                     f"{o['min_ms']:.2f} | {o['max_ms']:.2f} | "
                     f"{_cell(o.get('drift_ms_per_min'))} | {o['n_matched']}/{o['n']} |")
    return lines


# ---------------------------------------------------------------------------
# report.html
# ---------------------------------------------------------------------------


_W, _H = 680, 220  # a chart's box, before its margins
_M = {"l": 52, "r": 16, "t": 12, "b": 34}

_CSS = """
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --good:#0ca30c; --critical:#d03b3b; --good-ink:#006300; --critical-ink:#b42f2f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --good-ink:#0ca30c; --critical-ink:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --good-ink:#0ca30c; --critical-ink:#e66767; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 32px 16px 64px; }
header .meta { color: var(--ink-2); font-size: 13px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
h3 { font-size: 15px; margin: 20px 0 6px; font-weight: 600; }
h2 { font-size: 17px; margin: 40px 0 8px; font-weight: 600; display:flex; gap:10px;
  align-items:center; }
.hero { font-size: 48px; font-weight: 600; line-height: 1.1; margin: 20px 0 4px; }
.card { background: var(--surface); border-radius: 10px; box-shadow: 0 0 0 1px var(--ring);
  padding: 16px; margin: 12px 0; overflow-x: auto; }
.note { color: var(--ink-2); font-size: 13px; margin: 6px 0; }
.status { display:inline-flex; align-items:center; gap:6px; font-weight:600; font-size:13px;
  white-space: nowrap; }
.status i { font-style: normal; width:18px; height:18px; border-radius:50%;
  display:inline-grid; place-items:center; color:#fff; font-size:12px; }
.pass i { background: var(--good); } .pass { color: var(--good-ink); }
.fail i { background: var(--critical); } .fail { color: var(--critical-ink); }
.other i { background: var(--axis); color: var(--ink); } .other { color: var(--ink-2); }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; }
.tile { background: var(--surface); border-radius: 10px; box-shadow: 0 0 0 1px var(--ring);
  padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 24px; font-weight: 600; }
.tile .sub { color: var(--muted); font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; color: var(--ink-2); font-weight: 500; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
details summary { cursor: pointer; color: var(--ink-2); font-size: 13px; margin-top: 8px; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--ink-2); margin: 4px 0 8px;
  flex-wrap: wrap; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px;
  border-radius:50%; margin-right:6px; background: var(--c); vertical-align: -1px; }
svg { display:block; max-width:100%; height:auto; }
svg text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg [data-tip] { cursor: default; outline: none; }
svg [data-tip]:hover, svg [data-tip]:focus { opacity: 0.75; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
  box-shadow: 0 0 0 1px var(--ring), 0 4px 16px rgba(0,0,0,0.12); border-radius: 8px;
  padding: 6px 10px; font-size: 13px; display: none; max-width: 280px; z-index: 2; }
code { font-size: 12px; }
"""

_JS = """
const tip = document.getElementById('tip');
function show(el, x, y) { tip.textContent = el.dataset.tip; tip.style.display = 'block';
  tip.style.left = Math.min(x + 14, innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = (y + 14) + 'px'; }
document.querySelectorAll('[data-tip]').forEach(el => {
  el.setAttribute('tabindex', '0');
  el.addEventListener('pointermove', e => show(el, e.clientX, e.clientY));
  el.addEventListener('pointerleave', () => tip.style.display = 'none');
  el.addEventListener('focus', () => { const r = el.getBoundingClientRect();
    show(el, r.left, r.bottom); });
  el.addEventListener('blur', () => tip.style.display = 'none');
});
"""


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _num(v: Any, fmt: str = ".2f", unit: str = "") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:{fmt}}{unit}"


def _status(status: str) -> str:
    kind, icon = {"pass": ("pass", "✓"), "fail": ("fail", "✕")}.get(status, ("other", "–"))
    return f'<span class="status {kind}"><i aria-hidden="true">{icon}</i>{_e(status)}</span>'


# ---------------------------------------------------------------------------
# Charts (inline SVG)
# ---------------------------------------------------------------------------


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering ``lo..hi``."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    step = 10 ** np.floor(np.log10(raw))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if m * step >= raw)
    start = np.floor(lo / step) * step
    count = int(np.ceil((hi - start) / step - 1e-9)) + 1  # the last tick at or past hi
    return [float(start + i * step) for i in range(count)]


def _frame(xt: Sequence[float], yt: Sequence[float], sx, sy, xlabel: str,
           ylabel: str) -> list[str]:
    """Gridlines, axis and tick labels of a chart."""
    out = []
    for y in yt:
        out.append(f'<line class="grid" x1="{_M["l"]}" x2="{_M["l"] + _W}" y1="{sy(y):.1f}" '
                   f'y2="{sy(y):.1f}"/><text x="{_M["l"] - 6}" y="{sy(y) + 4:.1f}" '
                   f'text-anchor="end">{y:g}</text>')
    base = _M["t"] + _H
    out.append(f'<line class="axis" x1="{_M["l"]}" x2="{_M["l"] + _W}" y1="{base}" y2="{base}"/>')
    for x in xt:
        out.append(f'<text x="{sx(x):.1f}" y="{base + 16}" text-anchor="middle">{x:g}</text>')
    out.append(f'<text x="{_M["l"] + _W / 2}" y="{base + 31}" text-anchor="middle">'
               f'{_e(xlabel)}</text>')
    out.append(f'<text x="12" y="{_M["t"] + _H / 2}" text-anchor="middle" '
               f'transform="rotate(-90 12 {_M["t"] + _H / 2})">{_e(ylabel)}</text>')
    return out


def _svg(parts: list[str], label: str) -> str:
    w, h = _M["l"] + _W + _M["r"], _M["t"] + _H + _M["b"]
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(label)}">'
            + "".join(parts) + "</svg>")


def _span(values: np.ndarray) -> tuple[float, float, int]:
    """The 0.5th-99.5th percentile range, padded, and how many values fall outside it:
    a stray value would otherwise squash the rest into a line."""
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    return float(lo), float(hi), int(np.sum((values < lo) | (values > hi)))


def _outside(n: int) -> str:
    return (f'<p class="note">{n} value{"s" if n > 1 else ""} beyond this range, left out of '
            f"the chart; the min and max are in the numbers given with it.</p>" if n else "")


def histogram(values: np.ndarray, unit: str, label: str, bins: int = 40) -> str:
    """Columns of how many values fall in each bin: one series, one hue."""
    values = values[np.isfinite(values)]
    if not len(values):
        return ""
    lo, hi, outside = _span(values)
    values = values[(values >= lo) & (values <= hi)]
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi + 1e-9))
    xt = _ticks(lo, hi)
    yt = _ticks(0, counts.max())
    sx = lambda v: _M["l"] + (v - xt[0]) / (xt[-1] - xt[0]) * _W  # noqa: E731
    sy = lambda v: _M["t"] + _H - v / yt[-1] * _H  # noqa: E731
    parts = _frame(xt, yt, sx, sy, unit, "count")
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        if not c:
            continue
        x0, x1 = sx(a) + 1, sx(b) - 1  # the 2px surface gap between neighbours
        w = min(24.0, max(1.0, x1 - x0))
        x0 += (x1 - x0 - w) / 2
        top, base = sy(c), _M["t"] + _H
        r = min(4.0, w / 2, base - top)
        parts.append(f'<path fill="var(--s1)" d="M{x0:.1f},{base} V{top + r:.1f} '
                     f'Q{x0:.1f},{top:.1f} {x0 + r:.1f},{top:.1f} H{x0 + w - r:.1f} '
                     f'Q{x0 + w:.1f},{top:.1f} {x0 + w:.1f},{top + r:.1f} V{base} Z" '
                     f'data-tip="{c} between {a:.2f} and {b:.2f} {_e(unit)}"/>')
    return _svg(parts, label) + _outside(outside)


def dots(series: list[tuple[str, str, np.ndarray, np.ndarray]], xlabel: str, ylabel: str,
         label: str) -> str:
    """Points per series, ``(name, colour var, x, y)``; at most three (the all-pairs limit)."""
    xs = np.concatenate([s[2] for s in series])
    ys = np.concatenate([s[3][np.isfinite(s[3])] for s in series])
    if not len(ys):
        return ""
    lo, hi, outside = _span(ys)
    xt, yt = _ticks(0, float(xs.max())), _ticks(lo, hi)
    # Many points: small and ringless, or the rings bury each other's fills.
    dense = len(ys) > 200
    r, ring = ("2.5", 'fill-opacity="0.8"') if dense else ("4", 'stroke="var(--surface)" '
                                                              'stroke-width="2"')
    sx = lambda v: _M["l"] + (v - xt[0]) / (xt[-1] - xt[0]) * _W  # noqa: E731
    sy = lambda v: _M["t"] + _H - (v - yt[0]) / (yt[-1] - yt[0]) * _H  # noqa: E731
    parts = _frame(xt, yt, sx, sy, xlabel, ylabel)
    for name, colour, x, y in series:
        for xi, yi in zip(x, y):
            if np.isfinite(yi) and lo <= yi <= hi:  # the rest is counted in _outside
                parts.append(f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="{r}" '
                             f'fill="var({colour})" {ring} '
                             f'data-tip="{_e(name)}, flash {int(xi) + 1}: {yi:.2f} ms"/>')
    return _svg(parts, label) + _outside(outside)


def sparkline(values: np.ndarray, width: int = 180, height: int = 28) -> str:
    """A line of ``values`` over time, down-sampled to the widest value per pixel."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return ""
    buckets = np.array_split(values, min(width, len(values)))
    peaks = np.array([b.max() for b in buckets])
    lo, hi = float(values.min()), float(peaks.max()) or 1.0
    span = hi - lo or 1.0
    pts = " ".join(f"{i / (len(peaks) - 1) * width:.1f},{height - 2 - (v - lo) / span * (height - 4):.1f}"
                   for i, v in enumerate(peaks))
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'aria-hidden="true"><polyline points="{pts}" fill="none" stroke="var(--s1)" '
            f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/></svg>')


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _html_npz(folder: str, pattern: str) -> Any:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    return np.load(paths[-1], allow_pickle=True) if paths else None


def _html_tiles(m: dict) -> str:
    rc = m["summaries"]
    tiles = []
    d = rc.get("display")
    if d:
        tiles.append(("Refresh", f"{d['display']['refresh_rate']} Hz",
                      "locked" if d["locked"] else "NOT locked"))
        tiles.append(("Flip interval SD", _num(d["flips"]["sd_ms"], ".2f", " ms"),
                      f"{d['flips']['late']} missed refreshes"))
    f = rc.get("frames")
    if f:
        tiles.append(("Late frames", f"{f['late_total']}",
                      f"{f['lost_total']} lost, {len(f['blocks'])} blocks"))
    p = rc.get("photodiode", {})
    if p.get("offset", {}).get("n_matched"):
        tiles.append(("Flip → photodiode", _num(p["offset"]["median_ms"], ".1f", " ms"),
                      f"median, SD {_num(p['offset']['sd_ms'])} ms"))
    if p.get("sound_offset", {}).get("n_matched"):
        tiles.append(("Flip → sound", _num(p["sound_offset"]["median_ms"], ".1f", " ms"),
                      "at the microphone"))
    elif p.get("click_dac_offset", {}).get("n_matched"):
        tiles.append(("Flip → sound out", _num(p["click_dac_offset"]["median_ms"], ".1f", " ms"),
                      "at the DAC"))
    t = rc.get("triggers", {})
    if "sent" in t:
        tiles.append(("Codes sent", str(t["sent"]["n"]),
                      "read back" if t["sent"]["lsl_readback"] == "ok" else "check the recording"))
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{_e(a)}</div><div class="value">{_e(b)}</div>'
        f'<div class="sub">{_e(c)}</div></div>' for a, b, c in tiles) + "</div>"


def _html_table(head: Sequence[str], rows: Sequence[Sequence[Any]], nums: Sequence[int] = ()) -> str:
    th = "".join(f'<th class="{"num" if i in nums else ""}">{_e(h)}</th>'
                 for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(
        f'<td class="{"num" if i in nums else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _html_display(m: dict, folder: str) -> str:
    d = m["summaries"]["display"]
    s, f = d["display"], d["flips"]
    out = [f'<p class="note">{_e(s["driver"])}, {_e("x".join(map(str, s["size"])))} px, '
           f'{"fullscreen" if s["fullscreen"] else "windowed"}, monitor {s["monitor"]}, '
           f'{s["refresh_rate"]} Hz, vsync {"on" if s["vsync"] else "off"}. {f["n"]} flips: '
           f'mean {f["mean_ms"]:.2f} ms, SD {f["sd_ms"]:.2f}, max {f["max_ms"]:.2f}; '
           f'{f["late"]} missed a refresh.</p>']
    z = _html_npz(folder, "display.npz")
    if z is not None:
        out.append('<p class="note">Flip intervals: one tight column at the refresh period '
                   'is a locked screen.</p>')
        out.append(histogram(np.asarray(z["intervals_ms"]), "ms between flips",
                             "Histogram of flip intervals"))
    return "".join(out)


def _html_section(section: Callable[[dict, str], str], m: dict, folder: str,
                  verdict: dict | None) -> str:
    """A measurement's section; for one that broke part-way and left its summary
    half-written (see :func:`~.measure._measure`), a note that it stopped, and why."""
    try:
        return section(m, folder)
    except (KeyError, IndexError, TypeError):
        why = (verdict or {}).get("why")
        return (f'<p class="note">This measurement stopped before it finished'
                f'{": " + _e(why) if why else ""}.</p>')


def _html_frames(m: dict, folder: str) -> str:
    f = m["summaries"]["frames"]
    out = [f'<p class="note">A test pattern played at each rate and load, paced as a game '
           f'block is; refresh {f["refresh_rate"]} Hz. Each line is that block\'s frame '
           f'intervals over its duration: flat is steady, spikes are late frames.</p>']
    if not f.get("vsync"):
        out.append('<p class="note"><b>vsync was off:</b> frames were paced by the clock '
                   'alone; late frames cannot be told from jitter.</p>')
    z = _html_npz(folder, "frames.npz")
    rows = []
    for i, b in enumerate(f["blocks"]):
        flips = f"flip_time_{i}"
        line = sparkline(np.diff(z[flips]) * 1000) if z is not None and flips in z else ""
        t = b["triggers"]
        marks = "off" if t is None else (f"{t['marked']}/{t['expected']}"
                                         + ("" if t["ok"] else f" — {_e(t['why'])}"))
        rows.append([f"{b['fps']:g}" + ("" if b["divides_refresh"] is not False else "*"),
                     _e(b["load"]), f"{b['n']}/{b['expected']}", b["lost"], b["late"],
                     b["pacing_resets"], _num(b["interval_median_ms"]),
                     _num(b.get("interval_sd_ms")), _num(b["interval_max_ms"]), line, marks])
    out.append(_html_table(["fps", "load", "frames", "lost", "late", "resets", "median ms", "SD ms",
                       "max ms", "intervals over the block", "frame triggers"], rows,
                      nums=(2, 3, 4, 5, 6, 7, 8)))
    out.append('<p class="note">* does not divide the refresh: frames alternate between two '
               'hold times by design.</p>')
    return "".join(out)


def _html_triggers(m: dict, folder: str) -> str:
    t = m["summaries"]["triggers"]
    o = t["outputs"]
    ports = ", ".join(map(port_name, o["ports"])) or "none with hardware"
    out = [f'<p class="note">This PC: {_e(ports)} ({o["n_writable"]} writable); LSL: '
           f'{_e(o["lsl"])}.</p>']
    if "triggers" in t:
        s = t["triggers"]["settings"]
        out.append(f'<p class="note">Config: start <b>{_e(s["sync"]["mode"])}</b>'
                   + (f' (scanner key {_e(repr(s["sync"]["key"]))})' if s["sync"]["mode"] == "wait"
                      else "") + f', codes over <b>{_e(t["triggers"]["active"])}</b>.</p>')
    rows = []
    if "sent" in t:
        x = t["sent"]
        rows.append(["codes sent", x["n"], f"send call median {x['send_median_ms']:.3f} ms, max "
                     f"{x['send_max_ms']:.3f} ms; " + ("all read back from the network"
                                                      if x["lsl_readback"] == "ok"
                                                      else "check each in the recording")])
    if "pulses" in t:
        p = t["pulses"]
        rows.append(["scanner pulses", p["n"], f"TR {p['tr_s']:.3f} s (median), "
                     f"{p['tr_train_s']:.6f} s over the train; missed {p['missed']}, "
                     f"doubled {p['doubles']}"])
    return "".join(out) + (_html_table(["", "n", ""], rows, nums=(1,)) if rows else "")


def _html_controls(m: dict, folder: str) -> str:
    c = m["summaries"]["controls"]
    devices = _e(", ".join(c["devices"]) or "none listed")
    out = [(f'<p class="note">Input devices: {devices}. '
            f'{_e(_pad_line(c.get("pad") or {}))}</p>')]
    rows = []
    for key, auto in c["automatic"].items():
        p = c["prompted"].get(key)
        asked = ("skipped" if p is None else
                 f"{p['latency_s']:.2f} s after asked, held {p['held_s'] * 1000:.0f} ms"
                 if p["pressed"] else "never pressed")
        others = ", ".join(p["strays"]) if p and p["strays"] else ""
        rows.append([f"<code>{_e(key)}</code>", "read back" if auto else "not read back",
                     _e(asked), _e(others)])
    return "".join(out) + _html_table(["key", "automatic", "when asked", "other keys"], rows)


def _pad_line(pad_out: dict) -> str:
    """One line on what the controls test found of the controller, for either report."""
    if not pad_out.get("present"):
        return "The rig lists no controller; the device in use must send the keys below itself."
    device = pad_out.get("device", "a controller")
    if not pad_out.get("measured"):
        return f"{device} plugged in, not measured: {pad_out.get('note') or pad_out.get('why')}."
    axes = "; ".join(f"{name} on past {a['on']:.2f} of its travel and off under {a['off']:.2f}"
                     for name, a in pad_out["axes"].items())
    return (f"{device} measured, written to {pad_out['config']}: {axes}; "
            f"{pad_out['confirmed']} controls "
            f"confirmed, {pad_out['corrections']} wrong presses asked again. Every run on this "
            "rig with it reads what was written.")


def _html_photodiode(m: dict, folder: str) -> str:
    p = m["summaries"]["photodiode"]
    out = [f'<p class="note">{p["n"]} flashes of {p["on_ms"]:g} ms; {_diode_text(p["readout"])}; '
           f'a click on every other flash'
           + (f' through {_e(p["audio_out"].get("device"))}, delay chosen '
              f'{p["audio_out"].get("delay_ms", 0):.1f} ms' if "audio_out" in p else "")
           + (", a microphone on input 1" if p.get("mic") else "") + ".</p>"]
    z = _html_npz(folder, "photodiode.npz")
    series = []
    if z is not None and "offset_s" in z:
        series.append(("flip → photodiode", "--s1", np.arange(len(z["offset_s"])),
                       z["offset_s"] * 1000))
    if z is not None and "click_dac" in z:
        idx = z["click_flash"]
        on = z["flip_on"][idx]
        if "sound_offset_s" in z:
            series.append(("flip → sound at the mic", "--s2", idx, z["sound_offset_s"] * 1000))
        series.append(("flip → click at the DAC", "--s3", idx, (z["click_dac"] - on) * 1000))
    if series:
        out.append('<div class="legend">' + "".join(
            f'<span style="--c:var({c})">{_e(n)}</span>' for n, c, _, _ in series) + "</div>")
        out.append(dots(series, "flash", "ms after the flip", "Offsets per flash"))
    rows = [[_e(label), f"{o['n_matched']}/{o['n']}", _num(o.get("median_ms")),
             _num(o.get("sd_ms")), _num(o.get("min_ms")), _num(o.get("max_ms")),
             _num(o.get("drift_ms_per_min"), ".3f")]
            for key, label in (("offset", "flip → photodiode"),
                               ("sound_offset", "flip → sound at the mic"),
                               ("click_dac_offset", "flip → click at the DAC"),
                               ("av_offset", "photodiode → sound"))
            if (o := p.get(key))]
    if rows:
        out.append(_html_table(["offset", "seen", "median ms", "SD", "min", "max", "drift ms/min"],
                          rows, nums=(1, 2, 3, 4, 5, 6)))
    return "".join(out)


def _cellnum(text: str | None, fmt: str = ".2f") -> str:
    """A TSV cell shown as a number, or as it is (``n/a``)."""
    try:
        return f"{float(text):{fmt}}"
    except (TypeError, ValueError):
        return _e(text)


def _html_history(m: dict, history: list[dict[str, str]]) -> str:
    rig = m["rig"]
    mine = [r for r in history if r.get("site") == rig["site"] and r.get("rig") == rig["rig"]]
    mine = sorted(mine, key=lambda r: r.get("date", ""))[-10:]
    if len(mine) < 2:
        return '<p class="note">No earlier rig measurement of this rig to compare with yet.</p>'
    rows = [[f'<span style="white-space:nowrap">{_e(r.get("date", "")[:16].replace("T", " "))}'
             "</span>", _e(r.get("task")), _status(r.get("status", "")),
             _cellnum(r.get("flip_sd_ms")), _e(r.get("frames_late")),
             _cellnum(r.get("photon_median_ms"), ".1f"), _cellnum(r.get("click_dac_median_ms"), ".1f"),
             _e((r.get("failed_tests") or "").replace(",", ", "))] for r in mine]
    return (f'<p class="note">The last {len(mine)} rig measurements of '
            f'{_e(rig["site"])}/{_e(rig["rig"])} '
            f'(this one last): today\'s numbers should sit with the others.</p>'
            + _html_table(["date", "task", "status", "flip SD ms", "late frames", "photodiode ms",
                      "sound out ms", "failed"], rows, nums=(3, 4, 5, 6)))


_html_SECTIONS = (("display", "Display", _html_display), ("frames", "Frames", _html_frames),
             ("triggers", "Triggers", _html_triggers), ("controls", "Controls", _html_controls),
             ("photodiode", "Photodiode and audio", _html_photodiode))


def render_html(m: dict, folder: str, history: list[dict[str, str]]) -> str:
    """The whole page for one finished rig measurement.

    :param m: its manifest, with ``tests``, ``rig`` and ``summaries``.
    :param folder: its folder, for the ``.npz`` the charts are drawn from.
    :param history: pooled rows (:func:`rows`) of the rig measurements under
        the same data root, this one included.
    """
    rig, tests = m["rig"], m["tests"]
    failed = [t for t, v in tests.items() if v["status"] == "fail"]
    verdict = "Every test passed" if not failed else f"{len(failed)} of {len(tests)} tests failed"
    body = [f'<header><h1>Rig measurement · {_e(rig["site"])} / {_e(rig["rig"])}</h1>'
            f'<div class="meta">{_e(m["start_time"])} · {_e(m["run"]["label"])} · PI '
            f'{_e(rig.get("pi", "?"))} · {_e(rig.get("modality", "?"))} · '
            f'fmri-gym {_e(m["versions"]["fmri_gym"])} '
            f'({_e(m["versions"]["git"])})</div></header>',
            f'<div class="hero">{_status("pass" if not failed else "fail")} {_e(verdict)}</div>',
            '<div class="card">' + _html_table(["test", "verdict", "why"], [
                [_e(t), _status(v["status"]), _e(v["why"] or "")] for t, v in tests.items()])
            + "</div>", _html_tiles(m)]
    for key, title, section in _html_SECTIONS:
        test = "audio" if key == "photodiode" and "photodiode" not in tests else key
        body.append(f'<h2>{_e(title)} {_status(tests.get(test, {}).get("status", "not run"))}'
                    + (f' {_status(tests["audio"]["status"])}' if key == "photodiode"
                       and "audio" in tests else "") + "</h2>")
        body.append('<div class="card">' + (_html_section(section, m, folder, tests.get(test))
                                            if key in m["summaries"]
                                            else '<p class="note">Not run.</p>') + "</div>")
    body.append(f'<h2>This rig over time</h2><div class="card">{_html_history(m, history)}</div>')
    body.append('<h2>Rig</h2><div class="card">' + _html_table(["", ""], [
        [_e(k), _e(v)] for k, v in rig.items()]) + "</div>")
    body.append("<h2>This computer</h2>")
    for title, head, table in computer(m):
        body.append(f'<h3>{_e(title)}</h3><div class="card">'
                    + _html_table(head, [[_e(_cell(c)) for c in row] for row in table])
                    + "</div>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Rig measurement {_e(rig['site'])}/{_e(rig['rig'])}</title>"
            f"<style>{_CSS}</style></head><body><main>{''.join(body)}</main>"
            f'<div id="tip" role="tooltip"></div><script>{_JS}</script></body></html>\n')
