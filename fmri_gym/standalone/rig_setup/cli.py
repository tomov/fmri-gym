"""``rig-setup``: a rig's file, made, checked and changed in its form, and the rig measured.

It is not a run and plays no curriculum. **The rig file** (:mod:`fmri_gym.config.rig`, one per
rig) says what software cannot see: the site and rig, the PI, the monitor or projector, the
photodiode, the sound path to the ear, the trigger hardware. Measuring a rig is also how it gets
its file: when it is missing or invalid, the form opens to fill it in first
(:func:`ensure_rig`); a rig whose file says nothing of its triggers is asked for them, and the
answer is saved in the rig file: no rig measurement is anyone's default::

    rig-setup [--rig <rig file>]                    # the form; without --rig, pick a rig
    rig-setup --rig <rig file> --measure [--long]   # measure the rig
    rig-setup report <rig measurement folder>...    # re-file rig measurements already run
    rig-setup pool data/ /mnt/siteB/data/ --out rigmeasurements.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from ...config import rig as rig_file
from ...sync.triggers import TriggerError, Triggers
from .. import needs_pyside6
from .measure import LONG, QUICK, RigMeasurement, RigMeasurementSettings, take_measurements
from .report import finish, pool, write_outputs


def prepare(rig: rig_file.RigConfig) -> tuple[rig_file.RigConfig, str | None]:
    """Before the window: the rig's trigger settings, and a line the rig measurement can open.

    A rig whose file says nothing of its triggers is asked for them -- a
    preset, or one's own -- and the answer is saved in its file
    (:func:`_ask_triggers`). A trigger line that does not open fails the
    triggers test, and the rig measurement goes on without the line (said on the console
    and in the manifest), so the other measurements still run.

    :param rig: the rig to set up.
    :raises ValueError: when the rig has no trigger settings and none is chosen.
    :return: the rig to play on, and why its line did not open (``None`` when it did).
    """
    if not rig.has_trigger_setup:
        rig = _ask_triggers(rig)
    try:
        Triggers(rig.triggers).close()
        return rig, None
    except (TriggerError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    print(f"rig measurement: the trigger line did not open ({error}); the triggers test fails, "
          "the other measurements run without codes", file=sys.stderr)
    return rig.without_trigger_line(), error


def _ask_triggers(rig: rig_file.RigConfig) -> rig_file.RigConfig:
    """The rig once its file has trigger settings, chosen in rig-config's Triggers step.

    :raises ValueError: with no screen for the form, or when it is cancelled.
    """
    where = rig.path
    fix = (f'{where} has no "triggers": set them with rig-config --rig {where} or in the '
           "file, or in the form a rig measurement opens")
    if not can_show_form():
        raise ValueError(f"rig measurement: {fix}; there is no screen here for the form")
    from .window import fill_rig
    if fill_rig(rig_file.load(where), where, need_triggers=True) is None:
        raise ValueError(f"rig measurement: {fix}; it was cancelled")
    print(f"rig measurement: trigger settings saved in {where}", file=sys.stderr)
    return rig_file.RigConfig.open(where)


def measure_rig(rig_path: str, settings: RigMeasurementSettings) -> bool:
    """Set the rig up and measure it: its file (:func:`ensure_rig`, :func:`prepare`), the
    measurements (:func:`~.measure.take_measurements`); then file it all
    (:func:`~.report.finish`).

    :param rig_path: the rig's file, made in the form when it is missing or invalid.
    :param settings: how long and how many (:data:`~.measure.QUICK`, :data:`~.measure.LONG`).
    :return: ``True`` if a test failed or did not finish.
    :raises ValueError: when the rig file is not made, or a device does not open.
    """
    rig, trigger_error = prepare(ensure_rig(rig_path))
    print(f"rig: {rig.status()}", file=sys.stderr)
    rig_measurement = RigMeasurement.open(rig, settings, trigger_error)
    take_measurements(rig_measurement)
    return finish(rig_measurement)


def ensure_rig(path: str) -> rig_file.RigConfig:
    """The rig to set up; when its file is missing or invalid, the form to
    fill it in first (:func:`fmri_gym.standalone.rig_setup.window.fill_rig`), then the file.

    The form needs the ``gui`` extra and a screen; without them, or when it is
    cancelled, the rig file's own error is raised.

    :param path: ``--rig``, the rig file.
    :raises ValueError: when there is still no valid rig file, saying how to make one.
    """
    try:
        return rig_file.RigConfig.open(path)
    except ValueError as error:
        if not can_show_form():
            raise
        print(f"{error}\nrig measurement: opening the rig form", file=sys.stderr)
        from .window import fill_rig
        if fill_rig(rig_file.draft(path), path) is None:
            raise
    return rig_file.RigConfig.open(path)


def can_show_form() -> bool:
    """PySide6 is installed and there is a screen to show it on."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return sys.platform != "linux" or bool(os.environ.get("DISPLAY")
                                           or os.environ.get("WAYLAND_DISPLAY"))


def main(argv: Sequence[str] | None = None) -> None:
    """``rig-setup``: the form, the measurement, or ``report`` / ``pool`` (the module docstring)."""
    p = argparse.ArgumentParser(
        prog="rig-setup", description="Make, check or change a rig file in a form, or measure "
        "the rig (--measure), making its file first when it is missing or invalid. report and "
        "pool re-file and stack the rig measurements already filed.")
    p.add_argument("--rig", help="the rig file, e.g. configs/<team>/rigs/<rig>.json; made when "
                                 "missing. Without it, a window asks which rig")
    p.add_argument("--measure", action="store_true",
                   help=f"measure the rig rather than open its form ({QUICK.about})")
    p.add_argument("--long", action="store_true",
                   help=f"with --measure, measure the rig at length ({LONG.about}): once per "
                        "rig, and after any hardware, driver or OS change")
    sub = p.add_subparsers(dest="command")
    f = sub.add_parser("report", help="re-file rig measurements already run: their reports and the "
                                      "data root's rigmeasurements.tsv, with this version")
    f.add_argument("folders", nargs="+", help="rig measurement folders")
    f.add_argument("--data-root", default="data")
    q = sub.add_parser("pool", help="stack every rig measurement under the roots into one table")
    q.add_argument("roots", nargs="+")
    q.add_argument("--out", help="file to write (default: stdout)")
    args = p.parse_args(argv)
    if args.command == "report":
        _refile(args.folders, args.data_root)
    elif args.command == "pool":
        _write_pool(args.roots, args.out)
    elif args.measure:
        if args.rig is None:
            p.error("--measure needs --rig <rig file>")
        sys.exit(1 if measure_rig(args.rig, LONG if args.long else QUICK) else 0)
    elif args.long:
        p.error("--long is how long to measure: give it with --measure")
    else:
        _edit_rigs(args.rig)


def _edit_rigs(rig_path: str | None) -> None:
    """The form on ``rig_path``, or the window that picks a rig; needs the ``gui`` extra."""
    try:
        from .window import edit_rigs
    except ImportError as exc:
        needs_pyside6(exc)
    edit_rigs(rig_path)


def _refile(folders: list[str], data_root: str) -> None:
    """File each rig measurement folder again: its reports, and the data root's table."""
    for folder in folders:
        with open(os.path.join(folder, "manifest.json")) as fh:
            m = json.load(fh)
        if "summaries" not in m:
            raise ValueError(f"{folder}: its manifest is not a rig measurement's")
        write_outputs(folder, m, data_root)
        print(f"{os.path.join(folder, 'report.html')}")
    print(f"all rig measurements: {os.path.join(data_root, 'rigmeasurements.tsv')}")


def _write_pool(roots: list[str], out: str | None) -> None:
    """Every rig measurement under ``roots`` as one table, to ``out`` or stdout."""
    text = "\n".join(pool(roots)) + "\n"
    if out is None:
        sys.stdout.write(text)
        return
    with open(out, "w") as f:
        f.write(text)
