"""A run's timeline in words: what the run was, the numbers it is checked by, and each event in
order, in seconds from the trigger.

For a person checking a run: the head says what the logs are read with (the code, the
machine, the rig, the display, which logs there are, how the run ended), the checks say what
to look at, and the timeline lists the screens, blocks, episodes, presses, scanner pulses,
codes sent, printed lines, missed refreshes and device events no press accounts for, but not
the game frames one by one.
"""

from __future__ import annotations

from typing import Any

from .run_timeline import MISSED_REFRESH, Press, RunRecord, Timeline


def timeline_text(record: RunRecord, timeline: Timeline) -> str:
    """The run's head, checks and timeline.

    :param record: the run's manifest and logs.
    :param timeline: their timeline (:func:`~.run_timeline.build_timeline`).
    :return: the text, one event a line.
    """
    lines = _head(record, timeline) + ["", "Checks"] + _checks(timeline.quality)
    lines += ["", "Timeline (seconds from the trigger; before it, negative)"]
    lines += [f"{time:10.3f}  {what:<8} {detail}" for time, _, what, detail in
              sorted(_events(timeline), key=lambda event: (event[0], event[1]))]
    return "\n".join(lines) + "\n"


def _head(record: RunRecord, timeline: Timeline) -> list[str]:
    """What the run was: its name, the code, the machine, the rig, the display, its logs."""
    manifest = record.manifest
    versions, platform = manifest["versions"], manifest["platform"]
    sync, display = manifest["rig"]["triggers"]["sync"], manifest["display"]
    end = next((mark.detail for mark in timeline.marks if mark.what == "run"
                and mark.detail.startswith("end")), "no end line: the run stopped mid-way")
    return [
        f"run       {timeline.label} (attempt {manifest['run']['attempt']})",
        f"code      fmri-gym {versions['fmri_gym']}, git {versions['git']}; Python "
        f"{versions['python']}, pygame {versions['pygame']}, SDL {versions['sdl']}",
        f"machine   {platform['system']} on {platform['hostname']}; perf_counter: "
        f"{platform['perf_counter']['implementation']}",
        f"rig       {manifest['rig']['path']}: starts on {_sync_words(sync)}"
        + ("; dummy trigger (a test run)" if manifest["dummy_trigger"] else ""),
        f"display   {display['size'][0]}x{display['size'][1]}, vsync {display['vsync']}, "
        f"{display['refresh_rate']} Hz, driver {display['driver']}",
        f"logs      format {manifest['log_format']['format']} {manifest['log_format']['version']}"
        f": the run's log and {len(record.block_lines)} block log(s)",
        "session   " + (
            f"recording {record.session_recording.name}: "
            f"{len(record.session_recording.start['devices'])} device(s) "
            f"({record.session_recording.start['reader']}), {len(record.device_lines)} device "
            "events in the run" if record.session_recording else
            "no session recording covers this run"),
        f"ended     {end.removeprefix('end: ')}, {timeline.end:.3f} s after the trigger"]


def _sync_words(sync: dict[str, Any]) -> str:
    """How the run started, in words."""
    if sync["mode"] == "wait":
        return f"the scanner's key {sync['key']!r}"
    return "a start code sent" if sync["mode"] == "send" else "the experimenter's key"


def _checks(quality: dict[str, Any]) -> list[str]:
    """The checks, one a line."""
    pulses, refreshes, presses = quality["pulses"], quality["refreshes"], quality["presses"]
    devices = quality["devices"]
    out = [f"  scanner pulses  {pulses['count']}" + (
        f", every {pulses['median_interval_s']:.3f} s (median; {pulses['min_interval_s']:.3f}"
        f" to {pulses['max_interval_s']:.3f}), {pulses['gaps_missing_a_pulse']} gap(s) missing "
        "a pulse" if "median_interval_s" in pulses else "")]
    out.append(f"  refreshes       {refreshes['flips']} flips; " + (
        f"{refreshes['missed']} gap(s) of more than {MISSED_REFRESH:g} refreshes at "
        f"{refreshes['refresh_rate']} Hz (longest {1000 * refreshes['longest_gap_s']:.1f} ms)"
        if "missed" in refreshes
        else "not vsync-locked: misses not counted"))
    out.append(f"  presses         {presses['count']}, {presses['in_episodes']} in episodes: "
               f"{presses['given_to_a_step']} given to a step, {presses['given_to_none']} "
               "to none")
    out.append(_session_check(quality["session"]))
    if not devices["read"]:
        return out + ["  devices         not read from the OS"]
    return out + [f"  devices         {devices['events']} events, {devices['presses_timed']} "
                  "presses timed by their device" + (
                      f", read {devices['read_delay_median_ms']:.1f} ms later (median; 95%: "
                      f"{devices['read_delay_p95_ms']:.1f}, max "
                      f"{devices['read_delay_max_ms']:.1f})" if devices["presses_timed"] else "")
                  + f"; {devices['dropped']} drop(s), {devices['added']} plugged in, "
                  f"{devices['gone']} unplugged; recording "
                  f"{devices['ended']}"]


def _session_check(session: dict[str, Any]) -> str:
    """What the session's recording saw of the machine during the run, in one line."""
    if not session["recorded"]:
        return "  session         not recorded"
    return (f"  session         {session['focus_changes']} focus change(s), "
            f"{session['stalls']} stall(s)" + (
                f" (longest {session['longest_stall_ms']:.0f} ms)" if session["stalls"] else ""))

def _events(timeline: Timeline) -> list[tuple[float, int, str, str]]:
    """Every event as ``(time, order among ties, what, detail)``."""
    events = [(mark.time, 0, mark.what, mark.detail) for mark in timeline.marks]
    events += [(screen.onset, 1, "screen", f"{screen.kind}{_text_words(screen.text)} "
                f"({screen.duration:.3f} s)") for screen in timeline.screens]
    for block in timeline.blocks:
        events.append((block.onset, 2, "block", f"{block.name}: {block.backend} {block.game} "
                       "starts"))
        events.append((block.onset + block.duration, 5, "block",
                       f"{block.name} ends: {block.n_frames} frames"
                       + ("" if block.has_ended else " (the run stopped inside it)")))
    for episode in timeline.episodes:
        events.append((episode.onset, 3, "episode", f"{episode.episode} starts"))
        if episode.duration is not None:
            events.append((episode.onset + episode.duration, 4, "episode",
                           f"{episode.episode} ends: {episode.outcome}, score {episode.score:g}"))
    events += [(press.onset, 3, "pulse" if press.is_scanner_pulse else "press",
                _press_words(press)) for press in timeline.presses]
    return events


def _text_words(text: str | None) -> str:
    """A screen's text, on one line: its first line, and how many more there are."""
    if not text:
        return ""
    lines = text.split("\n")
    more = f" [+{len(lines) - 1} lines]" if len(lines) > 1 else ""
    return f": {lines[0][:70]!r}{more}"


def _press_words(press: Press) -> str:
    """A press in words: the control, how long, the step it went to, when its device sent it."""
    words = [press.control]
    if press.duration is not None:
        words.append(f"held {press.duration:.3f} s")
    if press.frame is not None:
        words.append(f"to frame {press.frame} of episode {press.episode}")
    if press.device_time is not None:
        words.append(f"read {1000 * (press.read_time - press.device_time):.1f} ms after its "
                     "device sent it")
    return ", ".join(words)
