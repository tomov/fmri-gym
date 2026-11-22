---
name: troubleshooting
description: What to do when fmri-play, rig-setup, gamepad-calibration or a session script stops before the run starts, or session-logger stops, or bids-from-logs writes nothing, with an error about the rig file, the curriculum, a controller or its calibration, the triggers, the audio output, a game that cannot start, --resume or --loop. Use when a run refuses to start, when asked what an fmri-gym error means, or how to fix one.
---

# When a run will not start

A run checks everything it can before the participant screen opens, and stops on the first thing that is wrong (AGENTS.md, Rule 4). Each message says what is wrong in the terms of the module that found it: a rig's `inputs`, a controller, a trigger setting. How to fix it can involve a tool or a file that module knows nothing about, so the fixes are here. Find the message by its beginning below.

## The rig file

- **`<path>: no such rig file`**: `--rig` names a file that is not there. Make one in the rig form, `uv run rig-setup --rig <path>`, or copy `configs/examples/rigs/minimal.json` and fill it in. A rig file lives in `configs/<team>/rigs/`.
- **`<path>: unknown key ...`, `... must be ...`, `"site" is ..., but the file is ...`**: the rig file does not have the shape `humans/config/rig.py` describes. Fix it by hand, or open it in `uv run rig-setup --rig <path>`, which writes only valid files. `site` and `rig` must match the team folder and the file name.
- **`"inputs": expected a list of distinct names from [...]`**: a device in the rig's `inputs` is misspelled or listed twice. The names are the ones listed in the message.
- **`"inputs": every rig has the 'keyboard'`**: the rig's `inputs` leave out the keyboard, which every rig needs, since the experimenter starts and stops a run on it. Add `"keyboard"` to the rig's `inputs` (a rig with the keyboard alone has `"inputs": ["keyboard"]`, or no `inputs`); opening and saving the rig in `rig-setup`, which always ticks it, does the same.

## The curriculum against the rig

- **`phase N: needs [...], which none of <rig>'s inputs has (...)`**: the curriculum maps a control (a controller button, a stick direction) that no input device of the rig has. Add the device to the rig's `inputs` (in rig-setup, or by hand), or play the curriculum on a rig that has it. A keyboard curriculum plays on any rig.
- **`phase N: controls: <combo> uses '<key>', the key <rig>'s scanner types at every volume`**: with `triggers.sync.mode` `wait`, the scanner's trigger box types that key at every volume, so a game bound to it would act on its own. Bind the game to another control in the curriculum.
- **`controls: [...] are the experimenter's keys, which no game may bind`**, or **`... is no control name a curriculum may use (... but the experimenter's keys: ...)`** for a menu's, hint's, message's or survey's control: the curriculum binds a key the experimenter runs the session with (RETURN starts a run, CTRL+ESCAPE stops it, and the function keys and modifiers are kept free; the list is `EXPERIMENTER_KEYS` in `humans/inputs/control_names.py`). Bind the action to another key, and change the curriculum's instructions text and controls image to match (the `controls-images` skill).
- **`--duration <s>: <rig>'s fixations take <s> s`**: the run would be all fixation. Give a longer `--duration`, or shorten the rig's `start_fixation` / `end_fixation`.
- **`controls: "" missing: the env's action space is ...`**: a real-time phase steps every frame, so it needs the action sent while nothing is held. Add `"": 0` (or the env's no-op) to the phase's `controls`, or set `"is_turn_based": true` to step only on presses.
- **`controls: "": ...: the env's action is MultiBinary(...)`**: on a MultiBinary env, no control held is every button up already; remove the `""` entry.
- **`phase N: <backend> game '<game>' cannot start (ModuleNotFoundError ...)`** (each game is built once at start-up, unless `--skip-load-games`): the backend's extra is not installed. The message names the `uv sync --extra` line; name every extra the machine uses in one sync, since a sync uninstalls the extras it does not name.

## Controllers and their calibration

- **`no "<kind>" is plugged in (joysticks SDL sees: ...)`**: the rig lists a controller in `inputs` and none of that kind is connected. Plug it in (on the GMPD3, the 932 interface in joystick mode), or take it out of the rig's `inputs` if this rig plays without it.
- **`no calibration of '<controller>' in <folder>, and its stick controls are required`**: the curriculum maps a stick direction and this machine has never calibrated that controller. Run `uv run gamepad-calibration` (or Calibrate controller... in rig-setup) with the controller plugged in; the calibration is kept per machine in the folder named.
- **`<file>: calibrates the axes [...], but '<controller>' has [...]`**: the newest calibration was made on another kind of controller under the same name. Calibrate again with `gamepad-calibration`.
- **`<file>: format: expected <n>, got ... (a calibration of another layout)`**: the controller's calibration was written by another version of fmri-gym. Calibrate again with `gamepad-calibration`. A run reads only the newest file of each controller, or the one given with `--calibration`, so older files can stay in the folder.
- **`<file>: axes.<axis>: its centre drifts <d>, past its release point on its <side> side (...)`**: the stick came back from a push further off centre than its release threshold, so a push that way would stay pressed. Calibrate again, hands off the stick while it measures the centre; if it is refused again, the stick is worn or the pad needs re-zeroing.
- **`<file>: a calibration of '<device>', which is not a controller opened (...)`, `<file>: a second calibration of '<device>' ...`**: a `--calibration` file names a device that is not plugged in, or two name the same one. Pass one file per controller plugged in.
- **`<file>: not JSON ...`, `<file>: ... expected ...`**: the calibration file is damaged. Calibrate again, or pass a good one with `--calibration`.

## Triggers

- **`triggers: backend '<b>' needs a port (e.g. ...)`**: set `triggers.port` in the rig file (rig-setup lists the ports it finds), or set `triggers.backend` to `"null"` to run without trigger codes.
- **`triggers: the <b> backend needs the <pkg> package (pip install <pkg>)`**: install it into the environment the run uses, or set the backend to `"null"`.
- **`triggers: cannot open <b> triggers on <target>: ...`**: the device is unplugged, the port is wrong, or the user lacks permission (on Linux, `/dev/ttyUSB*`, `/dev/parport*`, `/dev/hidraw*` need a group or a udev rule). Check them, or set the backend to `"null"`.
- **`triggers: sync.mode "send" sends scanner_start on the trigger line, ...`**: a `send` sync needs a real backend; pick one, or use sync mode `wait` or `none`.
- **`triggers: code ...`, `frame_bits ...`, `unknown ...`**: the code scheme is described in `humans/sync/triggers.py`'s module docstring; codes must not share a bit with each other or with the frame bits.

## A session's recording (`session-logger`)

- **`session-logger: no input device is named '<name>'; these are: ...`** (Linux), **`... no HID device is named ...`** (macOS) or **`... no raw input device is named ...`** (Windows): the rig names a device in `"device_events"` that is not plugged in, or not by that name. Copy the name from the list in the message into the rig file. On Windows a device is named by its USB product string, so one without (a laptop's built-in keyboard) cannot be named; the message counts them.
- **`session-logger: /dev/input/eventN (<name>) cannot be read by this user`** (Linux): reading a device's node takes the `input` group. `sudo usermod -aG input $USER`, then log in again (or start it with `sg input -c "..."` until then).
- **`session-logger: this app lacks the Input Monitoring permission, ...`** (macOS): a keyboard-type device (a trigger box that types keys among them) is read only with it. System Settings > Privacy & Security > Input Monitoring: add the app it is started from (the terminal), then restart that app.
- **`session-logger: perf_counter reads ...`** or **`... perf_counter is ... off mach absolute time ...`**: this Python's `perf_counter` is not on the clock the OS stamps its events with, so they could not be matched to the runs. Use the Python the project pins (`uv run`).
- **`session-logger: Windows would not send this process the devices' raw input (error N)`** (Windows): registering for the devices' raw input failed; the error is Windows' (`GetLastError`). Start the session from a desktop session of the user at the rig, not over a remote service.
- **`session-logger: no device reader for <platform>`**: there is no device reader for this OS. A rig without `"device_events"` is still recorded: its marks and the clock.
- **`session-logger: the recorder wrote nothing in ... s`**: the recorder started but did not open the devices in time; its log, in the message, says where it stopped.
- **`session-logger: WARNING: '<name>' (device N) was unplugged at HH:MM:SS: ...`** (`stop`): the device left the machine during the session, and the recording holds none of its events after that time; the runs still have their own times for its presses (when they read them). Check its cable before the next session.
- **`session-logger: WARNING: the recording in <folder> has stopped: ...`** (`mark`) or **`... did not end when told to: ...`** (`stop`): the recorder stopped before the session ended. The message says when its last line was written, and names its log, which says why; the recording keeps every line until then.

## Audio

- **`audio: no usable output device (...)`** or **`audio: this output reports no DAC timestamps, ...`**: make another output the system default (`python -m sounddevice` lists them), or set `"has_audio": false` in the rig file for a silent run; the game's sound is still logged.
- **`audio: the game makes ... ms of sound per step but steps every ... ms`**: the phase's `fps` is not the engine's own rate, so its sound cannot follow the picture. Leave `fps` out of the phase to play at the engine's rate, or set `"has_audio": false` on the phase.

## Resuming and looping

- **`resume: <run> played this curriculum to its end`**: there is nothing to carry on. Start the curriculum over without `--resume`, or add `--loop` to start it again.
- **`resume: phase N, the last game <run> played, is not the same game in this curriculum (... changed)`**: the curriculum was edited since that run. Put the named fields back, or start over without `--resume`.
- **`<run> stopped phase N mid-episode, and the <backend> backend keeps no savestate to carry it on`**: that backend cannot restore an episode. Start over without `--resume`, or start at a later phase.
- **`loop: plays the curriculum until the run's duration, and it has none`**: `--loop` needs a `--duration`.

## The data folder

- **`subject '<s>': BIDS needs sub-<letters or digits>`**: give `--subject sub-01` (letters and digits only after `sub-`).
- **`<curriculum>: no letter or digit in its name to make a BIDS task label from`**: rename the curriculum file; its path after `curricula/` is the task label.

## A BIDS dataset from the logs

`bids-from-logs` checks every run before it writes anything, and lists each run it cannot convert under `nothing was written`.

- **`<run label>: N attempts reached the trigger, and BIDS keeps one run`**: the run was played again (its folders end in `_02`, `_03`...) and more than one attempt reached the scanner. Keep the attempt the session uses: move the others out of the data root (into a folder of attempts set aside, say), then run `bids-from-logs` again.
- **`<run folder>: log format 1, recorded before runs kept a log of their own`**: the run was recorded by a version of fmri-play that kept its lines in its block folders only, so its screens, fixations and presses outside the games were never logged. Convert another subject (`--subject`) or move the run out of the data root.
- **`skipped, never reached its trigger: <run folder>`** is not an error: that attempt stopped before the scanner started (at start-up, or at the ready screen), so it has nothing to convert.
