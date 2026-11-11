---
name: troubleshooting
description: What to do when fmri-play, rig-setup, gamepad-calibration or a session script stops before the run starts, with an error about the rig file, the curriculum, a controller or its calibration, the triggers, the audio output, a game that cannot start, --resume or --loop. Use when a run refuses to start, when asked what an fmri-gym error means, or how to fix one.
---

# When a run will not start

A run checks everything it can before the participant screen opens, and stops on the first thing that is wrong (AGENTS.md, Rule 4). Each message says what is wrong in the terms of the module that found it: a rig's `inputs`, a controller, a trigger setting. How to fix it can involve a tool or a file that module knows nothing about, so the fixes are here. Find the message by its beginning below.

## The rig file

- **`<path>: no such rig file`**: `--rig` names a file that is not there. Make one in the rig form, `uv run rig-setup --rig <path>`, or copy `configs/examples/rigs/empty.json` and fill it in. A rig file lives in `configs/<team>/rigs/`.
- **`<path>: unknown key ...`, `... must be ...`, `"site" is ..., but the file is ...`**: the rig file does not have the shape `humans/config/rig.py` describes. Fix it by hand, or open it in `uv run rig-setup --rig <path>`, which writes only valid files. `site` and `rig` must match the team folder and the file name.
- **`"inputs": expected a list of distinct names from [...]`**: a device in the rig's `inputs` is misspelled or listed twice. The names are the ones listed in the message.

## The curriculum against the rig

- **`phase N: needs [...], which none of <rig>'s inputs has (...)`**: the curriculum maps a control (a controller button, a stick direction) that no input device of the rig has. Add the device to the rig's `inputs` (in rig-setup, or by hand), or play the curriculum on a rig that has it. A keyboard curriculum plays on any rig.
- **`phase N: controls: <combo> uses '<key>', the key <rig>'s scanner types at every volume`**: with `triggers.sync.mode` `wait`, the scanner's trigger box types that key at every volume, so a game bound to it would act on its own. Bind the game to another control in the curriculum.
- **`--duration <s>: <rig>'s fixations take <s> s`**: the run would be all fixation. Give a longer `--duration`, or shorten the rig's `start_fixation` / `end_fixation`.
- **`controls: "" missing: the env's action space is ...`**: a real-time phase steps every frame, so it needs the action sent while nothing is held. Add `"": 0` (or the env's no-op) to the phase's `controls`, or set `"is_turn_based": true` to step only on presses.
- **`controls: "": ...: the env's action is MultiBinary(...)`**: on a MultiBinary env, no control held is every button up already; remove the `""` entry.
- **`phase N: <backend> game '<game>' cannot start (ModuleNotFoundError ...)`** (each game is built once at start-up, unless `--skip-load-games`): the backend's extra is not installed. The message names the `uv sync --extra` line; name every extra the machine uses in one sync, since a sync uninstalls the extras it does not name.

## Controllers and their calibration

- **`no "<kind>" is plugged in (joysticks SDL sees: ...)`**: the rig lists a controller in `inputs` and none of that kind is connected. Plug it in (on the GMPD3, the 932 interface in joystick mode), or take it out of the rig's `inputs` if this rig plays without it.
- **`no calibration of '<controller>' in <folder>, and its stick controls are required`**: the curriculum maps a stick direction and this machine has never calibrated that controller. Run `uv run gamepad-calibration` (or Calibrate controller... in rig-setup) with the controller plugged in; the calibration is kept per machine in the folder named.
- **`<file>: calibrates the axes [...], but '<controller>' has [...]`**: the newest calibration was made on another kind of controller under the same name. Calibrate again with `gamepad-calibration`.
- **`<file>: not JSON ...`, `<file>: ... expected ...`**: a calibration file is damaged. Calibrate again, then delete or move the damaged file out of the folder; files there are never rewritten.

## Triggers

- **`triggers: backend '<b>' needs a port (e.g. ...)`**: set `triggers.port` in the rig file (rig-setup lists the ports it finds), or set `triggers.backend` to `"null"` to run without trigger codes.
- **`triggers: the <b> backend needs the <pkg> package (pip install <pkg>)`**: install it into the environment the run uses, or set the backend to `"null"`.
- **`triggers: cannot open <b> triggers on <target>: ...`**: the device is unplugged, the port is wrong, or the user lacks permission (on Linux, `/dev/ttyUSB*`, `/dev/parport*`, `/dev/hidraw*` need a group or a udev rule). Check them, or set the backend to `"null"`.
- **`triggers: sync.mode "send" sends scanner_start on the trigger line, ...`**: a `send` sync needs a real backend; pick one, or use sync mode `wait` or `none`.
- **`triggers: code ...`, `frame_bits ...`, `unknown ...`**: the code scheme is described in `humans/sync/triggers.py`'s module docstring; codes must not share a bit with each other or with the frame bits.

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
