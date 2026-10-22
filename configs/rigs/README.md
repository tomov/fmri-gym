# configs/rigs/

One file per rig, in a folder per site: `rigs/<site>/<rig>.json`, e.g. `rigs/boston-college/prisma-3t-gamepad.json`. A rig is one setup a site runs sessions on -- a monitor or projector, a scanner, a response device, a trigger line, an audio path -- and a site can have several. Its file says everything about how a run is played there that a curriculum must not: the screen (size, fullscreen, monitor, vsync), the participant's device and its keys, audio, the data root, the trigger hardware. Every key in it is optional (an empty `{}` plays on the defaults); a `site` or `rig` written in the file must be its folder and file name, and `fmri-play --rig configs/rigs/<site>/<rig>.json` names it (`fmri_gym/config/rig.py`; README "Rigs" lists its fields). The rig check's form writes one when its rig has none: commit it.

The example rigs play out of the box, one per kind of setup; each says what it stands for in its `notes`. `example/empty.json` is `{}`: every setting its default, a 1024x768 window with audio on whose run waits for the scanner's `=` after SPACE (QUICKSTART's). Three are a 1024x768 window with audio on and no scanner (a run starts on SPACE), one per device: `example/laptop.json`, a laptop and its keyboard alone (no controller read, so no bindings); `example/xbox.json`, an Xbox controller; `example/curdes-v1.json`, a Current Designs controller with one stick and six buttons on its 932 interface. Two are a fullscreen scanner with that Current Designs controller: `example/fmri.json`, whose run waits for the `=` the 932 types at each volume, and `example/meg.json`, whose run starts the recording and codes its events over a serial port (the codes below). Every controller example binds its controls to the ten keys the curricula expect (README "Rigs"). Name one with `--rig configs/rigs/example/empty.json`; every run names its rig file, so a session never lands on an example by default. Copy one as the start of a real rig.

A rig is site-specific and shared by every run and every game on it; a curriculum (`../curricula/`) is game-specific and shared by every site; a session (`../subjects/`) is one subject's one visit. Keep each setting in the one place it belongs.

A rig check files what it measured on the rig beside its file, in `measured/<rig>-<n>.json`, one new file per check that measured a latency (README, "Timing"). They are this machine's measurements, not the rig's settings, so git ignores them. A run on a rig with none warns, and the first run of a session, from a terminal, offers to run a check first.

`rig-check.sh` plays the rig check (`../curricula/checks/rig-check-long.json`) on `example/laptop.json`, filed under `sub-rig`; set your rig's file on its lines (or on the editor's Launch tab: `fmri-edit --session configs/rigs/rig-check.sh`).

The rig's `"triggers"` say how a run starts and what the recording gets (README, "Triggers"). A rig whose MEG or EEG recording starts from the trigger line, codes over serial:

```json
"triggers": {
  "sync": {"mode": "send", "delay": 0.0},
  "backend": "serial", "port": "/dev/ttyUSB0",
  "codes": {"frame_bits": 3, "task_start": 8, "task_stop": 16, "episode_start": 32, "scanner_start": 64}
}
```

On a PC with no parallel port (a Mac), the trigger input can be wired to a USB DAQ's digital port instead; for a Measurement Computing USB-1208FS on its port A, `"backend": "hid", "hid_device": "mcc-1208fs", "port": "A"`.

An fMRI rig whose trigger box types `=` at every volume needs only `"triggers": {"sync": {"mode": "wait", "key": "="}, "backend": "null"}`.
