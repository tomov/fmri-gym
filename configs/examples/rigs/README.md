# Example rigs

The example rigs play out of the box, one per kind of setup; each says what it stands for in its `notes`. `empty.json` is `{}`: every setting its default, a 1024x768 window with audio on whose run waits for the scanner's `=` after SPACE (QUICKSTART's). Three are a 1024x768 window with audio on and no scanner (a run starts on SPACE), one per device: `laptop.json`, a laptop and its keyboard alone; `xbox.json`, an Xbox controller (`"inputs": ["gamepad"]`); `gmpd3.json`, a Current Designs GMPD3 with one stick and six buttons on its 932 interface (`"inputs": ["gmpd3"]`). Two are a fullscreen scanner with that GMPD3: `fmri.json`, whose run waits for the `=` the 932 types at each volume, and `meg.json`, whose run starts the recording and codes its events over a serial port (the codes below). A rig names its devices under `"inputs"` and a curriculum names their controls (README "Rigs"), so a curriculum plays on a rig whose inputs have every control it names: the keyboard curricula on any of them, the `*-GMPD3-*` ones on a rig with the GMPD3. Name one with `--rig configs/examples/rigs/empty.json`; every run names its rig file, so a session never lands on an example by default. Open one in `rig-config` and save it under your team (`configs/<team>/rigs/`) as the start of a real rig. The data records their site as `examples`.

The rig's `"triggers"` say how a run starts and what the recording gets (README, "Triggers"). A rig whose MEG or EEG recording starts from the trigger line, codes over serial:

```json
"triggers": {
  "sync": {"mode": "send", "delay": 0.0},
  "backend": "serial", "port": "/dev/ttyUSB0",
  "codes": {"frame_bits": 3, "task_start": 8, "task_stop": 16, "episode_start": 32, "scanner_start": 64}
}
```

On a PC with no parallel port (a Mac), the trigger input can be wired to a USB DAQ's digital port instead; for a Measurement Computing USB-1208FS on its port A, `"backend": "hid", "hid_device": "mcc-1208fs", "port": "A"`.

An fMRI rig whose trigger box types `=` at every volume needs only `"triggers": {"sync": {"mode": "wait", "key": "EQUALS"}, "backend": "null"}` (`key` is the key's name, as a curriculum writes it).
