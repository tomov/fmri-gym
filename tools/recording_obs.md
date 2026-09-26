# Local OBS comparison

This is an experimental controller, independent of the fmri-gym experiment loop.
Requires OBS 28+ on X11, `xprop`, `pactl`, and Python with `websocket-client`. Protocol/source
settings are based on OBS 30.0.2; actual recording must be validated on the machine.

Prepare a **new** private configuration directory (does not launch OBS):

```bash
python tools/recording_obs.py --state /tmp/fmri-obs-test prepare \
  --output /tmp/fmri-obs-recordings --width 640 --height 480 --fps 60
```

Run the printed `launch_command` separately. It uses `XDG_CONFIG_HOME`, creates no
global desktop or microphone inputs, and never auto-starts recording. The random
WebSocket password stays in mode-0600 files inside a mode-0700 state directory;
it is not passed on the command line or printed. OBS listens on its usual network
interfaces, with authentication required; the controller connects only to loopback.

Open the game's 640×480 window and route **only that process** to a dedicated Pulse
sink named `fmri_gym_record_validation` in the examples below. Names must be
`fmri_gym_record` or `fmri_gym_record_<suffix>`. This script does not
create that sink or change audio routing. Check the sink's streams separately.
If playback is looped to speakers, use the same route in baseline/PyAV comparisons.

**`PULSE_SINK` alone is insufficient.** In this setup it routed pygame's mixer, but
the experiment's sounddevice/PortAudio stream used ALSA's PipeWire default and
bypassed the dedicated sink, producing a silent OBS audio track. Verify the actual
game stream against `pactl --format=json list sink-inputs` and the sink index.
For this PipeWire 1.0.5 machine, launch only the game process with both variables:

```bash
PIPEWIRE_NODE=fmri_gym_record_validation \
PULSE_SINK=fmri_gym_record_validation python your_game_command.py
```

`PIPEWIRE_NODE` selects the ALSA PipeWire target; `PULSE_SINK` covers Pulse clients
such as pygame's mixer. A 2-second silent PortAudio probe verified its output went
to this sink and disappeared when closed, without changing the desktop default.
The ALSA stream did not expose a PID in Pulse metadata: correlate its new index,
`target.object`, PipeWire client, and lifetime instead of assuming a PID is present.
The attempted custom `ALSA_CONFIG_PATH` override was superseded by the system's
late config hooks: the effective `pcm.default.playback_node` was still `-1`.
Verify recorded audio is nonzero before accepting a run.
This controller validates the monitor's identity, not the game's actual routing.

```bash
python tools/recording_obs.py --state /tmp/fmri-obs-test setup \
  --window-id 0x123456 --title 'fmri-gym Recording Validation' \
  --sink-monitor fmri_gym_record_validation.monitor
python tools/recording_obs.py --state /tmp/fmri-obs-test start
python tools/recording_obs.py --state /tmp/fmri-obs-test stat --count 20 --interval 1
python tools/recording_obs.py --state /tmp/fmri-obs-test stop
```

Replace the window ID with the live game's ID. Selection uses **only this XID**,
not automatic title matching or the desktop. Window title is also validated.
Borders and mouse cursor are excluded; the window must match canvas dimensions,
with no scaling. Keep the window open until recording stops. Call `stop` in a
`finally` block in experiment drivers; disconnecting the control socket does not
stop a recording.

Window title verification reads only that XID's `_NET_WM_NAME` with `xprop`; Pulse
monitor verification reads `pactl` metadata. The OBS 30.0.2 property-list API crashed
locally in `linux-capture.so` while sorting window names (`qsort` → `strcmp`), so
this controller does not call `GetInputPropertiesListPropertyItems`.
The isolated profile also disables `BasicWindow.ShowContextToolbars`: the GUI's
window toolbar invokes the same faulty enumeration when a new source is selected.

The initial profile uses MKV, x264 ultrafast CRF 18, PCM s16le stereo at 48 kHz,
NV12/BT.709 partial-range video. This is **lossy video**, not an RGB preservation
test. `prepare --encoder h264_nvenc` selects NVENC constant-QP 18 with
`delay=0 zerolatency=1 bf=0 rc-lookahead=0`; compare codecs separately from capture
methods. The explicit settings avoid NVENC's queued tail: OBS 30's custom FFmpeg
output closes without draining the encoder. A local FFmpeg 6.1.1 probe submitted
120 frames: the previous profile yielded 106 before EOF plus 14 on flush; the new
profile yielded all 120 before EOF. A subsequent 20-second OBS calibration decoded
1501 frames, equal to its output count. Recorded audio followed flashes by a median
37.6 ms (p95 45.5 ms); these encoder flags do not imply zero audiovisual latency.
Repeat calibration for the final experiment setup and duration.
Disabling B frames can affect compression efficiency. Use `--sample-rate 44100`
when that matches the test audio. The encoder choice is explicit and is not silently
downgraded.

Python API (import from this file):

```python
prepare(state_dir, output_dir, width=640, height=480, fps=60)
# Launch the returned launch_args with launch_env merged into os.environ.
with OBSClient(state_dir) as obs:
    obs.setup_sources(window_id, title, 'fmri_gym_record_validation.monitor')
    before = obs.stats()
    obs.start()  # waits for OBS_WEBSOCKET_OUTPUT_STARTED
    try:
        run_test_sequence()
    finally:
        result = obs.stop()
```

`control.jsonl` contains monotonic and wall-clock timestamps, OBS request/response
times, output events and counters. Subtract initial from final render counters;
output counters reset when recording starts, so use their final values for that
recording. Compare those counts with decoded file frames as well. The STARTED
event is **not** a physical first-frame timestamp. Use
known flash/beep markers and decode their positions in the resulting MKV for
audio/video offset and drift; compare game frame intervals for recording overhead.
Screens and speakers still require external timing measurements for physical onset.

References: [OBS WebSocket protocol](https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md),
[Xcomposite settings](https://github.com/obsproject/obs-studio/blob/30.0.2/plugins/linux-capture/xcomposite-input.c),
[Pulse source settings](https://github.com/obsproject/obs-studio/blob/30.0.2/plugins/linux-pulseaudio/pulse-input.c).
