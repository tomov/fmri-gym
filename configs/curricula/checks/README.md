# Rig checks

Curricula whose phases are checks instead of games (README, "Timing"). They say only for how long and how many: what is tested, and with what, is the rig file's (`rig-config`, its Rig check equipment step) -- the keys of its controller's bindings and its response keys, its photodiode (`none`, `soundcard`, `recording`), a microphone, the recording's rate, more frame rates -- and a test the rig gives no reason to run is reported as not run, with why. These curricula may not set any of it. Run either from `rig-config`'s last step, once the rig file is saved.

- `rig-check.json`: before a session, under a minute, most of it pressing the buttons. Results: `report.md` and `rigcheck.tsv` in the run's folder; this rig's rig file is asked for when it has none.
- `rig-check-long.json`: once per rig, and after any hardware, driver or OS change; about 25 minutes. The display for 2 min; the frame test at 60, 30, 20 and 50 fps (50 divides no common refresh), 1 min each, idle and with every core busy; 20 passes of every code; 850 flashes. It measures offsets to a fraction of a ms, their drift, missed refreshes and pulses, and the TR on this PC's clock.

**Triggers.** The check tests the rig's own (its rig file's `"triggers"`): a rig whose file has none is asked for them on its first check (a preset or your own), and the answer is saved in the rig file; `rig-config`'s Triggers step changes them. With sync `wait` the check waits for the scanner and counts its pulses.
