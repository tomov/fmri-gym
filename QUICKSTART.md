# Quick start

Commands to install fmri-gym and play each game on a laptop, in order. What they do and why: [README.md](README.md).

## 1. Install

```bash
sudo apt install libportaudio2 libopenal1 libsdl2-2.0-0 libxcb-cursor0   # Linux only
brew install openal-soft                        # Apple Silicon Mac only
curl -LsSf https://astral.sh/uv/install.sh | sh # if you don't have uv
git submodule update --init                     # if you cloned without --recursive
uv sync --extra dbp --extra gui
uv run fmri-play --help                         # check: prints usage
```

## 2. Play a game

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/minihack/levels/room5x5.json
uv run session-editor                           # or: design sessions and runs (the bottom line shows how to play them)
uv run rig-setup                                # pick, make or change a rig file; run its rig measurements
uv run gamepad-calibration                      # calibrate the game controller plugged in, and see what it sends
uv run rig-setup --rig configs/examples/rigs/minimal.json --measure   # measure the rig (--long: at length, once per rig)
uv run bids-from-logs --rig configs/examples/rigs/minimal.json        # the runs played: a behavioural BIDS dataset and each run's timeline, in data/bids
```

Press **RETURN**, then **=** (the scanner trigger; on a laptop you press it). **CTRL+ESCAPE** (hold CTRL, press ESCAPE) quits; **CTRL+RETURN** ends the run but lets a session script go on to its next run; either way the data is saved under `data/`. Any other level: swap in another file from `configs/curricula/<backend>/levels/`.

## 3. Every game

### ViZDoom

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/basic.json                      # strafe and shoot one monster
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/deadly_corridor.json            # fight down a corridor to the vest
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/deathmatch.json                 # arena, full arsenal, scored by kills
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/defend_center.json              # surrounded; turn and shoot
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/defend_line.json                # they advance down a hall and respawn
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/health_gathering_supreme.json   # acid-floor maze; find medikits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/my_way_home.json                # navigate a 9-room maze to the vest
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/predict_position.json           # lead a moving target with a rocket
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vizdoom/levels/take_cover.json                 # no weapon; dodge fireballs```

### COOM

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/pitfall.json          # cross a corridor of randomized pits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/chainsaw.json         # hunt maze enemies at melee range
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/run_and_gun.json      # find and shoot maze enemies
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/health_gathering.json # draining floor; collect health kits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/hide_and_seek.json    # evade enemies, grab kits when low
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/arms_dealer.json      # collect weapons, deliver to platforms
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/floor_is_lava.json    # stay on the briefly-appearing platforms
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/parkour.json          # jump the gaps and ledges
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/coom/levels/raise_the_roof.json   # press wall switches before the ceiling crushes you
```

### Crafter

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/crafter/levels/L4.json
```

### Rush Hour

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/rushhour/levels/easy.json      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/rushhour/all.json    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

### Baba Is AI

Not in `dbp`; sync the `baba` extra first.

```bash
uv sync --extra baba
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/baba/levels/make_win.json
```

### Baba Is Auto

Included in `dbp`. On Linux, first run `sudo apt install build-essential python3-dev`; on a Mac, `xcode-select --install`.

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/baba_auto/levels/baba_is_you.json
```

### VGDL

```bash
uv sync --extra dbp --extra vgdl
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/vgdl/all.json   # all ten, 15 s each
```

### AI GameStore

First install [Google Chrome](https://www.google.com/chrome/) if the machine has none.

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game1.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game2.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game3.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game4.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game5.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game6.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game7.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game8.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game9.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/aigamestore/game10.json
```

### MiniHack

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/minihack/levels/room5x5.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/minihack/levels/room15x15.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/minihack/levels/mazewalk9x9.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/minihack/levels/river.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/minihack/levels/corridor.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/minihack/levels/eat.json
```

### SuperTuxKart

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/examples/rigs/minimal.json --curriculum configs/curricula/stk/levels/hacienda.json
```

## Before a real session

```bash
uv run rig-setup                   # New rig... (team, rig name), fill in its six steps, Save,
                                   # then run its long rig measurement from the last step (once per rig)
uv run gamepad-calibration         # a game controller: calibrate it (also from rig-setup, step 3)
uv run session-editor --session configs/examples/sessions/pilot.sh   # adapt the example session, save it under your team
sh configs/examples/sessions/pilot.sh  # play it: about an hour (edit SUBJECT=, SES=, RUN_DURATION=, and RIG= at the top for your team)
```

## A session's recording

A session script records itself: its head starts `session-logger` before the first line and stops it when the script ends, however it ends (the example sessions and every script `session-editor` writes). By hand, the same three steps:

```bash
RECORDING=$(uv run session-logger start --rig configs/<team>/rigs/<rig>.json --subject sub-01 --ses 1)   # before anything else
uv run session-logger mark "$RECORDING" "the localizer"   # a line of your own, stamped now
uv run session-logger stop "$RECORDING"                   # after everything else
```

It records every event of every input device, the ones plugged in during the session too, read from the OS at the time the OS stamped it, which application has the focus, the machine's stalls, the marks and the clock, from start to stop, beside whatever the session runs; `bids-from-logs` gives each press its device's time. Before the first session on a machine, give it the right to read every device ([README, A session's recording](README.md#a-sessions-recording-session-logger)): on macOS, the Input Monitoring permission for the app the script is started from; on Linux, the `input` group; on Windows, nothing, and the session script runs in Git Bash.

## After a session

```bash
uv run bids-from-logs --rig configs/<team>/rigs/<rig>.json   # the runs as a behavioural BIDS dataset, with each run's timeline to check, in <data root>/bids
```

What goes into the dataset is in the README, "A BIDS dataset from the logs"; its `dataset_description.json` is yours to add.
