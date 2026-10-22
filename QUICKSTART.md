# Quick start

Commands to install fmri-gym and play each game on a laptop, in order. What they do and why: [README.md](README.md).

## 1. Install

```bash
sudo apt install libportaudio2 libopenal1 libsdl2-2.0-0 libxcb-cursor0   # Linux only
curl -LsSf https://astral.sh/uv/install.sh | sh # if you don't have uv
git submodule update --init                     # if you cloned without --recursive
uv sync --extra dbp --extra gui
uv run fmri-play --help                         # check: prints usage
```

## 2. Play a game

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/room5x5.json
uv run fmri-edit                                # or: open the config editor, whose Play starts what it shows
```

Press **SPACE**, then **=** (the scanner trigger; on a laptop you press it). **ESC** quits; the data is saved under `data/`. Any other level: swap in another file from `configs/curricula/<backend>/levels/`.

## 3. Every game

### ViZDoom

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/basic.json                      # strafe and shoot one monster
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/deadly_corridor.json            # fight down a corridor to the vest
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/deathmatch.json                 # arena, full arsenal, scored by kills
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/defend_center.json              # surrounded; turn and shoot
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/defend_line.json                # they advance down a hall and respawn
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/health_gathering_supreme.json   # acid-floor maze; find medikits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/my_way_home.json                # navigate a 9-room maze to the vest
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/predict_position.json           # lead a moving target with a rocket
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/take_cover.json                 # no weapon; dodge fireballs
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/take_cover_defend_line.json     # those last two back to back, 150 s each
```

### COOM

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/pitfall.json          # cross a corridor of randomized pits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/chainsaw.json         # hunt maze enemies at melee range
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/run_and_gun.json      # find and shoot maze enemies
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/health_gathering.json # draining floor; collect health kits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/hide_and_seek.json    # evade enemies, grab kits when low
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/arms_dealer.json      # collect weapons, deliver to platforms
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/floor_is_lava.json    # stay on the briefly-appearing platforms
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/parkour.json          # jump the gaps and ledges
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/raise_the_roof.json   # press wall switches before the ceiling crushes you
```

### Crafter

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/crafter/levels/L4.json
```

### Rush Hour

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/rushhour/levels/easy.json      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/rushhour/complete.json    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

### Baba Is AI

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/baba/levels/make_win.json
```

### Baba Is Auto

On Linux, first run `sudo apt install build-essential python3-dev`; on a Mac, `xcode-select --install`.

```bash
uv sync --extra dbp --extra baba_auto
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/baba_auto/levels/baba_is_you.json
```

### VGDL

```bash
uv sync --extra dbp --extra vgdl
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vgdl/all.json   # all ten, 15 s each
```

### AI GameStore

First install [Google Chrome](https://www.google.com/chrome/) if the machine has none.

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game1.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game2.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game3.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game4.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game5.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game6.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game7.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game8.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game9.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game10.json
```

### MiniHack

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/room5x5.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/room15x15.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/mazewalk9x9.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/river.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/corridor.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/eat.json
```

### SuperTuxKart

**On a Mac, first run** `tools/stk_macos.sh` (once, about 20 minutes), then in every new shell `export STK_ENV_BIN=~/.cache/supertuxkart-gym/macos/SuperTuxKart-gym-macos-$(uname -m)/supertuxkart`.

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/stk/levels/hacienda.json
```

## Before a real session

```bash
uv run fmri-play --subject sub-rig --ses 1 --run 1 --rig configs/rigs/<site>/<rig>.json --curriculum configs/curricula/checks/rig-check-long.json   # once per rig: makes its rig file
uv run python -m fmri_gym.display --fullscreen      # flips lock to the refresh
uv run python -m fmri_gym.photodiode --fullscreen   # once per rig, photodiode on the screen
```
