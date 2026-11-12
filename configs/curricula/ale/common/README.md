# Atari (ALE) curricula

The `ale` backend's curricula: ten Atari 2600 games, one file each (300 s of play), and `all.json`, the ten for 15 s each. A game's file and `all.json` both include `common/<game>.json` (its env id, mode and fps) and state their own keys and duration (README, "Shared phases").

## Setup

`uv sync --extra ale` installs ale-py, whose ROMs ship with it: nothing to import.

## Keys

A phase's `controls` values are indices into the game's own action set, which differs per game. Read it with `gym.make("ALE/Pong-v5").unwrapped.get_action_meanings()` (README, "Keys"): for Pong, `['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']`, so RIGHT=2, LEFT=3, and `""` (no key held) is 0.

## Curricula

| file | plays |
|---|---|
| `all.json` | the ten games below, 15 s each |
| `asterix.json` | Asterix |
| `beamrider.json` | BeamRider |
| `boxing.json` | Boxing |
| `breakout.json` | Breakout |
| `enduro.json` | Enduro |
| `mspacman.json` | MsPacman |
| `pong.json` | Pong |
| `qbert.json` | Qbert |
| `seaquest.json` | Seaquest |
| `spaceinvaders.json` | SpaceInvaders |
| `mspacman-GMPD3-stick-nav.json`, `spaceinvaders-GMPD3-stick-nav.json` (in `ale/`) | MsPacman or SpaceInvaders on a controller's left stick (diagonals are the stick's two axes held together), SpaceInvaders firing on RT, each with its instructions and controls image, at the console's own 60 frames per second: real speed, and sprites flicker as on the console |

**Speed and flicker.** The adapter steps one console frame at a time, so a phase's `fps` is its speed: 60 is the console's own, and lower slows the game. The 2600 draws some sprites on alternate frames only (each Ms. Pac-Man ghost on one frame in four, Space Invaders' shots on every other one); at 60 fps they flicker as they did on a TV, while at 30 fps each gap holds for two refreshes and the sprites blink visibly. The stick curricula play them as the console did.
