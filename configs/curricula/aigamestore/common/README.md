# AI GameStore curricula

The `aigamestore` backend's curricula: the ten public [AI GameStore](https://aigamestore.org) games, one file per game with one game phase per level, plus `all.json`, all ten for 15 s each. Every game has `common/<game>.json`, which its phases include (`common/<game>.json`), stating only the level and its title (README, "Shared phases").

## Setup

`uv sync --extra aigamestore` installs the vendored aigamestore-gym env (`gym/aigamestore`), which needs a Chromium browser (system Chrome by default). The env is lock-stepped: each step advances the game `frame_skip` frames, so `fps` must equal 60 / `frame_skip` (10 with the default `frame_skip` 6).

## Keys

The env's action is MultiBinary over the game's keys, in `aigamestore_gym.GAME_KEYS` order, so a phase's `keys` values are those indices (the JSON key is what the subject presses), and held keys combine. The env relabels the on-canvas hints to the pressed key; the instructions' Controls text is kept in sync with `keys` by hand. The games' keys are the standard rig's ten (README, "Rigs") but for game3 and game7, which have more actions than a controller's six buttons: they are played on a keyboard, and a rig with a controller refuses them (as it does `all.json`, which plays them). The `*-ABXY-nav` files (`common/` and the matching curricula, for game1, game4, game7 and game8) move on the face buttons instead of the stick and are written for `configs/examples/rigs/curdes-v1-ABXY-nav.json`, whose controls press `A` `B` `X` `Y` `L` `R`; they play on that rig and no other with a controller.

| game | `GAME_KEYS` order | reference |
|---|---|---|
| game1 | 0=LEFT, 1=RIGHT, 2=SPACE, 3=Z | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game1/input.js) |
| game2 | 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=SPACE, 5=Z | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game2/input.js) |
| game3 | 0=W, 1=A, 2=S, 3=D, 4=LEFT, 5=RIGHT, 6=UP, 7=DOWN, 8=SPACE, 9=Z, 10=LSHIFT | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game3/input.js) |
| game4 | 0=SPACE, 1=UP | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game4/input.js) |
| game5 | 0=LEFT, 1=RIGHT, 2=UP | [controls.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game5/controls.js) |
| game6 | 0=LEFT, 1=RIGHT, 2=UP, 3=SPACE, 4=Z | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game6/input.js) |
| game7 | 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=Z, 5=SPACE, 6=LSHIFT, 7=1, 8=2, 9=3, 10=4, 11=5, 12=X (next weapon) | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game7/input.js) |
| game8 | 0=LEFT, 1=RIGHT, 2=SPACE, 3=Z | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game8/input.js) |
| game9 | 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=Z, 5=SPACE, 6=LSHIFT | [input.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game9/input.js) |
| game10 | 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=LSHIFT, 5=SPACE, 6=Z | [game.js](https://github.com/tomov/fmri-gym/blob/main/gym/aigamestore/game10/game.js) |

## Levels

An episode is one level: it ends on a win, a loss or a level clear, and `reset()` restarts the same level. Each level is one phase; drop or reorder them as the session needs. The blocks are blocking, as the games are: each is its level played until cleared (mode `episode`, `n_episodes` 1, `advance_on` `["won"]`); a loss restarts the same level, so the levels run back to back at the subject's own pace, and `duration` 120 s is the cap on a level nobody clears. Three games differ:

- **game4** (jetpack runner) has no levels and cannot be won: each episode is one run and ends when it is lost. The block is 10 runs (`n_episodes` 10, no `advance_on`, so every run counts), `duration` 300 s the cap.
- **game6** (shadow platformer) is one long level with checkpoints: the block is one attempt that counts only when won, a loss restarts it, and `duration` is 300 s.
- **game7** (top-down extract shooter) generates levels without an upper bound; the first five are listed.

## Curricula

| file | plays |
|---|---|
| `all.json` | the ten games, 15 s each (`mode` `duration`) |
| `game1.json` | Water Sort, levels 1–9 |
| `game2.json` | Angry-Birds-like, levels 1–9 |
| `game3.json` | twin-stick / top-down shooter (Three.js + canvas HUD), levels 1–7 |
| `game4.json` | jetpack runner, 10 runs |
| `game5.json` | mouse / cheese platformer, levels 1–6 |
| `game6.json` | shadow platformer, one long level |
| `game7.json` | top-down extract shooter, levels 1–5 |
| `game8.json` | bubble shooter, levels 1–9 |
| `game9.json` | Mega Man-like action platformer, levels 1–7 |
| `game10.json` | haunted mansion escape, levels 1–6 |
| `game<N>-ABXY-nav.json` | game1, game4, game7 or game8 on the face buttons, for the `curdes-v1-ABXY-nav` rig |
