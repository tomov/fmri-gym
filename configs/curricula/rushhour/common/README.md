# Rush Hour curricula

The `rushhour` backend's curricula: slide cars to free the red one. The 49 puzzles of `all.json` include `common/rushhour.json` (keys, rules text, settings) and state only their puzzle and title; `easy.json`, a different task, stands alone.

## Setup

`uv sync --extra rushhour` (rushhour-gym >= 0.7.0). On first use the package downloads the rushhour-env Go engine of the matching Rush-Hour release into `~/.cache/rushhour-gym/` (checksum-verified); `RUSHHOUR_ENV_BIN` or the phase's `binary` field names a binary of your own. The env is `RushHourHuman-v0`: the Rush-Hour program's own interface (car selection, its picture, its trial flow, its results columns in `info`), all in the package; the adapter is a keymap plus the fields to log.

## Keys

The env's action space is Discrete(12), rushinput's meta-actions: 0-3 choose the car up / down / left / right (spatial, what Rush-Hour's gamepad stick does), 4/5 choose the previous / next car, 6/7 slide it back (left / up) / forward (right / down) along its own axis (the four-button box), and 8-11 slide it left / right / up / down. A slide across the car's axis (LEFT on a vertical car) moves nothing and is logged as event `wrong_axis`. The keys are the rig's ten: A/S choose, the arrows slide by direction, Z/X slide back / forward; 3/4 and 1/2 do the same as in `rushinput.DefaultMap` (`rushhour_gym.human.DEFAULT_KEYS`), for a response box that sends digits. Choosing calls no engine step; a slide becomes the engine's Discrete(32) action, logged as `env_action` (-1 for a choose or a wrong-axis press). The instructions' Controls text is kept in sync with `keys` by hand. Reference: https://github.com/chrplr/Rush-Hour/blob/main/python/src/rushhour_gym/human.py

White triangles on the selected car show which of its two slides are legal (from the engine's `action_mask`), because a refused move looks exactly like a dropped keypress. `movable_only` true (Rush-Hour's `-movable-only` default) makes selection skip cars that cannot move at all; false makes noticing which cars are stuck part of the search, a different task, so keep it the same across sessions being compared.

## Curricula

| file | plays |
|---|---|
| `all.json` | the complete session of Rush-Hour's own program: all 49 puzzles of the library |
| `all-ABXY-nav.json` | the same 49 puzzles on an ABXY gamepad rig, from `common/rushhour-ABXY-nav.json`: L/R choose the car, X/B/Y/A slide it, hold L for the pause menu |
| `easy.json` | 5 min of random easy puzzles (`min_moves_range` 0-12), one after another |

## `all.json`

It ports the participant flow of https://github.com/chrplr/Rush-Hour `main.go`: one game phase per puzzle, each blocking (`advance_on` `["won"]`: a puzzle counts once solved, which is also the only way the engine ends one). The first 12 puzzles of the library come in its order (easiest first, what Rush-Hour's `-n` presents), then the other 37 (p13-p49) once each in a fixed shuffled order (`random.Random(20260913).shuffle`); `puzzle` names each.

Each puzzle is preceded by a blank 0.8 s message (the ITI), opens on its instructions (headed "Puzzle i of N", over the rules, until SPACE), and is followed by the block's own result screen ("Final score" and "Puzzle solved", for `show_outcome_duration` seconds, default 2; text on the plain background, not the solved board Rush-Hour holds). Each puzzle is its own block, with its own log and manifest entry; the puzzle number is the phase's place in the list, as the board carries no counter. The board is drawn like rushui (same layout, palette, white outline and arrows on the selected car), and the logged variables carry the columns of Rush-Hour's results file (`event`, `car`, `from_*` / `to_*`, `n_slides`, `solved`, `t_ms`, `trial_ms`).

Rush-Hour stamps keypresses and flips on SDL's monotonic clock; here a key is timestamped when the loop processes it, within 1/`fps`, hence `fps` 60 (a turn-based block steps only on a keydown, so the rate costs nothing). `duration` is per puzzle and effectively unlimited; ESC is the experimenter's way out.
