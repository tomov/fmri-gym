# Rush Hour curricula

The `rushhour` backend's curricula: slide cars to free the red one. The 441 puzzles of `library.json` include `common/rushhour.json` (keys, rules text, settings) and state only their puzzle and title; `easy.json`, a different task, stands alone.

## Setup

`uv sync --extra rushhour` (rushhour-gym >= 0.8.0). On first use the package downloads the rushhour-env Go engine of the matching Rush-Hour release into `~/.cache/rushhour-gym/` (checksum-verified); `RUSHHOUR_ENV_BIN` or the phase's `binary` field names a binary of your own. The env is `RushHourHuman-v0`: the Rush-Hour program's own interface (car selection, its picture, its trial flow, its results columns in `info`), all in the package; the adapter is a keymap plus the fields to log.

## Keys

The env's action space is Discrete(12), rushinput's meta-actions: 0-3 choose the car up / down / left / right (spatial, what Rush-Hour's gamepad stick does), 4/5 choose the previous / next car, 6/7 slide it back (left / up) / forward (right / down) along its own axis (the four-button box), and 8-11 slide it left / right / up / down. A slide across the car's axis (LEFT on a vertical car) moves nothing and is logged as event `wrong_axis`. The keys are the rig's ten: A/S choose, the arrows slide by direction, Z/X slide back / forward; 3/4 and 1/2 do the same as in `rushinput.DefaultMap` (`rushhour_gym.human.DEFAULT_KEYS`), for a response box that sends digits. Choosing calls no engine step; a slide becomes the engine's Discrete(32) action, logged as `env_action` (-1 for a choose or a wrong-axis press). The instructions' Controls text is kept in sync with `controls` by hand. Reference: https://github.com/chrplr/Rush-Hour/blob/main/python/src/rushhour_gym/human.py

White triangles on the selected car show which of its two slides are legal (from the engine's `action_mask`), because a refused move looks exactly like a dropped keypress. `movable_only` true (Rush-Hour's `-movable-only` default) makes selection skip cars that cannot move at all; false makes noticing which cars are stuck part of the search, a different task, so keep it the same across sessions being compared.

## Curricula

| file | plays |
|---|---|
| `library.json` | the whole puzzle library, both its blocks: 441 puzzles, one game phase each |
| `library-GMPD3-ABXY-nav.json` | the same 441 puzzles on a controller, from `common/rushhour-GMPD3-ABXY-nav.json`: LT/RT choose the car, X/B/Y/A slide it, hold LT for the pause menu |
| `library-GMPD3-stick-nav.json` | the same on a controller's left stick, from `common/rushhour-GMPD3-stick-nav.json`: LT/RT choose the car, the stick slides it, hold X for the pause menu |
| `all-GMPD3-ABXY-nav.json` | the library's first block alone, its 49 puzzles, on the ABXY controller: p01-p12 in order, then p13-p49 in a fixed shuffled order (`random.Random(20260913).shuffle`) |
| `all-GMPD3-stick-nav.json` | the same 49 puzzles in the same order on the stick controller |
| `easy.json` | 5 min of random easy puzzles (`min_moves_range` 0-12), one after another |

## `library.json`

It ports the participant flow of https://github.com/chrplr/Rush-Hour `main.go` to the whole library, its `p` block (one puzzle per minimum number of moves) and its `e` block (several per minimum): one game phase per puzzle, each blocking (`advance_on` `["won"]`: a puzzle counts once solved, which is also the only way the engine ends one). The first 12 puzzles of the `p` block come in its order (one move harder each, what Rush-Hour's `-n` presents), then the other 429 once each in a fixed shuffled order (`random.Random(20261125).shuffle` over their names, sorted), so after the first 12 a puzzle's difficulty does not depend on its place. The order is the same for every subject; `puzzle` names each.

Each puzzle is preceded by a blank 0.8 s message (the ITI), opens on its instructions (headed "Puzzle i of N", over the rules, until SPACE), and is followed by the block's own result screen ("Final score" and "Puzzle solved", for `show_outcome_duration` seconds, default 2; text on the plain background, not the solved board Rush-Hour holds). Each puzzle is its own block, with its own log and manifest entry; the puzzle number is the phase's place in the list, as the board carries no counter. The board is drawn like rushui (same layout, palette, white outline and arrows on the selected car), and the logged variables carry the columns of Rush-Hour's results file (`event`, `car`, `from_*` / `to_*`, `n_slides`, `solved`, `t_ms`, `trial_ms`).

Rush-Hour stamps keypresses and flips on SDL's monotonic clock; here a key is timestamped when the loop processes it, within 1/`fps`, hence `fps` 60 (a turn-based block steps only on a keydown, so the rate costs nothing). `duration` is the cap on each puzzle; ESC is the experimenter's way out. The session is hours long: `--resume` carries it on from the puzzle a run stopped in. The controller versions list the same puzzles in the same order, without the blanks and the closing message.

