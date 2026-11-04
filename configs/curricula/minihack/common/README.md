# MiniHack curricula

The `minihack` backend's curricula: one MiniHack task per file. A task's two views share their game phase in `common/<task>.json` (keys, instructions, settings) and state only the env id; the designed levels stand alone.

## Setup

`uv sync --extra minihack` (MiniHack, NLE and `minihack_levels`; NLE needs `setuptools<81`). Reference: https://github.com/facebookresearch/minihack

## Two views of each task

Each standard task has two curricula on the same level:

- `<task>.json` plays `minihack_levels`' `-Allocentric-v0` twin of the MiniHack id, whose `pixel_crop` is fixed on the level's map: the level holds still and the agent moves.
- `<task>_egocentric.json` plays MiniHack's own id, whose `pixel_crop` is a 9x9 window centred on the agent: the agent holds still and the level slides.

## Keys

Every phase is `is_turn_based`: action 0 (move N) is also the noop, so a real-time loop would walk the agent north on its own. Arrows map to the cardinals; the instructions' Controls text is kept in sync with `controls` by hand.

| shared phase | action space | keys |
|---|---|---|
| `room5x5`, `room15x15`, `mazewalk9x9`, `river` | Discrete(8) compass: 0=N, 1=E, 2=S, 3=W, 4=NE, 5=SE, 6=SW, 7=NW | arrows, and A/S/Z/C for the diagonals NW/NE/SW/SE, laid out as on the keyboard |
| `corridor` | Discrete(11): compass 0-7, 8=OPEN, 9=KICK, 10=SEARCH | arrows, NetHack's Y/U/B/N for the diagonals, and O/K/S for open, kick and search: more than six buttons, so keyboard only |
| `eat` | Discrete(85), the full command set, of which movement and 29=EAT are mapped | arrows, A/S/Z/C, and SPACE to eat |
| designed levels | Discrete(4): 0=N, 1=E, 2=S, 3=W | arrows only |

**Eating** is two NetHack steps: SPACE (eat), then NetHack asks "eat it? [ynq]" and the player answers A, the NW diagonal, which sends NetHack's y. The prompt is drawn on the tty message line, which is not part of the pixel observation, so the game seems to freeze after SPACE; the instructions give the SPACE-then-A recipe. In NetHack y and n are also the NW and SE movement keys, so Y (action 7, 'y') and N (action 5, 'n') double as yes and no.

## Blocking

Each block is its task played until completed (`mode` `episode`, `n_episodes` 1, `advance_on` `["won"]`): a death or the env's own step limit restarts the same level, and `duration` caps the block (300 s for the standard tasks, 600 or 900 s for the designed levels). ESC is the experimenter's way out.

## Designed levels

`room_*`, `mazewalk_*` and `river_*` other than the standard tasks are custom MiniHack layouts from `minihack_levels`, with no monsters or traps, four-direction movement, and a fixed `seed`. Holding X for 1.5 s opens the pause menu (reset, forfeit, resume; UP/DOWN to move, SPACE to pick). In the river levels unfilled water cannot be entered, and filling it with boulders keeps the game's native randomness: the seed reproduces a tested route, not every possible action sequence.

## Curricula

| file | plays |
|---|---|
| `corridor.json` | MiniHack-Corridor-R2-Allocentric-v0 |
| `corridor_egocentric.json` | MiniHack-Corridor-R2-v0 |
| `eat.json` | MiniHack-Eat-Allocentric-v0 |
| `eat_egocentric.json` | MiniHack-Eat-v0 |
| `mazewalk9x9.json` | MiniHack-MazeWalk-9x9-Allocentric-v0 |
| `mazewalk9x9_egocentric.json` | MiniHack-MazeWalk-9x9-v0 |
| `mazewalk_branches.json` | MazeWalk: Branches |
| `mazewalk_loops.json` | MazeWalk: Loops |
| `river.json` | MiniHack-River-Allocentric-v0 |
| `river_egocentric.json` | MiniHack-River-v0 |
| `river_commit_to_a_route.json` | River: Commit to a Route |
| `river_four_stone_warehouse.json` | River: Four-Stone Warehouse |
| `river_reposition.json` | River: Reposition |
| `river_three_deliveries.json` | River: Three Deliveries |
| `room5x5.json` | MiniHack-Room-5x5-Allocentric-v0 |
| `room5x5_egocentric.json` | MiniHack-Room-5x5-v0 |
| `room15x15.json` | MiniHack-Room-15x15-Allocentric-v0 |
| `room15x15_egocentric.json` | MiniHack-Room-15x15-v0 |
| `room_beyond_the_next_wall.json` | Room: Beyond the Next Wall |
| `room_find_the_connections.json` | Room: Find the Connections |
| `room_look_before_moving.json` | Room: Look Before Moving |
| `all.json` (in `minihack/`) | room5x5, room15x15, mazewalk9x9, corridor, river, eat (allocentric) in a row |
