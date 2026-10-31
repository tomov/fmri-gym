# Crafter curricula

The `crafter` backend's curricula: the paradigm's four levels, one file each. All four include `common/crafter.json` (the eight-button interface, the pacing, the HUD, the cues, the logging) and state only their level (`env_kwargs.level`), their instructions, and, on level 1, the rewind. Everything below the level table applies to all four.

## Setup

`uv sync --extra crafter` installs `crafter-gym` (`gym/crafter/`), which pins crafter to the `chengfanbrain/crafter@deterministic` fork: crafter 1.8.3 with one change, so that a seed plus an action list replays exactly (see "State"). Plain PyPI crafter plays identically for a subject, but its blocks cannot be replayed, so a model could not be given the same episode.

## Levels

The levels are `crafter_gym.levels.LEVELS`, the rig's own table, applied as `gymnasium.Wrapper`s shipped with the env, as the menu is (`gym/crafter/crafter_gym/menu.py`): a level an adapter applied would stop humans and models from playing the same game. On every level health cannot regenerate from 0 and the view is tinted red while the player is dead. The task chain (`crafter_gym.tasks`) names one entry at a time above the game.

| file | level | |
|---|---|---|
| `L1.json` | 1, affordance | Nothing hostile; the four life stats are frozen at 9 and hidden, so only lava can kill. Eleven tasks. `rewind_on` `["lost"]` with `rewind_frames` 1 is the rig's rollback: a death sends the world back one move and play goes on, as a new episode from that state. It is a phase field rather than a level rule, so a model played on the same env dies as the game says. |
| `L2.json` | 2, homeostasis | The homeostat runs and is shown; nothing hostile. Fifteen tasks, drink / eat / sleep among them once a stat first drops to 8 of 9. |
| `L3.json` | 3, predation | The homeostat runs and zombies and skeletons spawn. Seventeen tasks: drink / eat / sleep once a stat first drops to 8 of 9, and defeating a zombie or a skeleton once one is on screen. |
| `L4.json` | 4, survival | Stock crafter, rule for rule, with no task chain: hostiles on, health / food / drink / energy decay, death ends the episode, nothing hidden from the HUD. |

## One interface

There is one crafter interface, `CrafterMenu-v0`, because the scanner has a button box and not a keyboard: the eight buttons are the ones the rig's own frontend uses (crafter-for-brain-scan v0.33, `core.py:ButtonMapper`, `frontend_pygame.py:KEYMAP`), so a subject who practised in the rig keeps their fingers. `Crafter-v0` (crafter's seventeen actions, no menu) is registered too and is what `agents/` can be pointed at; `CrafterMenu-v0` is the same env under `crafter_gym/menu.py`.

## Keys and menu

Crafter's Discrete(17): 0=noop, 1=move_left, 2=move_right, 3=move_up, 4=move_down, 5=do, 6=sleep, 7=place_stone, 8=place_table, 9=place_furnace, 10=place_plant, 11=make_wood_pickaxe, 12=make_stone_pickaxe, 13=make_iron_pickaxe, 14=make_wood_sword, 15=make_stone_sword, 16=make_iron_sword. The menu wrapper's action space is Discrete(19): those 17, then cycle (17) and confirm (18).

Eight keys are bound, the rig's own: the arrows move, SPACE does (interact / attack / collect), S sleeps, X cycles the menu and A performs the item under the cursor. Cycle and confirm are written as the literal ids 17 and 18, since they are not crafter actions and nothing else names them. There is no `""` entry: `is_turn_based` steps only on a press. The ten menu actions are not listed anywhere: they are whatever the action space has left over, so a crafter that gained an action would put it in the menu rather than drop it. The instructions' Controls text is kept in sync with `keys` by hand.

X advances the cursor and steps the env with noop; A performs the action the cursor is on; both draw the cursor over the player for that one frame. A look costs a turn, as it does in the rig: a free cycle would make all ten menu actions reachable for one press. The log records the button in `actions` and the engine's action in `env_action`, plus `menu_idx` and `menu_sel` read after the press, and `action_names` covers all 19 so the button column decodes. Two deliberate differences from the rig: the cursor shows while the last press was cycle or confirm rather than for 3 wall-clock seconds (`--menu-s`), because this way the block is reconstructible from the logged actions alone; and the cursor sits at 3.5/9 of the frame height, the centre of the played area above crafter's own two HUD rows, as in the rig.

## Pacing

`is_turn_based` is true, the one place this parts company with the rig. Eight buttons cost more presses per action than seventeen keys, so a fixed cadence would spend the survival clock on menu cycling; here the world advances once per press and never otherwise, so the run is paced by the subject and every step is a decision. `fps` 15 is then a polling rate, not a game speed: it bounds how long a press waits to be seen (~67 ms), as in the other turn-based DBP curricula. `has_latched_keys` is irrelevant under `is_turn_based` (the loop already reads keydowns) and is off. The rig has no turn-based mode: it runs a fixed 5 Hz cadence at half speed (`frontend_pygame.py --cadence 5.0 --speed 0.5`, 2.5 steps/s), so wall-clock comparisons with a rig run do not transfer; only the action sequences do.

## Sleep

Sleep is where `is_turn_based` and stock crafter meet, and it is settled in the loop rather than in the rules, so the rules stay stock. Crafter takes no action from a sleeping player: `Player.update` overwrites it with sleep until energy is full (`objects.py:99-131`), and the step on which it fills is the one that wakes them and runs their action normally. Stock crafter charges eleven sleeping steps per point of energy, so three to a full nine is sixty-six steps: sixty-six presses in a block where a step costs a press, every one thrown away by the engine. The adapter's autoplay returns noop while `info['sleeping']` is set, and the loop steps at `fps` without waiting, so a sleep costs one press and about four seconds. Nothing about the game changes: the night passes at its own rate, hunger, thirst and health move at their sleeping rates, a zombie still closes in and deals 7 to a sleeper against 2 awake (`objects.py:305-308`), and the frames are drawn and logged like any others, action noop, with a `sleeping` column marking them. A blow clears sleeping (`_wake_up_when_hurt`, `objects.py:169-172`), so the picture stops on the frame with the zombie beside the player and the health bar already down. Over 40 worlds asleep from step 150 (night, energy 0), a blow lands inside 66 steps in 32: sleeping is a real decision.

## HUD

`show_score` draws the achievement count in the HUD strip above the frame, which crafter's own HUD does not show. It covers no game pixel and reports only values already in `info['achievements']`, so the model harness can be given the same scalar; a model evaluated without it sees less than the subject did, and that should be said. The count is shown rather than the block's cumulative reward, which is the same quantity written less legibly: crafter pays +1 per first unlock and (health - last_health)/10 per step, and the health terms telescope to at most -0.9 over an episode.

## Cues

`cues` turns on three synthesized sounds in the adapter, because crafter ships none and three outcomes are otherwise invisible in the bore: a blow that landed on a creature without killing it (five bare-handed presses for a zombie), a press the engine refused (no pickaxe for that rock, no table in reach), and a press the rig dropped. Priority is score > hit > blocked, at most one a frame, since audio output queues rather than mixes. The cue is decided from the state the engine is about to act on, not by diffing state afterwards, which cannot see a non-lethal blow; it is exact because the player updates first every step, and it reads crafter's own collect / place / make tables without touching the world or its RNG. Collecting grass stays silent on purpose: `data.yaml` gives it probability 0.1, so a press that yields nothing is a failed roll, not a refusal. `cue_overlay` writes the same feedback as a field in the HUD strip and is off: the subject has already heard the cue, and a word appearing above the frame pulls the eyes off it between presses. The agent harness turns it on for itself, since a policy cannot hear. The columns `hit` / `no_effect` / `cue` / `target` are logged however both flags are set.

## State

`state_stride` 25 counts frames, and a frame is a press, so anchors fall every 25 presses. At `size` 384 a state is 2.3 MB and ~3 ms to write or read, of which 0.16 MB is the game (the rest is the observation space's constant bounds and the cached frame); zlib leaves ~57 KB an anchor. A 300 s block played in real time at 2.5 fps held 32 anchors, 76 MB raw and 1.8 MB compressed, and every one restored and then played its episode out frame for frame identically to the seed replay: that is what makes an anchor a branch point a model can be rolled out from. A turn-based block of the same duration has far fewer frames. `frames.h5` keeps the displayed pixels too (per-frame zlib: 7.4 KB median in daylight; crafter mixes per-pixel noise into the view at night, so night frames reach a median 185 KB). Frames are kept even though the fork replays exactly, because they are the record that does not depend on whoever opens the block having the fork installed. On stock crafter they are the only record: it rebalances creatures every tenth step by iterating a Python set, so the despawn choice follows object `id()` and two runs of the same seed and actions diverge (measured 2026-09-28 across two `PYTHONHASHSEED`s: step 30 of a 225-action episode); terrain is identical either way, creatures are not.

## Size

`env_kwargs.size` 384 is an integer half of the 768-pixel display height, not the rig's `GAME_PX` 288: `display.draw_frame` scales unfiltered, so a fractional factor smears crafter's pixel art, the counters in its bottom two tile rows first. Nothing about the game depends on it; crafter renders its 9x9 view at whatever size it is given. `length` 0 removes crafter's own 10000-step cap: the block is bounded by its duration, so every ending is a real death.
