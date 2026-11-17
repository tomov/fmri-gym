# NetHack gamepad

Full NetHack 3.6.0 from the supplied NetHackDS source, with its original tiles,
name/character selection and adventure on one desktop screen. The Gymnasium
environment supplies the same controller interface to humans and AI agents. It
is independent of the existing NLE and MiniHack backends.

## Install and launch

Run these commands from the repository root, using Python 3.10–3.13:

```bash
uv sync --locked --extra nethack-gamepad
uv run nethack-gamepad-build
uv run fmri-play --subject sub-test --ses 1 --run 1 \
    --rig configs/examples/rigs/laptop.json \
    --curriculum configs/curricula/nethack_gamepad/levels/modern.json \
    --dummy-trigger --new-campaign
```

Use `classic.json` instead of `modern.json` for original permadeath. The curriculum
sets `env_kwargs.game_mode`; there is no game-specific flag on `fmri-play`. Use
the scanner's measured rig and omit `--dummy-trigger` for collection. The shared
launcher supplies input calibration, triggers, display, pauses and experiment
recording. Its `--duration`, `--resume` and subject/session/run options retain
their standard meanings. Both example curricula enable `continue_campaign`.
Use `--new-campaign` only to start a new character/adventure; omit it on later
visits to restore the latest checkpoint and start a fresh scan duration. The
experimenter screen, console and manifest identify creation or continuation.

The explicit build needs `cc` and `make` (on macOS, install Apple's command line
tools). No Nintendo SDK, emulator, ROM, bison, flex or source download is needed.
The build copies source into `~/.cache/fmri-gym/3.6.0-desktop-4-cursor/source` and
writes a runtime beside it. `NETHACK_GAMEPAD_HOME` overrides that cache directory;
use a path without whitespace because of upstream Makefile restrictions.
`runtime=` selects a runtime when constructing the environment. Game startup and
stepping never compile or download. Keep the checkout for its editable package.

For standalone testing without experiment logs:

```bash
uv run nethack-gamepad
uv run nethack-gamepad --mode classic
uv run nethack-gamepad --seed 1000
```

The standalone player defaults to Modern mode. Without an explicit seed or
campaign file, it uses native saves in `player-modern` or `player-classic` under
the build cache. Enter the same character name to restore a native save; choose
Save from the command list to save, or close the window for native disconnect
saving. Escape closes the standalone player. Rebuilding replaces source/runtime,
not those player folders.

## Controls

Use arrows to select keys on the supplied DS keyboard, Z to type, and its Enter
key to submit a name. NetHack then offers random or manual role, race, gender and
alignment selection, followed by its original introduction.

| Keyboard | Standalone SDL controller | Dungeon | Menus |
|---|---|---|---|
| Arrows | D-pad | Move N/E/S/W | Move focus or scroll |
| Z | A | Pick up | Confirm or toggle selection |
| X | B | Search | Cancel or back |
| C | X | DS command list | Keyboard for text/key prompts |
| V | Y | Inventory | Reserved |

Press C twice from the dungeon for the DS keyboard. Its Shift, Caps, Ctrl and Alt
keys provide native combinations, including diagonal movement. Direction prompts
offer eight compass directions, here, up and down. Multi-select menus have a
Finish selection entry. Arrows scroll documents. The native map cursor remains
visible during invisibility and targeting; in targeting, arrows move it, Z selects
and X cancels. Four face buttons are supported; two-button play needs additional
bindings. The example curricula bind keyboard controls. A scanner-controller
curriculum maps calibrated fmri-gym control names to the same eight actions;
standalone SDL bindings do not configure the experiment rig.

## Classic and Modern modes

`game_mode="classic"` is the environment default and retains permadeath.
`game_mode="modern"` enables a GnollHack-inspired recovery adaptation:

- Death heals fatal conditions and revives near the latest arrival point on the
  current floor, retaining inventory, experience and dungeon state. This is a
  respawn, not a rewind: consumed resources and previous actions remain.
- Stairs, ladders, branch entrances, portals, falls and level teleports update the
  checkpoint, including return visits. The first character position is the first
  checkpoint. Arrival is recorded before stair/fall damage.
- Walking, same-floor teleportation and revival do not replace the checkpoint.
- Recovery uses the nearest valid square on that floor by Manhattan distance,
  excluding occupied/impassable tiles, traps/portals, water, lava, boulders and
  closed doors. If none is valid, it heals in place and logs `recovery_fallback`.
- Brief protection from monster attacks follows recovery. Original life-saving
  amulets take precedence; successful life-saving does not increment revivals.
- Recovery adds no altar logic or map features. Ordinary altars retain their
  original gameplay. There is no return to a previous floor or special endgame
  relocation rule.

The final NetHack score is divided by 3 initially, 9 after one revival, 27 after
two, then 54, 81, 108, etc. Experience is not divided. The player-facing footer
shows mode and revival count; the score divisor remains in scoring and logs.
Quit, escape and ascension end an episode; recoverable deaths do not. This adapts
GnollHack's healing and mortality formula, not its full engine, content or UI.
Floor-entry checkpoints are this backend's rule change, not a GnollHack feature.

## Continuing between scans

The first visit uses `--new-campaign`, as in the launch example above. Later
visits use the same curriculum and subject, with the new session/run numbers:

```bash
uv run fmri-play --subject sub-test --ses 2 --run 1 \
    --rig configs/examples/rigs/laptop.json \
    --curriculum configs/curricula/nethack_gamepad/levels/modern.json \
    --dummy-trigger
```

The latest checkpoint is selected from that subject's recordings of the same
task and phase, regardless of session/run number. Completed and interrupted scans
both hand over their final game state; each new visit gets the full configured
block duration. Prior recordings remain unchanged. The new episode log embeds
its immutable starting state, and the manifest identifies the source block.
Changing the subject, rig data root or curriculum task selects a different history.
Classic and Modern curricula have separate histories.

If no checkpoint exists, startup explains that `--new-campaign` is required.
Missing state, an unclosed latest block, an ended character or incompatible code/
configuration stop startup instead of falling back to an older save or new game.
Use `--new-campaign` deliberately to begin again. Campaign blocks stop when an
episode ends instead of automatically creating another character.

The existing `--resume` remains an explicit alternative for continuing an
interrupted curriculum with its remaining time and phase progress. It cannot be
combined with `--new-campaign`; campaign curricula do not support `--loop`.
For collection, use the scanner rig and omit `--dummy-trigger`. Let the run finish
or quit through its normal interface so the final state is recorded.

Experiment runs use isolated seeded environments. The adapter refuses a mutable
`player_directory`; native save directories are for standalone play. State is
read from the existing `block_end.extra.state` (or `continuation_state` for native
Save) and restored through the same adapter path used by `--resume`. Checkpoint
compatibility is checked before the participant window opens.

`nethack-campaign` remains available for manually exporting a closed block into
an explicit starting-state curriculum. For that advanced workflow, set
`continue_campaign` to `false` in the exported phase before launching it; the
embedded state then supplies its starting point.

For a standalone continuing campaign, run:

```bash
uv run nethack-gamepad --campaign player-001.campaign --seed 1000
```

Close the window or press Escape, then use the same command to continue. For
Classic, add `--mode classic` from the start. Mode changes on loading are rejected.
A new campaign filename starts a new character. A later `--seed` does not replace
an existing campaign's seed.

Checkpoints contain the original seed and full controller history, using the
shared ReplayState wrapper. Restoration replays that history and draws the final
frame, so latency and checkpoint size grow with play time. Measure both in a
realistic multi-session pilot. Seeded episodes fix the calendar to 2021-01-01 UTC
to avoid date-dependent moon/holiday changes; unseeded standalone play uses the
actual date. Checkpoints bind to engine hashes and frontend source: keep the
matching checkout, runtime and logs. Earlier releases, including those under the
previous backend name, require their matching code for old campaigns. Checkpoint
blobs use the repository's pickle-based format and are for trusted local files.

## Gymnasium and recording

```python
import gymnasium as gym

env = gym.make("nethack_gamepad_gym:NetHackGamepad-v0", game_mode="modern")
observation, info = env.reset(seed=1000)
observation, reward, terminated, truncated, info = env.step(1)
env.close()
```

The action space is Discrete(8): up/right/down/left/confirm/back/commands/inventory.
Observations contain `pixels` uint8 (720,1280,3) and `tiles`/`glyphs` int32 (21,79),
with -1 for unreported map cells. Name entry and character creation are part of
the same action stream for agents and people. No role, scenario or step limit is
forced. Reward is zero and the engine is silent.

A menu press can leave dungeon time unchanged; one native reply can advance many
turns through travel, occupations or paralysis. Logged frame/action numbers and
NetHack turn numbers therefore measure different things. The thin adapter uses
the existing experiment loop, logger, cached RGB frames and replay interfaces.

| Recorded variable | Meaning |
|---|---|
| `tiles`, `glyphs` | Copied native maps |
| `native.request` | Input boundary, including turn, HP and hero position |
| `native.windows` | Menus, status, text and cursor row/column |
| `controller` | Focus, keyboard modifiers and pending typed text |
| `engine_reply`, `engine_stepped` | Native reply or local-only navigation |
| `native.mode`, `native.revivals`, `native.score_divisor` | Mode, revival count and score adjustment |
| `native.checkpoint` | Floor arrival coordinates and reason |
| `native.game_events` | Checkpoint/death/revival/fallback events for this press, with sequence numbers |
| `seed`, `calendar_epoch`, `build` | Seed, fixed calendar and runtime manifest/hashes |

The standard logger writes actions/variables to `events.jsonl`, sampled RGB
frames to `frames.h5`, and state anchors with its existing encoding. Phase
`frame_stride` and `state_stride` retain their standard meanings. There are no
changes to shared experiment input, output, clock, logger or replay modules.

## Source and build provenance

[NetHackDS provenance](../../external/nethack_ds/PROVENANCE.json) identifies the
supplied archive by hash; original source and license notices remain intact.
The supplied 3.6.0 fork contains DS changes and is not claimed to be identical to a
current upstream NetHack release. Only a disposable build copy is patched:

1. Register the JSON/stdin desktop window port and matching native tile mapping.
2. Extract DS character/extended-command selection, inventory-choice helpers,
   command definitions and keyboard layout; replace hardware hooks with shims.
3. Add opt-in RNG/calendar hooks, explicit map-targeting input boundaries and the
   optional Modern recovery rules.
4. Correct the old BMP writer's LP64 header fields; generate tiles from the
   supplied tile text with upstream tools and load the supplied DS bitmap font.
5. Use supplied generated parsers and serialize data generation because upstream
   tools share `grep.tmp`.

`runtime/build.json` records engine/bridge/recovery hashes. Native saves are
architecture/version-specific. Limited unchanged GnollHack reference files,
license and archive provenance are in `external/gnollhack_reference`; they are
not compiled. No GnollHack content or artwork is imported. The bridge and copied
NetHack functions are NetHack-derived code under the included NetHack license.

## Checks and limitations

From the repository root, after installing the extra:

```bash
uv run nethack-gamepad-build
uv run nethack-gamepad-build --test-driver --home /tmp/fmri-nethack-gamepad-tests
export NETHACK_GAMEPAD_TEST_HOME=/tmp/fmri-nethack-gamepad-tests
uv run pytest --backend nethack_gamepad --strict-backends --thorough --no-audio
uv run python -m tests.verify_replay --curriculum configs/curricula/nethack_gamepad/levels/modern.json --max-frames 200 --n-episodes 2
```

The isolated test runtime is rejected by normal clients. A missing selected
runtime fails rather than substituting another game. The normal backend suite
covers logging, replay, resume and the experiment loop; the campaign integration
test additionally records a real death and reconstructs a second-session start.
Engine/transport checks can run without Gymnasium or SDL (numpy/Pillow required):

```bash
python -m unittest discover -s gym/nethack_gamepad/tests -p test_native.py -v
python -m unittest discover -s gym/nethack_gamepad/tests -p test_modern_engine.py -v
python -m unittest discover -s gym/nethack_gamepad/tests -p test_entry_recovery.py -v
```

Linux and macOS are the build targets; Windows compilation is not provided.
macOS, physical controllers and scanner presentation timing need target-machine
validation. Before collection, pilot naming, menus, targeting, blindness/
invisibility, stairs, repeated revival, saving and cross-scan continuation, along
with the rig's pause/resume controls, log volume and restoration time.
