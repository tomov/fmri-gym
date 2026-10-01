# crafter-gym

[Crafter](https://github.com/danijar/crafter) as a Gymnasium env. `crafter.Env` speaks the old `gym` API -- `reset` returns the observation alone, `step` a 4-tuple -- and is seeded at construction only, so this holds one `crafter.Env`, presents the Gymnasium contract in front of it, and rebuilds the game on `reset(seed=)`. The game is untouched.

```bash
pip install -e .            # from this directory
```

```python
import crafter_gym
env = crafter_gym.make_plain(size=(512, 512))    # or gym.make("Crafter-v0", size=(512, 512))
obs, info = env.reset(seed=0)                    # obs is the (512, 512, 3) frame
obs, reward, terminated, truncated, info = env.step(5)   # do
```

Keyword arguments are `crafter.Env`'s: `area`, `view`, `size`, `reward`, `length`, `seed`, plus `level` (below). `CrafterEnv(**kwargs)` is still the bare env and takes everything but `level`; `make_plain` is the one that puts a level on, so prefer it. Actions are crafter's Discrete(17): 0 = noop, 1..4 = move left/right/up/down, 5 = do, 6 = sleep, 7..10 = place stone/table/furnace/plant, 11..16 = make wood/stone/iron pickaxe, wood/stone/iron sword. `info` carries `inventory`, `achievements`, `player_pos`, `semantic`. There is no savestate API, but the env pickles whole, so a caller that wants one takes it that way; an episode also replays from its seed and its actions, and replays exactly only on the crafter fork this package pins (see `pyproject.toml`).

## The eight-button menu (`CrafterMenu-v0`)

A button box has fewer buttons than crafter has actions. `MenuWrapper` gives six of them a button of their own (the four moves, `do`, `sleep`) and puts the other ten in a list the player steps through, so the whole tech tree stays reachable from eight buttons. The action space is `Discrete(19)`: crafter's own 17, then `cycle` (17) and `confirm` (18). Cycling spends a turn, stepping the engine with `noop`; otherwise looking through the menu would be free and any of the ten would cost one press. `info` gains `env_action` (what the engine was given), `menu_idx` and `menu_sel` (where the cursor ended up, and the name it is on), so a log keeps both the button and its effect.

```python
env = crafter_gym.make_menu(size=(384, 384))     # or gym.make("CrafterMenu-v0", size=(384, 384))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(17)   # cycle: the cursor moves, the world steps
```

The cursor is game state rather than a display detail: which action `confirm` fires depends on how many times `cycle` was pressed, so an agent evaluated on this interface meets the same cursor a subject does and a replay reproduces it. `direct=` chooses which action ids keep a button. Reach the wrapper's own attributes (`menu`, `menu_idx`, `action_names`) through `make_menu`: `gym.make` hands back an `OrderEnforcing` around the wrapper, and Gymnasium 1.3 forwards no attribute through it, so `env.menu_idx` raises `AttributeError` and `env.unwrapped` skips the wrapper entirely.

## The four levels (`level=`)

The scanner paradigm does not play one game, it plays four, and they differ by rules rather than by worlds. `LevelWrapper` is that table (`crafter_gym.levels.LEVELS`, a port of the rig's own `core.py:LEVELS`), imposed on the game:

| `level` | what it is |
| --- | --- |
| `L1_affordance` | nothing hostile, the four life stats frozen at 9 and hidden from the panel, no homeostatic death: what is left is what the world affords. Lava still kills. |
| `L2_homeostasis` | the homeostat back on and visible, still nothing hostile. |
| `L3_predation` | zombies and skeletons too. |
| `L4_survival` | stock crafter. |

```python
env = crafter_gym.make_menu(size=(384, 384), level="L1_affordance")
obs, info = env.reset(seed=0)        # the frame is drawn after the rules go on
env.env.level, env.env.rules         # "L1_affordance", and its row of LEVELS
```

The level sits innermost, under the menu if there is one: the rules are the game, the buttons are how it is played. It changes no space and no API, so everything that can play stock crafter can play all four. Two rules are shared by every level and are not in the table: health cannot regenerate from 0, and the view is tinted red while the player is dead (the engine's own commented-out tint, and the one death signal that survives a hidden health icon). Leaving `level` out plays stock crafter with neither of them, which is therefore *not* the same as `L4_survival`.

Why a wrapper and not a patch an adapter applies: a level is part of the game, so a model compared against a subject has to meet it too, and a savestate has to carry it. The crafter adapter's savestate is a pickle of the whole env chain, so the rules have to pickle -- which is why they are instances of module-level classes rather than closures over the live player, and why each reaches the rule underneath it through `type(player)` rather than through the player's own attribute, where unpickling would resolve it straight back to itself. `docs/levels_check.py` in the parent repo measures all of this against the engine's own state and pixels, the rig's table included.
