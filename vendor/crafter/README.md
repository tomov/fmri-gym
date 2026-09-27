# crafter-gym

[Crafter](https://github.com/danijar/crafter) as a Gymnasium env. `crafter.Env`
speaks the old `gym` API -- `reset` returns the observation alone, `step` a
4-tuple -- and is seeded at construction only, so this holds one `crafter.Env`,
presents the Gymnasium contract in front of it, and rebuilds the game on
`reset(seed=)`. The game is untouched.

```bash
pip install -e .            # from this directory
```

```python
import crafter_gym
env = crafter_gym.CrafterEnv(size=(512, 512))    # or gym.make("Crafter-v0", size=(512, 512))
obs, info = env.reset(seed=0)                    # obs is the (512, 512, 3) frame
obs, reward, terminated, truncated, info = env.step(5)   # do
```

Keyword arguments are `crafter.Env`'s: `area`, `view`, `size`, `reward`,
`length`, `seed`. Actions are crafter's Discrete(17): 0 = noop, 1..4 = move
left/right/up/down, 5 = do, 6 = sleep, 7..10 = place stone/table/furnace/plant,
11..16 = make wood/stone/iron pickaxe, wood/stone/iron sword. `info` carries
`inventory`, `achievements`, `player_pos`, `semantic`. No savestate: an
episode replays from its seed and actions.
