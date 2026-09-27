# baba-gym

[Baba Is You](https://github.com/nacloos/baba-is-ai) (`baba`) as a Gymnasium
env. `baba` speaks the old `gym` API -- `reset` returns the observation alone,
`step` a 4-tuple, `render` takes a mode -- so this holds one `baba` env and
presents the Gymnasium contract in front of it. The game is untouched.

```bash
pip install -e .            # from this directory; pulls baba from GitHub
```

```python
import baba_gym
env = baba_gym.BabaEnv("env/make_win")     # or gym.make("Baba-v0", game="env/make_win")
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(1)   # up
frame = env.render()                        # (256, 256, 3) uint8
```

Actions are `BabaIsYouEnv.Actions`: 0 = idle, 1 = up, 2 = right, 3 = down,
4 = left. The observation is the grid's own `(H, W, 3)` encoding; the picture
is `render()`. No savestate: an episode replays from its seed and actions.
