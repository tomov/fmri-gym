# arc3-gym

[ARC-AGI-3](https://arcprize.org/arc-agi/3) games (turn-based puzzles on a 64x64 grid of 16 colours, in several levels) as a Gymnasium env, through the `arc-agi` toolkit run in its offline mode. Needs Python 3.12 or later.

```bash
pip install -e .                  # from this directory
python -m arc3_gym ls20 tr87      # once, with the network on: downloads the games' files
```

```python
import arc3_gym
env = arc3_gym.Arc3Env("ls20")    # or gym.make("Arc3-v0", game="ls20")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)   # the game's first simple action
frame = env.render()              # (512, 512, 3) uint8
```

The game files go to `external/arc3` under the fmri-gym root, or to the `environments_dir` you pass; playing makes no network call and needs no API key. The action space is `Discrete(k)` over the game's simple actions in ascending order (`ACTION1..5`, then `ACTION7` undo, those the game offers). Games that offer `ACTION6` (a click) are refused. The reward is the number of levels completed by the step, and the episode ends on `WIN` or `GAME_OVER`. Deterministic: an episode replays from its seed and actions.
