# vizdoom-gym

Two wrappers over [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)'s own Gymnasium envs (`vizdoom.gymnasium_wrapper`, the `Vizdoom*-v1` ids), and a function for their sound on macOS.

`KeyboardOnlyAction(env)`: a scenario with mouse axes (Deathmatch, the full-game maps) has a `Dict{"binary", "continuous"}` action space; this holds the axes at 0 and exposes the binary part, for play with keys only.

`StartedEpisodes(env)`: ViZDoom starts its first episode with `game.init()` and every later one with `game.new_episode()`, and the two mix an episode's sound slightly differently (pixels and game variables agree). This resets once on construction, so every episode you reset begins the same way and a replay of any of them matches its audio too.

`use_openal_soft()`: ViZDoom's engine makes its sound with OpenAL Soft, which the macOS wheels do not bundle and Homebrew installs keg-only (`brew install openal-soft`), outside dyld's search path; without it, every audio buffer is zeros. On macOS this puts the installed library's directory on `DYLD_FALLBACK_LIBRARY_PATH` and returns it, or returns None when there is none. Call it before the env's first `reset()`, which starts the engine. Off macOS it does nothing.

```python
import gymnasium as gym
from vizdoom import gymnasium_wrapper  # noqa: F401  (registers Vizdoom*-v1)
from vizdoom_gym import KeyboardOnlyAction, StartedEpisodes, use_openal_soft
use_openal_soft()
env = StartedEpisodes(KeyboardOnlyAction(gym.make("VizdoomDeathmatch-v1", render_mode="rgb_array")))
```
