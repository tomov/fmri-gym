# vgdl-gym

The VGDL games of
[tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)
as standard Gymnasium envs. The fork's `VGDLEnv` is already `gymnasium`, but
it is built from files, its `reset`/`step` take a `with_img` flag and no seed,
its `render` opens a pygame window of its own and its `close` quits pygame.
This wraps one of them: a game name and level in, `reset(seed=)`, an
offscreen `render()`, and the display left alone.

```bash
git clone -b dbp https://github.com/tomov/language_and_experience.git ../language_and_experience
pip install -e .            # from this directory
export VGDL_REPO=../language_and_experience
```

```python
import vgdl_gym
env = vgdl_gym.VGDLEnv("aliens", level=0)       # or gym.make("VGDL-v0", game="aliens")
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(5)   # SPACE
frame = env.render()                              # (H, W, 3) uint8
state = env.get_state(); env.set_state(state)     # exact savestate
```

Actions are the fork's fixed order: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT,
4 = NO_OP, 5 = SPACE. `info` carries the symbolic per-cell `state`, the
collision `events_triggered`, `won` and `lose`. Games: `aliens`,
`beesAndBirds`, `avoidGeorge`, `jaws`, `missile_command`, `plaqueAttack`,
`portals`, `preconditions`, `pushBoulders`, `relational`.
