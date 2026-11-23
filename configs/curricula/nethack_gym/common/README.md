# NetHack Gym curricula

Install from the repository root with Python 3.10–3.13. The `nethack_gym` extra installs the matching standalone `nethack-gym` package. This source bundle expects the `nethack-gym/` checkout beside the fmri-gym checkout, as configured in `tool.uv.sources`.

```bash
uv sync --extra nethack_gym
uv run --locked nethack-gym-build
```

Launch through the shared experiment interface:

```bash
uv run fmri-play --subject sub-test --ses 1 --run 1 \
    --rig configs/examples/rigs/laptop.json \
    --curriculum configs/curricula/nethack_gym/levels/modern.json \
    --dummy-trigger --new-campaign
```

`levels/classic.json` selects original permadeath; `levels/modern.json` selects
floor-entry revival. Mode is the game phase's `env_kwargs.game_mode`. For the
scanner, select its rig and omit `--dummy-trigger`. The optional `nethack-gym`
player is for standalone play without experiment logging.

Actions 0–7 are up/right/down/left/confirm/back/commands/inventory. These examples
bind arrows and Z/X/C/V; a controller curriculum maps calibrated control names to
the same indices. Name entry and character creation remain part of gameplay.

For subsequent visits omit `--new-campaign` and use the new session/run numbers.
The same subject and curriculum automatically continue from the last checkpoint
with a fresh scan duration. Missing checkpoints require explicit creation; an
invalid or unclosed latest recording is not silently skipped. `--resume` retains
its existing interrupted-curriculum meaning (remaining time). The standalone `nethack-gym` package provides the `nethack_gym` import, `NetHackGym-v0` environment, and `nethack-gym` and `nethack-gym-build` commands.

The fmri-gym backend and extra are `nethack_gym`, and curricula live under `configs/curricula/nethack_gym/`. These paths determine recording task labels. Existing recordings retain their original labels; use the original checkout and curriculum when continuing or replaying those campaigns.
