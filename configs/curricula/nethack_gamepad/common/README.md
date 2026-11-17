# NetHack gamepad curricula

Install/build once from the repository root:

```bash
uv sync --locked --extra nethack-gamepad
uv run nethack-gamepad-build
```

Launch through the shared experiment interface:

```bash
uv run fmri-play --subject sub-test --ses 1 --run 1 \
    --rig configs/examples/rigs/laptop.json \
    --curriculum configs/curricula/nethack_gamepad/levels/modern.json \
    --dummy-trigger --new-campaign
```

`levels/classic.json` selects original permadeath; `levels/modern.json` selects
floor-entry revival. Mode is the game phase's `env_kwargs.game_mode`. For the
scanner, select its rig and omit `--dummy-trigger`. The optional `nethack-gamepad`
player is for standalone play without experiment logging.

Actions 0–7 are up/right/down/left/confirm/back/commands/inventory. These examples
bind arrows and Z/X/C/V; a controller curriculum maps calibrated control names to
the same indices. Name entry and character creation remain part of gameplay.

For subsequent visits omit `--new-campaign` and use the new session/run numbers.
The same subject and curriculum automatically continue from the last checkpoint
with a fresh scan duration. Missing checkpoints require explicit creation; an
invalid or unclosed latest recording is not silently skipped. `--resume` retains
its existing interrupted-curriculum meaning (remaining time). See the
[environment README](../../../../gym/nethack_gamepad/README.md) for controls,
mode rules, continuation commands, recorded fields and validation commands.
