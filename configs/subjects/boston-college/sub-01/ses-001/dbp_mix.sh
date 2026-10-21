#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-001}
uv run fmri-play --curriculum configs/curricula/crafter/levels/L4.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/pitfall.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/minihack/levels/room5x5.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/rushhour/levels/easy.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/baba/levels/make_win.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/vizdoom/levels/defend_center.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/run_and_gun.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/minihack/levels/mazewalk9x9.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game2.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/chainsaw.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
