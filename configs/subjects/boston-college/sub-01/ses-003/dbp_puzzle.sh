#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-003}
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game2.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game3.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/levels/game4.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game5.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/levels/game6.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game7.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game8.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/aigamestore/game9.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/rushhour/levels/easy.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/baba/levels/make_win.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
