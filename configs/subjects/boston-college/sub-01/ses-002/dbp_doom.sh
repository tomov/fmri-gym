#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-002}
uv run fmri-play --curriculum configs/curricula/coom/levels/pitfall.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/arms_dealer.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/floor_is_lava.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/hide_and_seek.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/chainsaw.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/raise_the_roof.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/run_and_gun.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/health_gathering.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/coom/levels/parkour.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/vizdoom/levels/defend_center.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/vizdoom/levels/deadly_corridor.json --subject sub-01 --ses "$SES" --run 1 --duration 300 --rig configs/rigs/example/laptop.json
