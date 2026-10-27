#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SUBJECT=sub-01
RIG=configs/examples/rigs/laptop.json
SES=${1:-002}
uv run fmri-play --curriculum configs/curricula/coom/levels/pitfall.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/arms_dealer.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/floor_is_lava.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/hide_and_seek.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/chainsaw.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/raise_the_roof.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/run_and_gun.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/health_gathering.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/parkour.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/vizdoom/levels/defend_center.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/vizdoom/levels/deadly_corridor.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
