#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SUBJECT=sub-01
RIG=configs/examples/rigs/laptop.json
SES=${1:-001}
uv run fmri-play --curriculum configs/curricula/crafter/levels/L4.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/pitfall.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/minihack/levels/room5x5.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/rushhour/levels/easy.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/baba/levels/make_win.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/vizdoom/levels/defend_center.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/run_and_gun.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/minihack/levels/mazewalk9x9.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game2.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/coom/levels/chainsaw.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
