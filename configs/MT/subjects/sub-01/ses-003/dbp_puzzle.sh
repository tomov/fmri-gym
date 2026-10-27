#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SUBJECT=sub-01
RIG=configs/examples/rigs/laptop.json
SES=${1:-003}
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game2.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game3.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game4.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game5.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game6.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game7.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game8.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/game9.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/rushhour/levels/easy.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/baba/levels/make_win.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
