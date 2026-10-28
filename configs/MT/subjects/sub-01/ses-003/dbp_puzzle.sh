#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SUBJECT=sub-01
RIG=configs/examples/rigs/laptop.json
SES=${1:-003}
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game1.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game2.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game3.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game4.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game5.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game6.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game7.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game8.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/aigamestore/games/game9.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/rushhour/levels/easy.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
uv run fmri-play --curriculum configs/curricula/baba/levels/make_win.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
