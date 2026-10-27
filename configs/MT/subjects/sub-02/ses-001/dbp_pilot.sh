#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SUBJECT=sub-02
RIG=configs/examples/rigs/laptop.json
SES=${1:-001}
# uv run fmri-play --curriculum configs/curricula/coom/all.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
# uv run fmri-play --curriculum configs/curricula/coom/all.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/vizdoom/take_cover_defend_line.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/vizdoom/take_cover_defend_line.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game4.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game4.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game7.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game7.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game8.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game8.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/crafter/levels/L1.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/crafter/levels/L1.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/rushhour/complete.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/rushhour/complete.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/baba_auto/all.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/baba_auto/all.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/minihack/all.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/minihack/all.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/stk/all.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 60 --loop
uv run fmri-play --curriculum configs/curricula/stk/all.json --subject "$SUBJECT" --ses "$SES" --run 2 --rig "$RIG" --duration 60 --loop
