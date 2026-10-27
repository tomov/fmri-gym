#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SUBJECT=${SUBJECT:?run as: SUBJECT=sub-01 RIG=configs/<team>/rigs/<rig>.json sh $0 [session]}
RIG=${RIG:?run as: SUBJECT=sub-01 RIG=configs/<team>/rigs/<rig>.json sh $0 [session]}
SES=${1:-$(uv run fmri-ses --subject "$SUBJECT" --rig "$RIG")}
uv run fmri-play --curriculum configs/curricula/baba_auto/all.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop
uv run fmri-play --curriculum configs/curricula/vizdoom/take_cover_defend_line.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop
uv run fmri-play --curriculum configs/curricula/minihack/all.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop
uv run fmri-play --curriculum configs/curricula/stk/all.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop
uv run fmri-play --curriculum configs/curricula/crafter/levels/L1.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop
