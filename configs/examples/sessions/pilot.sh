#!/bin/sh
# fmri-gym session: one line per run, in order.

set -e  # Exit on error

# Session Parameters
SUBJECT=sub-01
RIG=configs/examples/rigs/fmri-curdes-v1-ABXY-nav.json
SES=1
RUN_DURATION=600 # seconds

# Runs, one per line

uv run fmri-play --subject $SUBJECT --ses $SES --run 1 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/aigamestore/game1-ABXY-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 2 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/aigamestore/game4-ABXY-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 3 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/aigamestore/game7-ABXY-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 4 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/aigamestore/game8-ABXY-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 5 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/rushhour/all-ABXY-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 6 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/baba_auto/all-ABXY-nav.json
