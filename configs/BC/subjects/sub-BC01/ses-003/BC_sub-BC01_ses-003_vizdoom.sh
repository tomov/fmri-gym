#!/bin/sh
# fmri-gym session: one line per run, in order.

set -e  # Exit on error

# Session parameters
SUBJECT=sub-BC01
RIG=configs/BC/rigs/3T_MIT_GMPD3.json
SES=3
RUN_DURATION=600

# The session's recording (session-logger): started first, stopped when the script ends.
RECORDING=$(uv run session-logger start --subject "$SUBJECT" --ses "$SES" --rig "$RIG" --session-pid $$)
trap 'uv run session-logger stop "$RECORDING"' EXIT
trap 'exit 130' INT TERM HUP

# Runs, one per line

#uv run fmri-play --subject $SUBJECT --ses $SES --run 1 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

#uv run fmri-play --subject $SUBJECT --ses $SES --run 2 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

#uv run fmri-play --subject $SUBJECT --ses $SES --run 3 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

#uv run fmri-play --subject $SUBJECT --ses $SES --run 4 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 5 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 6 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 7 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 8 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 9 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json
