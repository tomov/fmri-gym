#!/bin/sh
# fmri-gym session: one line per run, in order.

set -e  # Exit on error

# Session parameters
SUBJECT=sub-01
RIG=configs/examples/rigs/xbox.json
SES=2
RUN_DURATION=180

# Runs, one per line

uv run fmri-play --subject $SUBJECT --ses $SES --run 1 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/gym/classic_control-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 2 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/ale/mspacman-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 3 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/ale/spaceinvaders-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 4 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/retro/airstriker-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 5 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/minihack/mazewalk9x9-GMPD3-stick-nav.json
