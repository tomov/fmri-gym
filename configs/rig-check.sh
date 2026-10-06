#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-rig)}
# uv run fmri-play --curriculum configs/rig-check.json --subject sub-rig --ses "$SES" --run 1
uv run fmri-play --curriculum configs/rig-check-long.json --subject sub-rig --ses "$SES" --run 1
