#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-01)}
uv run fmri-play --curriculum configs/dbp_games/coom__pitfall.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__arms_dealer.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__floor_is_lava.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__hide_and_seek.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__chainsaw.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__raise_the_roof.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__run_and_gun.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__health_gathering.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__parkour.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/vizdoom__defend_center.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/vizdoom__deadly_corridor.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
