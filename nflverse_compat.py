"""Compatibility shim for nflverse's player-stats rebuild.

nflverse stopped publishing new seasons to the `player_stats` release that
nfl_data_py's import_weekly_data() downloads from — 2024 is the last season
there, and nfl_data_py itself is archived. Seasons 2025+ only exist in the
`stats_player` release, which uses a new schema. This module serves the old
interface from both sources so the data scripts keep working unchanged:
seasons <= 2024 come from nfl_data_py (byte-identical to what the models
were built on), 2025+ from the new release with columns renamed back to the
old names.

Verified against 2024 (published in both formats): core team-week aggregates
(yards, TDs, interceptions, sacks) are exact matches; EPA columns match except
rare stat corrections. Known drift in the new source: `dakota` is gone
(kept as NaN below), and `wopr` / `fantasy_points*` are computed differently —
none of these feed the margin v2 model; they only reach the legacy RF's
feature set, where serving maps missing values to 0.0.
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd

# First season that is only available from the new `stats_player` release.
NEW_STATS_FIRST_SEASON = 2025

_NEW_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{year}.parquet"
)

# New-schema column name -> old player_stats name the scripts expect.
_RENAMES = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "sack_yards_lost": "sack_yards",
}


def import_weekly_data(years):
    old = [y for y in years if y < NEW_STATS_FIRST_SEASON]
    new = [y for y in years if y >= NEW_STATS_FIRST_SEASON]
    frames = []
    if old:
        frames.append(nfl.import_weekly_data(old))
    for year in new:
        df = pd.read_parquet(_NEW_STATS_URL.format(year=year))
        df = df.rename(columns=_RENAMES)
        # dakota was dropped in the stats rebuild and has no replacement.
        # Keep the column (as NaN) so downstream feature construction stays
        # schema-stable; serving already maps missing values to 0.0.
        df["dakota"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
