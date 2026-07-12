"""
v2 feature builder — adds the cold-start-handling features per user 2026-05-30:

    prev_season_cover_rate          team's cover rate from season-1 (league avg fallback)
    blended_cover_rate              weight(w)*prev_season + (1-weight(w))*current_record
                                    where weight(w) = exp(-w / tau)
    weight_on_prior_season          the actual weight used (for debuggability)

Intuition (user's framing): "by week 4 the prior should be from the current season,
not the prior season. The previous season prior is best for cold start."

With tau=3 (default):
    week 1: weight=1.0   (all prior season)
    week 2: weight=0.72  (mostly prior)
    week 3: weight=0.51
    week 4: weight=0.36
    week 5: weight=0.26
    week 8: weight=0.07  (essentially all current season)

Reads features.parquet (built by build_features.py) and the labels, augments, and writes
features_v2.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "features.parquet"
OUT = ROOT / "features_v2.parquet"

TAU = 3.0
LEAGUE_FALLBACK = 0.5  # pushes-as-loss makes the actual league rate ~0.47, but 0.5
                       # is the neutral fallback when the team didn't exist last year


def main():
    df = pd.read_parquet(INPUT)

    # Need raw label history per team-season to compute prev_season_cover_rate.
    labels = pd.read_csv(REPO_ROOT / "nfl_training_data.csv", usecols=["season", "team", "is_cover"])
    labels_curr = pd.read_csv(REPO_ROOT / "nfl_current_data.csv", usecols=["season", "team", "is_cover"])
    labels = pd.concat([labels, labels_curr], ignore_index=True).dropna(subset=["is_cover"])
    labels["is_cover"] = labels["is_cover"].astype(int)

    team_season = labels.groupby(["season", "team"])["is_cover"].mean().reset_index()
    team_season = team_season.rename(columns={"is_cover": "cover_rate"})

    # Shift: prev_season for (s, T) = (s-1, T)'s cover rate
    team_season["prev_season"] = team_season["season"] + 1
    prev = team_season.rename(columns={"season": "_dropme", "prev_season": "season",
                                       "cover_rate": "prev_season_cover_rate"})
    prev = prev[["season", "team", "prev_season_cover_rate"]]

    df = df.merge(prev, on=["season", "team"], how="left")
    df["prev_season_cover_rate"] = df["prev_season_cover_rate"].fillna(LEAGUE_FALLBACK)

    # blended: weight = exp(-week / tau), applied to prev_season vs is_cover_record_shifted
    df["weight_on_prior_season"] = np.exp(-df["week"] / TAU)
    # at week 1, is_cover_record_shifted is NaN (no shifted data) -> treat as prev_season
    cur = df["is_cover_record_shifted"].fillna(df["prev_season_cover_rate"])
    df["blended_cover_rate"] = (
        df["weight_on_prior_season"] * df["prev_season_cover_rate"]
        + (1 - df["weight_on_prior_season"]) * cur
    )

    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} -- shape {df.shape}")
    print(f"\nNew features added: prev_season_cover_rate, weight_on_prior_season, blended_cover_rate")
    print(f"tau = {TAU}")
    print("\nSpot check — DAL across 2024 by week:")
    spot = df[(df["team"] == "DAL") & (df["season"] == 2024)][
        ["season", "week", "team", "prev_season_cover_rate", "is_cover_record_shifted",
         "weight_on_prior_season", "blended_cover_rate"]
    ].sort_values("week").head(10)
    print(spot.to_string(index=False))


if __name__ == "__main__":
    main()
