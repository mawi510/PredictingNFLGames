"""
Build the curated ~45-feature dataset from the existing 1001-col CSVs.

Implements the decisions in research/feature_shortlist.md:
  - Situational + market columns verbatim.
  - Team form (offense + defense) — _rolling_shifted only.
  - Pressure/line play, QB (QBR), ATS/W-L records.
  - Drop: rank_*, expand_* (except where the only available form), raw unshifted,
    receiving family, wopr family, 2pt/fumble counters.
  - ADD: matchup-relative features
      passing_epa_minus_opp_passing_epa_allowed
      rushing_epa_minus_opp_rushing_epa_allowed
    Built by joining each row to its schedule opponent (pipeline.margin_model).

Inputs:
    nfl_training_data.csv  (2018-2023)
    nfl_current_data.csv   (2024)

Output:
    research/xgboost/features.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
OUT = Path(__file__).resolve().parent / "features.parquet"

SITUATIONAL = [
    "spread", "spread_odds", "moneyline", "exp_total_points", "over_odds", "under_odds",
    "is_home", "is_fav", "rest", "div_game",
    "dome", "outdoors", "closed",
    "grass", "fieldturf", "astroturf",
    "players_injured_adj",
]

OFFENSE_ROLLING = [
    "points_scored_rolling_shifted", "scoring_margin_rolling_shifted",
    "total_passing_epa_rolling_shifted", "avg_passing_epa_rolling_shifted",
    "total_rushing_epa_rolling_shifted", "avg_rushing_epa_rolling_shifted",
    "passing_yards_rolling_shifted", "rushing_yards_rolling_shifted",
    "passing_tds_rolling_shifted", "rushing_tds_rolling_shifted",
    "interceptions_rolling_shifted", "sacks_rolling_shifted",
    "dakota_rolling_shifted",
]

DEFENSE_ROLLING = [
    "points_allowed_rolling_shifted", "scoring_margin_allowed_rolling_shifted",
    "total_passing_epa_allowed_rolling_shifted",
    "total_rushing_epa_allowed_rolling_shifted",
    "passing_yards_allowed_rolling_shifted",
    "rushing_yards_allowed_rolling_shifted",
    "interceptions_allowed_rolling_shifted",
]

PRESSURE = [
    "times_pressured_pct_rolling_shifted",
    # NO _allowed_rolling — using _allowed_shifted instead (single-week, prior-week pass rush %)
    "times_pressured_pct_allowed_shifted",
    "passing_bad_throw_pct_rolling_shifted",
    "rushing_yards_before_contact_avg_rolling_shifted",
    "rushing_yards_after_contact_avg_rolling_shifted",
]

QB = [
    "qbr_total_adj_rolling_shifted",
    "epa_total_adj_rolling_shifted",
    "qbr_total_adj_allowed_rolling_shifted",
]

RECORDS = [
    "is_cover_record_shifted", "is_winner_record_shifted",
    "is_over_record_shifted", "is_fav_record_shifted",
]

# columns used internally to pair opponents (not features themselves)
ID_COLS = ["season", "week", "team", "is_cover"]

FEATURE_COLS = SITUATIONAL + OFFENSE_ROLLING + DEFENSE_ROLLING + PRESSURE + QB + RECORDS


def load_raw() -> pd.DataFrame:
    use = ID_COLS + FEATURE_COLS
    hist = pd.read_csv(REPO_ROOT / "nfl_training_data.csv", usecols=lambda c: c in use)
    curr = pd.read_csv(REPO_ROOT / "nfl_current_data.csv", usecols=lambda c: c in use)
    df = pd.concat([hist, curr], ignore_index=True)
    df["is_cover"] = df["is_cover"].astype(int)
    return df.sort_values(["season", "week", "team"]).reset_index(drop=True)


def attach_opponent_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair each row with its schedule opponent and compute matchup features:
        passing_epa_minus_opp_passing_epa_allowed
        rushing_epa_minus_opp_rushing_epa_allowed
    """
    from pipeline.margin_model import attach_matchup_features

    return attach_matchup_features(df).drop(columns=["opponent"])


def main():
    df = load_raw()
    print(f"Loaded {len(df):,} rows, {len(FEATURE_COLS)} pre-matchup features")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing cols: {missing}")
    df = attach_opponent_matchup(df)
    final_features = FEATURE_COLS + [
        "passing_epa_minus_opp_passing_epa_allowed",
        "rushing_epa_minus_opp_rushing_epa_allowed",
    ]
    print(f"Final feature count: {len(final_features)}")
    out = df[ID_COLS + final_features].copy()
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} -- shape {out.shape}")
    print("\nSeason coverage:")
    print(out.groupby("season").size().to_string())
    print("\nNaN counts per feature (top 10):")
    print(out[final_features].isna().sum().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
