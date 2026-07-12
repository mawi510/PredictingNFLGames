"""
Extend the curated feature dataset through the completed 2025 season.

Same construction as build_features.py (which froze features.parquet at
2018-2024), with the 2025 season appended from nfl_2025_season.csv — the
full-season wide CSV produced by the fixed current_nfl_data.py (nflverse
stats_player source; dakota is NaN for 2025, see nflverse_compat.py).

2025 was never pulled during model development (the data pipeline was broken
for it), so it is a true holdout for the margin model.

Output:
    research/xgboost/features_2025.parquet   (2018-2025)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_features import FEATURE_COLS, ID_COLS, REPO_ROOT, attach_opponent_matchup

HERE = Path(__file__).resolve().parent
CSV_2025 = HERE / "nfl_2025_season.csv"
FROZEN = HERE / "features.parquet"
OUT = HERE / "features_2025.parquet"

MATCHUP_COLS = [
    "passing_epa_minus_opp_passing_epa_allowed",
    "rushing_epa_minus_opp_rushing_epa_allowed",
]


def load_raw() -> pd.DataFrame:
    use = ID_COLS + FEATURE_COLS
    hist = pd.read_csv(REPO_ROOT / "nfl_training_data.csv", usecols=lambda c: c in use)
    curr = pd.read_csv(REPO_ROOT / "nfl_current_data.csv", usecols=lambda c: c in use)
    new = pd.read_csv(CSV_2025, usecols=lambda c: c in use)
    df = pd.concat([hist, curr, new], ignore_index=True)
    df["is_cover"] = df["is_cover"].astype(int)
    return df.sort_values(["season", "week", "team"]).reset_index(drop=True)


def main():
    df = load_raw()
    print(f"Loaded {len(df):,} rows; seasons {df.season.min()}-{df.season.max()}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing cols: {missing}")
    df = attach_opponent_matchup(df)
    out = df[ID_COLS + FEATURE_COLS + MATCHUP_COLS].copy()

    # Guard: the 2018-2024 portion must be identical to the frozen parquet the
    # lockbox-2024 result was computed on — otherwise this is not the same
    # experiment and the 2025 run would not be a clean extension.
    frozen = pd.read_parquet(FROZEN)
    check = out[out.season <= 2024].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        check.sort_values(["season", "week", "team"]).reset_index(drop=True),
        frozen.sort_values(["season", "week", "team"]).reset_index(drop=True),
        check_dtype=False,
    )
    print("2018-2024 rows identical to frozen features.parquet ✓")

    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} -- shape {out.shape}")
    print("\nSeason coverage:")
    print(out.groupby("season").size().to_string())
    print("\n2025 NaN counts (top 8):")
    print(out[out.season == 2025][FEATURE_COLS + MATCHUP_COLS].isna().sum()
          .sort_values(ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
