"""Recompute the two opponent-relative features in the frozen parquets.

They were built by pairing rows on opposite spread + opposite is_home, which
cross-pairs teams from different games whenever two games in a week share a
spread. Every other column is untouched; the rolling EPA inputs the matchup
features derive from are already in the parquets, so no raw CSV is needed.

Usage:  python research/xgboost/rebuild_matchup_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pipeline.margin_model import MATCHUP_FEATURES, attach_matchup_features  # noqa: E402

MATCHUP_COLS = list(MATCHUP_FEATURES)


def rebuild(path: Path) -> None:
    old = pd.read_parquet(path)
    cols = list(old.columns)
    new = attach_matchup_features(old.drop(columns=MATCHUP_COLS)).drop(columns=["opponent"])[cols]
    other = [c for c in cols if c not in MATCHUP_COLS]
    pd.testing.assert_frame_equal(old[other], new[other])
    changed = ~np.isclose(old[MATCHUP_COLS[0]], new[MATCHUP_COLS[0]], equal_nan=True)
    print(f"{path.name}: {len(new):,} rows, matchup features changed on {changed.sum():,} "
          f"({100 * changed.mean():.1f}%); unpaired rows now {new[MATCHUP_COLS[0]].isna().sum()} "
          f"(was {old[MATCHUP_COLS[0]].isna().sum()})")
    new.to_parquet(path, index=False)


if __name__ == "__main__":
    for name in ("features.parquet", "features_2025.parquet"):
        rebuild(HERE / name)
