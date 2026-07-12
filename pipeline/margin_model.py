"""Shared utilities for serving/scoring the margin model (v2).

The margin model predicts E[team_score - opp_score] from 44 features (see
api/feature_manifest_margin_v2.json) and converts to a cover probability at
the market spread via a Student-t CDF calibrated on out-of-sample residuals:

    P(cover) = t.cdf((mu_hat + spread) / scale_t, df=nu_t)

Two of its features are opponent-relative and don't exist as columns in the
wide weekly CSV; attach_matchup_features() builds them by pairing each
team-week with its opponent (same season/week, opposite spread, opposite
is_home — the opponent column isn't in the shifted CSV). pair_games() exposes
the same pairing for per-game pick accounting in the backtest.

Missing feature values stay NaN (XGBoost handles them natively); do NOT
fill with 0 — that was the old RandomForest convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = ROOT / "api" / "model_margin_v2.pkl"

MATCHUP_FEATURES = {
    "passing_epa_minus_opp_passing_epa_allowed": (
        "total_passing_epa_rolling_shifted",
        "total_passing_epa_allowed_rolling_shifted",
    ),
    "rushing_epa_minus_opp_rushing_epa_allowed": (
        "total_rushing_epa_rolling_shifted",
        "total_rushing_epa_allowed_rolling_shifted",
    ),
}


def load_bundle(path: str | Path = DEFAULT_BUNDLE_PATH):
    """Load the deploy bundle dict, or None if absent (stub/dev mode)."""
    path = Path(path)
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def pair_games(week_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Pair positional indices of opposing team-weeks within one week's rows.

    Same greedy matching the research backtests use: opponent = the unused row
    with the opposite spread and opposite is_home. Rows without a match (data
    gaps) are simply not paired.
    """
    pairs = []
    used: set[int] = set()
    spreads = week_df["spread"].to_numpy()
    homes = week_df["is_home"].to_numpy()
    n = len(week_df)
    for i in range(n):
        if i in used:
            continue
        for j in range(n):
            if j == i or j in used:
                continue
            if np.isclose(spreads[j], -spreads[i]) and homes[j] == 1 - homes[i]:
                pairs.append((i, j))
                used.update((i, j))
                break
    return pairs


def attach_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two opponent-relative EPA features to a team-week frame."""
    df = df.copy()
    for out_col in MATCHUP_FEATURES:
        df[out_col] = np.nan
    for (_, _), wk in df.groupby(["season", "week"], sort=False):
        for i, j in pair_games(wk):
            ii, jj = wk.index[i], wk.index[j]
            for out_col, (own, opp_allowed) in MATCHUP_FEATURES.items():
                df.loc[ii, out_col] = wk.iloc[i][own] - wk.iloc[j][opp_allowed]
                df.loc[jj, out_col] = wk.iloc[j][own] - wk.iloc[i][opp_allowed]
    return df


def predict_cover(df: pd.DataFrame, bundle) -> tuple[np.ndarray, np.ndarray]:
    """(mu_hat, p_cover) per row. df must contain the bundle's features + spread."""
    from scipy import stats

    X = df.reindex(columns=bundle["feature_names"]).astype(float)
    mu = bundle["regressor"].predict(X)
    p = stats.t.cdf(
        (mu + df["spread"].to_numpy(dtype=float)) / bundle["scale_t"],
        df=bundle["nu_t"],
    )
    return mu, p
