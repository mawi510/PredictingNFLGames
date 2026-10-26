"""Shared utilities for serving/scoring the margin model (v2).

The margin model predicts E[team_score - opp_score] from 44 features (see
api/feature_manifest_margin_v2.json) and converts to a cover probability at
the market spread via a Student-t CDF calibrated on out-of-sample residuals:

    P(cover) = t.cdf((mu_hat + spread) / scale_t, df=nu_t)

Two of its features are opponent-relative and don't exist as columns in the
wide weekly CSV. The CSV has no opponent column either, so attach_opponent()
joins it from the nflverse schedule by (season, week, team), and pair_games()
pairs the two sides of each game from that column for per-game pick
accounting. (Pairing used to match "opposite spread + opposite is_home"
within a week, which cross-paired teams from different games whenever two
games in a week shared a spread.)

Missing feature values stay NaN (XGBoost handles them natively); do NOT
fill with 0 (that was the old RandomForest convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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

# nflverse schedules keep historical codes; the weekly stats data uses current ones.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

_schedule_cache: dict[tuple[int, ...], pd.DataFrame] = {}


def load_bundle(path: str | Path = DEFAULT_BUNDLE_PATH):
    """Load the deploy bundle dict, or None if absent (stub/dev mode)."""
    path = Path(path)
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def load_schedule(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per team-game from the nflverse schedule, regular season only.

    Columns: season, week, team, opponent, is_home, margin (NaN until played).
    Team codes are normalized to the stats data's current codes.
    """
    key = tuple(sorted({int(s) for s in seasons}))
    if key in _schedule_cache:
        return _schedule_cache[key]
    import nfl_data_py as nfl

    sched = nfl.import_schedules(list(key))
    sched = sched[sched["game_type"] == "REG"]
    for col in ("home_team", "away_team"):
        sched[col] = sched[col].replace(TEAM_ALIASES)
    home = pd.DataFrame({
        "season": sched["season"], "week": sched["week"],
        "team": sched["home_team"], "opponent": sched["away_team"], "is_home": 1,
        "margin": sched["home_score"] - sched["away_score"],
    })
    away = pd.DataFrame({
        "season": sched["season"], "week": sched["week"],
        "team": sched["away_team"], "opponent": sched["home_team"], "is_home": 0,
        "margin": sched["away_score"] - sched["home_score"],
    })
    out = pd.concat([home, away], ignore_index=True)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    _schedule_cache[key] = out
    return out


def attach_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Add an `opponent` column by joining the schedule on (season, week, team)."""
    if "opponent" in df.columns:
        return df
    sched = load_schedule(df["season"].unique())[["season", "week", "team", "opponent"]]
    out = df.merge(sched, on=["season", "week", "team"], how="left")
    out.index = df.index
    return out


def pair_games(week_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Pair positional indices of the two sides of each game in one week's rows.

    Requires `team` and `opponent` columns (see attach_opponent). Rows whose
    opponent is missing from the frame are simply not paired.
    """
    teams = week_df["team"].to_numpy()
    opps = week_df["opponent"].to_numpy()
    pos = {t: i for i, t in enumerate(teams)}
    pairs = []
    used: set[int] = set()
    for i, opp in enumerate(opps):
        j = pos.get(opp)
        if i in used or j is None or j in used or opps[j] != teams[i]:
            continue
        pairs.append((i, j))
        used.update((i, j))
    return pairs


def attach_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two opponent-relative EPA features (and `opponent`) to a team-week frame."""
    df = attach_opponent(df.copy())
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
