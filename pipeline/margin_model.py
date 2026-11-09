"""Shared utilities for serving/scoring the margin model (v3).

The margin model is a ridge regression on the differences between the two
teams' features (home minus away, each divided by its training sd; a feature
missing for either team contributes nothing). It predicts the home team's
final margin directly, so the two sides of a game share one number:

    margin_home = sum_i coef_i * (f_home_i - f_away_i) / sd_i

No market input. The spread enters only when a margin is turned into a cover
probability:

    P(home covers) = t.cdf((margin_home + spread_home) / scale_t, df=nu_t)

`scale_t` is fitted on actual cover outcomes of out-of-sample predictions,
not on the margin residuals. It is several times the residual sd because the
market line already holds most of what the model knows; using the residual sd
gave probabilities that scored worse than a coin flip (research/margin_v3).

Two features are opponent-relative and are not columns in the wide weekly
CSV. The CSV has no opponent column either, so attach_opponent() joins it from
the nflverse schedule by (season, week, team), and pair_games() pairs the two
sides of each game from that column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = ROOT / "api" / "model_margin_v3.json"

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
    """Load the model bundle (plain JSON: coefficients, scales, feature order),
    or None if absent (stub/dev mode)."""
    path = Path(path)
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


def load_schedule(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per team-game from the nflverse schedule, regular season only.

    Columns: season, week, team, opponent, is_home, kickoff (ISO, Eastern),
    margin, points_for, points_against (NaN until played). Team codes are normalized to the stats data's
    current codes.
    """
    key = tuple(sorted({int(s) for s in seasons}))
    if key in _schedule_cache:
        return _schedule_cache[key]
    import nfl_data_py as nfl

    sched = nfl.import_schedules(list(key))
    sched = sched[sched["game_type"] == "REG"]
    for col in ("home_team", "away_team"):
        sched[col] = sched[col].replace(TEAM_ALIASES)
    kickoff = _kickoff_iso(sched["gameday"], sched["gametime"])
    home = pd.DataFrame({
        "season": sched["season"], "week": sched["week"],
        "team": sched["home_team"], "opponent": sched["away_team"], "is_home": 1,
        "kickoff": kickoff, "margin": sched["home_score"] - sched["away_score"],
        "points_for": sched["home_score"], "points_against": sched["away_score"],
    })
    away = pd.DataFrame({
        "season": sched["season"], "week": sched["week"],
        "team": sched["away_team"], "opponent": sched["home_team"], "is_home": 0,
        "kickoff": kickoff, "margin": sched["away_score"] - sched["home_score"],
        "points_for": sched["away_score"], "points_against": sched["home_score"],
    })
    out = pd.concat([home, away], ignore_index=True)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    _schedule_cache[key] = out
    return out


def _kickoff_iso(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """nflverse gives the date and an Eastern clock time; emit ISO with the offset."""
    from zoneinfo import ZoneInfo

    stamps = pd.to_datetime(gameday.astype(str) + " " + gametime.fillna("00:00").astype(str),
                            errors="coerce")
    return stamps.dt.tz_localize(ZoneInfo("America/New_York")).apply(
        lambda t: t.isoformat() if pd.notna(t) else None)


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


def standardized_diffs(f_home: np.ndarray, f_away: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(home - away) / sd per feature; missing on either side -> 0 (no information)."""
    return np.nan_to_num((np.asarray(f_home, dtype=float) - np.asarray(f_away, dtype=float)) / sd)


def game_margins(f_home: np.ndarray, f_away: np.ndarray, bundle: dict) -> np.ndarray:
    """Predicted home margin for each game. Rows are games, columns are the
    bundle's features in order."""
    z = standardized_diffs(f_home, f_away, np.asarray(bundle["feature_sd"], dtype=float))
    return z @ np.asarray(bundle["coef"], dtype=float)


def cover_probability(margin, spread, bundle: dict):
    """P(a team covers) given its predicted margin and its own spread."""
    from scipy import stats

    return stats.t.cdf(
        (np.asarray(margin, dtype=float) + np.asarray(spread, dtype=float)) / bundle["scale_t"],
        df=bundle["nu_t"],
    )


def predict_cover(df: pd.DataFrame, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """(margin, p_cover) per team-week row, from that team's side of its game.

    df needs season, week, team, opponent, is_home, spread and the bundle's
    features. The two rows of a game get opposite margins and probabilities
    that sum to 1. Rows whose opponent is not in the frame get NaN.
    """
    feats = bundle["features"]
    mu = np.full(len(df), np.nan)
    pos = {idx: k for k, idx in enumerate(df.index)}
    for (_, _), wk in df.groupby(["season", "week"], sort=False):
        pairs = pair_games(wk)
        if not pairs:
            continue
        X = wk[feats].to_numpy(dtype=float)
        homes = wk["is_home"].to_numpy()
        hi = np.array([i if homes[i] == 1 else j for i, j in pairs])
        ai = np.array([j if homes[i] == 1 else i for i, j in pairs])
        m = game_margins(X[hi], X[ai], bundle)
        mu[[pos[wk.index[k]] for k in hi]] = m
        mu[[pos[wk.index[k]] for k in ai]] = -m
    p = cover_probability(mu, df["spread"].to_numpy(dtype=float), bundle)
    return mu, p
