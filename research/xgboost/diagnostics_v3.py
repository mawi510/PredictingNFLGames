"""
Diagnostics responding to user's pushback on backtest_v3.py:

  Q1: Why such a low bet count? In production we should pick a side on every game
      unless there's a compelling reason not to.
  Q2: What does the model "know" in weeks 1-2 when no games have been played?
      Should we restrict to week 4+?

This script runs XGB-A (51 features, no calibration) under the corrected per-game
accounting (per-game pick, pushes excluded) and reports:

  A. "BET-EVERY-GAME" rule:   pick argmax(P_A, P_B) on EVERY game (no threshold).
                              This is the universe-level performance of the model.
  B. Threshold tiers:         compare p>0.5238 vs p>0.55 vs p>0.60 vs p>0.65.
  C. Per-week breakdown:      win-rate + ROI per (target_season, week) bucket
                              under the BET-EVERY-GAME rule. Shows whether early
                              weeks (1-3) are coin flips.
  D. Per-week count of NaN features per row at prediction time, so we can see
      what the model actually has to work with in weeks 1-2.

XGB hyperparams locked to the prior best: n_est=200, max_d=5, lr=0.03, seed=7.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features.parquet"

BREAKEVEN = 110 / 210
UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
ID_COLS = ["season", "week", "team", "is_cover"]

XGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=7, n_jobs=4,
)


def fit_predict_all(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Walk-forward season-by-season, return one row per team-week with predicted p."""
    parts = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        tr = df[df["season"] < s]
        te = df[df["season"] == s].copy()
        model = xgb.XGBClassifier(**XGB_KW)
        model.fit(tr[feat], tr["is_cover"])
        te["_p"] = model.predict_proba(te[feat])[:, 1]
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def pair_picks(predicted: pd.DataFrame) -> pd.DataFrame:
    """Pair team-weeks into games, emit one row per game with the picked side."""
    rows = []
    for (s, w), wk in predicted.groupby(["season", "week"], sort=False):
        wk = wk.reset_index(drop=True)
        used = set()
        for i in range(len(wk)):
            if i in used:
                continue
            ri = wk.iloc[i]
            cand = [
                j for j in range(len(wk))
                if j != i and j not in used
                and np.isclose(wk.iloc[j]["spread"], -ri["spread"])
                and wk.iloc[j]["is_home"] == 1 - ri["is_home"]
            ]
            if not cand:
                continue
            j = cand[0]
            used.add(i); used.add(j)
            rj = wk.iloc[j]
            pa, pb = ri["_p"], rj["_p"]
            ya, yb = int(ri["is_cover"]), int(rj["is_cover"])
            is_push = (ya == 0 and yb == 0)
            if pa >= pb:
                pick_team, pick_p, pick_y, pick_spread = ri["team"], pa, ya, ri["spread"]
                opp_team = rj["team"]
            else:
                pick_team, pick_p, pick_y, pick_spread = rj["team"], pb, yb, rj["spread"]
                opp_team = ri["team"]
            rows.append({
                "season": int(s), "week": int(w),
                "picked": pick_team, "opponent": opp_team,
                "pick_spread": pick_spread,
                "p": float(pick_p),
                "covered": pick_y,
                "is_push": is_push,
                "is_fav_pick": pick_spread < 0,
            })
    return pd.DataFrame(rows)


def score(picks: pd.DataFrame, threshold: float = 0.0) -> dict:
    """Score picks; threshold=0 means bet every (non-push) game."""
    p = picks[~picks["is_push"]]
    p = p[p["p"] > threshold] if threshold > 0 else p
    bets = len(p)
    if bets == 0:
        return dict(bets=0, wins=0, win_pct=0.0, units=0.0, roi=0.0, fav_pct=0.0)
    wins = int(p["covered"].sum())
    units = wins * UNIT_PROFIT_WIN - (bets - wins) * UNIT_LOSS
    return dict(
        bets=bets, wins=wins,
        win_pct=round(wins / bets, 4),
        units=round(units, 2),
        roi=round(units / bets, 4),
        fav_pct=round(p["is_fav_pick"].mean(), 4),
    )


def main():
    df = pd.read_parquet(FEATURES)
    feat = [c for c in df.columns if c not in ID_COLS]
    print(f"Loaded {len(df):,} rows x {len(feat)} features (XGB-A: 51 features, no cal)")

    predicted = fit_predict_all(df, feat)
    picks = pair_picks(predicted)
    picks.to_csv(ROOT / "picks_log_XGB_A.csv", index=False)
    print(f"Generated picks for {len(picks)} games "
          f"({picks['is_push'].sum()} pushes excluded). Log -> picks_log_XGB_A.csv")

    print("\n" + "=" * 72)
    print("A. BET-EVERY-GAME RULE (no threshold; pick argmax on every non-push game)")
    print("=" * 72)
    universal = score(picks, threshold=0.0)
    print(f"Bets: {universal['bets']}, win%: {universal['win_pct']*100:.2f}%, "
          f"units: {universal['units']}, ROI: {universal['roi']*100:+.2f}%, "
          f"fav_pct: {universal['fav_pct']*100:.1f}%")

    print("\nPer-season under bet-every-game:")
    per_s = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        sub = picks[picks["season"] == s]
        per_s.append({"season": s, **score(sub, threshold=0.0)})
    print(pd.DataFrame(per_s).to_string(index=False))

    print("\n" + "=" * 72)
    print("B. THRESHOLD TIERS (still per-game pick, but only bet if p > threshold)")
    print("=" * 72)
    tier_rows = []
    for thr in [0.00, 0.5238, 0.55, 0.58, 0.60, 0.65]:
        r = score(picks, threshold=thr)
        tier_rows.append({"threshold": thr, **r})
    print(pd.DataFrame(tier_rows).to_string(index=False))

    print("\n" + "=" * 72)
    print("C. PER-WEEK BREAKDOWN (bet-every-game rule, all seasons pooled)")
    print("=" * 72)
    print("Tests whether early weeks (cold start) are coin flips.\n")
    p = picks[~picks["is_push"]]
    by_week = p.groupby("week").agg(
        bets=("covered", "count"),
        wins=("covered", "sum"),
    ).reset_index()
    by_week["win_pct"] = (by_week["wins"] / by_week["bets"]).round(4)
    by_week["units"] = by_week["wins"] * UNIT_PROFIT_WIN - (by_week["bets"] - by_week["wins"]) * UNIT_LOSS
    by_week["roi"] = (by_week["units"] / by_week["bets"]).round(4)
    print(by_week.to_string(index=False))

    print("\n" + "=" * 72)
    print("D. WEEK BUCKETS (cold-start vs mid-season vs late)")
    print("=" * 72)
    def bucket(w):
        if w <= 3: return "wks 1-3 (cold start)"
        if w <= 6: return "wks 4-6 (early)"
        if w <= 12: return "wks 7-12 (mid)"
        return "wks 13+ (late)"
    p = p.copy()
    p["bucket"] = p["week"].apply(bucket)
    by_b = p.groupby("bucket").agg(bets=("covered", "count"), wins=("covered", "sum")).reset_index()
    by_b["win_pct"] = (by_b["wins"] / by_b["bets"]).round(4)
    by_b["units"] = by_b["wins"] * UNIT_PROFIT_WIN - (by_b["bets"] - by_b["wins"]) * UNIT_LOSS
    by_b["roi"] = (by_b["units"] / by_b["bets"]).round(4)
    bucket_order = ["wks 1-3 (cold start)", "wks 4-6 (early)", "wks 7-12 (mid)", "wks 13+ (late)"]
    by_b["_o"] = by_b["bucket"].apply(lambda b: bucket_order.index(b))
    by_b = by_b.sort_values("_o").drop(columns="_o")
    print(by_b.to_string(index=False))

    print("\n" + "=" * 72)
    print("E. NaN-feature count by week (what the model actually 'knows')")
    print("=" * 72)
    nan_by_week = predicted.groupby("week")[feat].apply(
        lambda x: x.isna().sum(axis=1).mean()
    ).reset_index(name="avg_nan_features_per_row")
    nan_by_week["avg_nan_features_per_row"] = nan_by_week["avg_nan_features_per_row"].round(1)
    print(nan_by_week.head(8).to_string(index=False))
    print(f"... (out of {len(feat)} features)")


if __name__ == "__main__":
    main()
