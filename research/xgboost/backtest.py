"""
Walk-forward XGBoost baseline on the curated 51-feature set.

For each target season s in 2019..2024:
    train_on = all team-weeks with season < s
    predict  = all team-weeks with season == s
    bet rule = if P(cover) > threshold, take that team-week

Compares directly against:
    - Prod RF on 2024:           -16.9% ROI
    - Bayesian Elo v2 aggregate: -9.8% ROI across 2019-2024
    - Breakeven -110:             0.00% ROI

Same accounting (1u stake at -110 -> +0.909u win / -1u loss) as the Bayesian spikes.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATURES_PARQUET = ROOT / "features.parquet"

BREAKEVEN = 110 / 210
UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0

ID_COLS = ["season", "week", "team", "is_cover"]


def backtest_one(
    df: pd.DataFrame,
    season: int,
    feature_cols: list[str],
    n_estimators: int,
    max_depth: int,
    lr: float,
    threshold: float,
    seed: int = 7,
) -> dict:
    train = df[df["season"] < season]
    test = df[df["season"] == season]
    X_tr, y_tr = train[feature_cols], train["is_cover"]
    X_te, y_te = test[feature_cols], test["is_cover"]

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )
    clf.fit(X_tr, y_tr, verbose=False)
    p = clf.predict_proba(X_te)[:, 1]

    bets = wins = losses = 0
    units = 0.0
    for prob, y in zip(p, y_te.values):
        if prob > threshold:
            bets += 1
            if y == 1:
                wins += 1
                units += UNIT_PROFIT_WIN
            else:
                losses += 1
                units -= UNIT_LOSS

    roi = units / bets if bets else 0.0
    return {
        "season": season,
        "n_est": n_estimators,
        "max_d": max_depth,
        "lr": lr,
        "threshold": round(threshold, 4),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_pct": round(wins / bets, 4) if bets else 0.0,
        "units": round(units, 2),
        "roi": round(roi, 4),
    }


def main():
    df = pd.read_parquet(FEATURES_PARQUET)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    print(f"Loaded {len(df):,} rows x {len(feature_cols)} features")

    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    n_est_grid = [200, 400, 800]
    depth_grid = [3, 5, 7]
    lr_grid = [0.03, 0.08]
    thr_grid = [0.50, BREAKEVEN]

    rows = []
    for s in seasons:
        for n_est, depth, lr, thr in product(n_est_grid, depth_grid, lr_grid, thr_grid):
            rows.append(backtest_one(df, s, feature_cols, n_est, depth, lr, thr))

    full = pd.DataFrame(rows)
    full.to_csv(ROOT / "backtest_per_season.csv", index=False)

    # Aggregate across seasons per config
    agg = (
        full.groupby(["n_est", "max_d", "lr", "threshold"], as_index=False)
        .agg(bets=("bets", "sum"), wins=("wins", "sum"),
             losses=("losses", "sum"), units=("units", "sum"))
    )
    agg["win_pct"] = (agg["wins"] / agg["bets"]).round(4)
    agg["roi"] = (agg["units"] / agg["bets"]).round(4)
    agg = agg.sort_values("roi", ascending=False)
    agg.to_csv(ROOT / "backtest_aggregate.csv", index=False)

    print("\n=== Aggregate across 2019-2024 (pooled), top 10 ===")
    print(agg.head(10).to_string(index=False))
    print("\n=== Aggregate bottom 5 ===")
    print(agg.tail(5).to_string(index=False))

    # Best config per-season breakdown
    best = agg.iloc[0]
    print(f"\n=== Best aggregate config per season ===")
    print(f"(n_est={int(best['n_est'])}, max_d={int(best['max_d'])}, "
          f"lr={best['lr']}, threshold={best['threshold']:.4f})\n")
    mask = (
        (full["n_est"] == best["n_est"])
        & (full["max_d"] == best["max_d"])
        & (full["lr"] == best["lr"])
        & (full["threshold"] == best["threshold"])
    )
    print(full[mask].sort_values("season").to_string(index=False))

    print("\n--- baselines ---")
    print("Prod RF 2024:                   -16.9%")
    print("Bayesian Elo v2 aggregate:       -9.8% (1397 bets)")
    print("Breakeven -110:                   0.00%")


if __name__ == "__main__":
    main()
