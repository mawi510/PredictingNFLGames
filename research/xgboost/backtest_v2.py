"""
v2 backtest — does adding `prev_season_cover_rate` + `blended_cover_rate` help?

Compares 4 configurations (walk-forward, 2019-2024, threshold = -110 breakeven):

    A. XGB on 51 feat                                  -- prior best baseline
    B. XGB on 53 feat (+ prev_season + blended)        -- new cold-start features
    C. XGB on 51 feat + isotonic calibration           -- prior best (selective)
    D. XGB on 53 feat + isotonic calibration           -- combined: new features + cal

XGB hyperparams from the prior sweep winner: n_est=200, max_d=5, lr=0.03.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
FEATURES_V1 = ROOT / "features.parquet"
FEATURES_V2 = ROOT / "features_v2.parquet"

BREAKEVEN = 110 / 210
UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
THRESHOLD = BREAKEVEN
ID_COLS = ["season", "week", "team", "is_cover"]

XGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=7, n_jobs=4,
)


def score(p, y):
    bets = wins = 0
    units = 0.0
    for prob, yi in zip(p, y):
        if prob > THRESHOLD:
            bets += 1
            if yi == 1:
                wins += 1
                units += UNIT_PROFIT_WIN
            else:
                units -= UNIT_LOSS
    return dict(
        bets=bets, wins=wins, losses=bets - wins,
        win_pct=round(wins / bets, 4) if bets else 0.0,
        units=round(units, 2),
        roi=round(units / bets, 4) if bets else 0.0,
    )


def fit_predict(X_tr, y_tr, X_te, calibrate: bool) -> np.ndarray:
    model = xgb.XGBClassifier(**XGB_KW)
    if not calibrate:
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]
    # isotonic on last 20% of train
    n = len(X_tr)
    cut = int(n * 0.8)
    model.fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
    p_cal = model.predict_proba(X_tr.iloc[cut:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_tr.iloc[cut:])
    return iso.transform(model.predict_proba(X_te)[:, 1])


def run(df: pd.DataFrame, feature_cols: list[str], calibrate: bool, label: str):
    rows = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        tr = df[df["season"] < s]
        te = df[df["season"] == s]
        p = fit_predict(tr[feature_cols], tr["is_cover"], te[feature_cols], calibrate)
        r = score(p, te["is_cover"].values)
        rows.append({"label": label, "season": s, **r})
    per = pd.DataFrame(rows)
    bets = per["bets"].sum()
    wins = per["wins"].sum()
    units = per["units"].sum()
    agg = dict(
        label=label, season="AGG",
        bets=bets, wins=wins, losses=bets - wins,
        win_pct=round(wins / bets, 4) if bets else 0.0,
        units=round(units, 2),
        roi=round(units / bets, 4) if bets else 0.0,
    )
    return per, agg


def main():
    v1 = pd.read_parquet(FEATURES_V1)
    v2 = pd.read_parquet(FEATURES_V2)
    base_cols = [c for c in v1.columns if c not in ID_COLS]
    new_cols = base_cols + ["prev_season_cover_rate", "blended_cover_rate", "weight_on_prior_season"]

    print(f"v1: {len(base_cols)} features. v2: {len(new_cols)} features (+3 cold-start).")

    configs = [
        ("A: XGB 51 (baseline)",                v1, base_cols, False),
        ("B: XGB 54 (+ prior+blended)",         v2, new_cols, False),
        ("C: XGB 51 + isotonic",                v1, base_cols, True),
        ("D: XGB 54 + isotonic",                v2, new_cols, True),
    ]

    per_all, agg_all = [], []
    for label, df, cols, cal in configs:
        per, agg = run(df, cols, cal, label)
        per_all.append(per)
        agg_all.append(agg)

    per_df = pd.concat(per_all, ignore_index=True)
    per_df.to_csv(ROOT / "backtest_v2_per_season.csv", index=False)
    agg_df = pd.DataFrame(agg_all)
    agg_df.to_csv(ROOT / "backtest_v2_aggregate.csv", index=False)

    print("\n=== Aggregate (2019-2024) ===")
    print(agg_df[["label", "bets", "win_pct", "units", "roi"]].to_string(index=False))

    print("\n=== Per-season ROI ===")
    print(per_df.pivot_table(index="season", columns="label", values="roi").to_string())

    print("\n=== Per-season bets ===")
    print(per_df.pivot_table(index="season", columns="label", values="bets").to_string())

    print("\n=== Per-season win% ===")
    print(per_df.pivot_table(index="season", columns="label", values="win_pct").to_string())


if __name__ == "__main__":
    main()
