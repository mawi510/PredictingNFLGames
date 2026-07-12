"""
Stacked + calibrated backtest.

Four configurations on the same walk-forward (train on all prior seasons, predict on target):

    A. XGB on 51 curated features                   -- baseline rerun (sanity check)
    B. XGB on 51 + theta_diff (Elo stacked)         -- does the Elo signal help?
    C. XGB on 51 + theta_diff + isotonic calibration -- does calibration shift the bet set?
    D. LightGBM on 51 + theta_diff                  -- independent tree-model comparator

Best-from-prior-sweep XGB hyperparams (n_est=200, max_d=5, lr=0.03). LightGBM uses a
matched starting config. All bet at threshold = -110 breakeven (0.5238).

Reports per-season ROI, aggregate ROI, and a side-by-side scoreboard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT.parent / "xgboost" / "features.parquet"
ELO = ROOT / "elo_features.parquet"

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

LGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary", random_state=7, n_jobs=4, verbose=-1,
)


def load() -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_parquet(FEATURES)
    elo = pd.read_parquet(ELO)
    df = feats.merge(elo[["season", "week", "team", "theta_diff"]],
                     on=["season", "week", "team"], how="left")
    base_cols = [c for c in feats.columns if c not in ID_COLS]
    return df, base_cols


def score_picks(p: np.ndarray, y: np.ndarray) -> dict:
    bets = wins = losses = 0
    units = 0.0
    for prob, yi in zip(p, y):
        if prob > THRESHOLD:
            bets += 1
            if yi == 1:
                wins += 1
                units += UNIT_PROFIT_WIN
            else:
                losses += 1
                units -= UNIT_LOSS
    return {
        "bets": bets, "wins": wins, "losses": losses,
        "win_pct": round(wins / bets, 4) if bets else 0.0,
        "units": round(units, 2),
        "roi": round(units / bets, 4) if bets else 0.0,
    }


def fit_predict(model_name: str, X_tr, y_tr, X_te, calibrate: bool = False) -> np.ndarray:
    if model_name == "xgb":
        model = xgb.XGBClassifier(**XGB_KW)
    elif model_name == "lgb":
        model = lgb.LGBMClassifier(**LGB_KW)
    else:
        raise ValueError(model_name)

    if calibrate:
        # hold out last 20% of training rows for the calibrator
        n = len(X_tr)
        cut = int(n * 0.8)
        model.fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
        p_cal = model.predict_proba(X_tr.iloc[cut:])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_tr.iloc[cut:])
        p_test_raw = model.predict_proba(X_te)[:, 1]
        return iso.transform(p_test_raw)
    else:
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]


def run_config(df: pd.DataFrame, feature_cols: list[str], model: str, calibrate: bool, label: str):
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    rows = []
    for s in seasons:
        train = df[df["season"] < s]
        test = df[df["season"] == s]
        X_tr, y_tr = train[feature_cols], train["is_cover"]
        X_te, y_te = test[feature_cols], test["is_cover"]
        p = fit_predict(model, X_tr, y_tr, X_te, calibrate=calibrate)
        s_res = score_picks(p, y_te.values)
        rows.append({"label": label, "season": s, **s_res})
    out = pd.DataFrame(rows)
    agg_bets = out["bets"].sum()
    agg_wins = out["wins"].sum()
    agg_units = out["units"].sum()
    agg = {
        "label": label, "season": "AGG",
        "bets": agg_bets, "wins": agg_wins, "losses": agg_bets - agg_wins,
        "win_pct": round(agg_wins / agg_bets, 4) if agg_bets else 0.0,
        "units": round(agg_units, 2),
        "roi": round(agg_units / agg_bets, 4) if agg_bets else 0.0,
    }
    return out, agg


def main():
    df, base_cols = load()
    print(f"Loaded {len(df):,} rows, {len(base_cols)} base features + theta_diff")
    with_elo = base_cols + ["theta_diff"]

    configs = [
        ("A: XGB baseline (51 feat)",        "xgb", False, base_cols),
        ("B: XGB + theta_diff",              "xgb", False, with_elo),
        ("C: XGB + theta_diff + isotonic",   "xgb", True,  with_elo),
        ("D: LightGBM + theta_diff",         "lgb", False, with_elo),
    ]

    per_season_all = []
    aggregates = []
    for label, model, calibrate, cols in configs:
        per_s, agg = run_config(df, cols, model, calibrate, label)
        per_season_all.append(per_s)
        aggregates.append(agg)

    per = pd.concat(per_season_all, ignore_index=True)
    per.to_csv(ROOT / "stacked_per_season.csv", index=False)
    agg_df = pd.DataFrame(aggregates)
    agg_df.to_csv(ROOT / "stacked_aggregate.csv", index=False)

    print("\n=== Aggregate across 2019-2024 ===")
    print(agg_df[["label", "bets", "win_pct", "units", "roi"]].to_string(index=False))

    print("\n=== Per-season ROI side-by-side ===")
    pivot = per.pivot_table(index="season", columns="label", values="roi")
    print(pivot.to_string())

    print("\n=== Per-season win% ===")
    pivot_w = per.pivot_table(index="season", columns="label", values="win_pct")
    print(pivot_w.to_string())

    print("\n--- prior baselines for context ---")
    print("Prod RF 2024:                     -16.9%")
    print("Bayesian Elo v2 (cover only):      -9.8% (1397 bets)")
    print("XGB 51-feat (prior result):        -0.87% (1194 bets)")
    print("Breakeven -110:                     0.00%")


if __name__ == "__main__":
    main()
