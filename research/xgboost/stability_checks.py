"""
Stability + sensitivity checks for Config D (XGB on 54 features + isotonic calibration).

Three checks:

1. SEED STABILITY -- rerun Config D across seeds {1,3,7,11,42}; report ROI mean +/- std,
   bet-count mean +/- std per season. If ROI swings wildly, the +2.80% headline is
   noise from a single seed.

2. TAU SWEEP -- rebuild features_v2 internally with tau in {2, 3, 4, 5} and rerun D.
   tau controls how fast the prior-season weight decays (week-1 weight = exp(-1/tau)).
   tau=2 -> aggressive decay (prior gone by wk 4). tau=5 -> slow (prior still 22% at wk 8).

3. PLATT (SIGMOID) CALIBRATION -- compare isotonic vs Platt as a comparator. Platt fits
   a 2-param logistic, so it's cheaper and may behave better on small calibration sets.

Threshold fixed at -110 breakeven. Walk-forward 2019-2024.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent

BREAKEVEN = 110 / 210
UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
THRESHOLD = BREAKEVEN
ID_COLS = ["season", "week", "team", "is_cover"]
LEAGUE_FALLBACK = 0.5


def build_features(tau: float) -> tuple[pd.DataFrame, list[str]]:
    """Build feature DF in-memory with a configurable tau on the blended_cover_rate."""
    v1 = pd.read_parquet(ROOT / "features.parquet")
    labels = pd.concat([
        pd.read_csv(REPO_ROOT / "nfl_training_data.csv", usecols=["season", "team", "is_cover"]),
        pd.read_csv(REPO_ROOT / "nfl_current_data.csv", usecols=["season", "team", "is_cover"]),
    ]).dropna(subset=["is_cover"])
    labels["is_cover"] = labels["is_cover"].astype(int)
    ts = labels.groupby(["season", "team"])["is_cover"].mean().reset_index()
    ts = ts.rename(columns={"is_cover": "cover_rate"})
    ts["season"] = ts["season"] + 1
    ts = ts.rename(columns={"cover_rate": "prev_season_cover_rate"})
    df = v1.merge(ts, on=["season", "team"], how="left")
    df["prev_season_cover_rate"] = df["prev_season_cover_rate"].fillna(LEAGUE_FALLBACK)
    df["weight_on_prior_season"] = np.exp(-df["week"] / tau)
    cur = df["is_cover_record_shifted"].fillna(df["prev_season_cover_rate"])
    df["blended_cover_rate"] = (
        df["weight_on_prior_season"] * df["prev_season_cover_rate"]
        + (1 - df["weight_on_prior_season"]) * cur
    )
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    return df, feat_cols


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
        bets=bets, wins=wins,
        win_pct=round(wins / bets, 4) if bets else 0.0,
        units=round(units, 2),
        roi=round(units / bets, 4) if bets else 0.0,
    )


def fit_predict(X_tr, y_tr, X_te, calibrate: str, seed: int) -> np.ndarray:
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed, n_jobs=4,
    )
    if calibrate == "none":
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]
    n = len(X_tr); cut = int(n * 0.8)
    model.fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
    p_cal = model.predict_proba(X_tr.iloc[cut:])[:, 1]
    y_cal = y_tr.iloc[cut:]
    p_te_raw = model.predict_proba(X_te)[:, 1]
    if calibrate == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
        return iso.transform(p_te_raw)
    elif calibrate == "platt":
        lr = LogisticRegression(C=1.0).fit(p_cal.reshape(-1, 1), y_cal)
        return lr.predict_proba(p_te_raw.reshape(-1, 1))[:, 1]
    raise ValueError(calibrate)


def run_one(df: pd.DataFrame, feat: list[str], calibrate: str, seed: int) -> tuple[pd.DataFrame, dict]:
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    rows = []
    for s in seasons:
        tr, te = df[df["season"] < s], df[df["season"] == s]
        p = fit_predict(tr[feat], tr["is_cover"], te[feat], calibrate, seed)
        rows.append({"season": s, **score(p, te["is_cover"].values)})
    per = pd.DataFrame(rows)
    bets = per["bets"].sum(); wins = per["wins"].sum(); units = per["units"].sum()
    agg = dict(bets=bets, wins=wins,
               win_pct=round(wins / bets, 4) if bets else 0.0,
               units=round(units, 2),
               roi=round(units / bets, 4) if bets else 0.0)
    return per, agg


def check_seed_stability():
    print("\n" + "=" * 70)
    print("CHECK 1: SEED STABILITY (Config D: 54 feat, tau=3, isotonic)")
    print("=" * 70)
    df, feat = build_features(tau=3.0)
    seeds = [1, 3, 7, 11, 42]
    aggs = []
    bets_per_season = []
    for seed in seeds:
        per, agg = run_one(df, feat, "isotonic", seed)
        aggs.append({"seed": seed, **agg})
        bp = {f"s{int(s)}_bets": int(b) for s, b in zip(per["season"], per["bets"])}
        bets_per_season.append({"seed": seed, **bp})
    a = pd.DataFrame(aggs)
    b = pd.DataFrame(bets_per_season)
    print("\nPer-seed aggregate:")
    print(a.to_string(index=False))
    print(f"\nROI mean: {a['roi'].mean()*100:.2f}% +/- {a['roi'].std()*100:.2f}%")
    print(f"Bets mean: {a['bets'].mean():.0f} +/- {a['bets'].std():.0f}")
    print("\nBets per season per seed:")
    print(b.to_string(index=False))


def check_tau_sweep():
    print("\n" + "=" * 70)
    print("CHECK 2: TAU SWEEP (Config D, isotonic, seed=7)")
    print("=" * 70)
    rows = []
    for tau in [2, 3, 4, 5]:
        df, feat = build_features(tau=float(tau))
        per, agg = run_one(df, feat, "isotonic", seed=7)
        rows.append({"tau": tau, **agg})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nReminder of weight schedule:")
    for tau in [2, 3, 4, 5]:
        print(f"  tau={tau}: wk1={np.exp(-1/tau):.2f}, wk4={np.exp(-4/tau):.2f}, "
              f"wk8={np.exp(-8/tau):.2f}")


def check_platt_vs_isotonic():
    print("\n" + "=" * 70)
    print("CHECK 3: ISOTONIC vs PLATT (Config D, tau=3, seed=7)")
    print("=" * 70)
    df, feat = build_features(tau=3.0)
    rows = []
    per_sess = {}
    for cal in ["isotonic", "platt"]:
        per, agg = run_one(df, feat, cal, seed=7)
        rows.append({"calibrator": cal, **agg})
        per_sess[cal] = per
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPer-season bet counts:")
    for cal, per in per_sess.items():
        print(f"  {cal}: " + ", ".join(f"{int(s)}={int(b)}"
              for s, b in zip(per["season"], per["bets"])))
    print("\nPer-season ROI:")
    for cal, per in per_sess.items():
        print(f"  {cal}: " + ", ".join(f"{int(s)}={r*100:+.2f}%"
              for s, r in zip(per["season"], per["roi"])))


def main():
    check_seed_stability()
    check_tau_sweep()
    check_platt_vs_isotonic()
    print("\nDone.")


if __name__ == "__main__":
    main()
