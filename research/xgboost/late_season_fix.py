"""
Late-season weakness fix -- test Z1 (weekly retraining) and Z2 (Bayesian adjustment).

Baseline: XGB-A (51 features, no calibration, trained once on prior seasons).
Late season (wks 13+) under bet-every-game rule: 54.35%, ROI +3.76%.

Z1: For each (target_season, target_week), train on
    (prior_seasons_all_weeks) + (current_season_weeks < target_week).
    Same XGB hyperparams. Tests whether including in-season weeks in TRAINING
    fixes the late-season weakness.

Z2: Keep the once-trained XGB. Maintain a per-team Beta-Binomial posterior on
    cover rate updated each week. Blend:
        p_final = (1 - w(week)) * p_xgb + w(week) * (alpha_T / (alpha_T + beta_T))
    Sweep three w(week) schedules:
        linear:    w = week / 17                         (smooth ramp)
        late:      w = max(0, (week - 8) / 9)            (kicks in week 9)
        sigmoid:   w = 1 / (1 + exp(-(week - 10) / 2))   (S-curve around wk 10)

Beta-Binomial init: alpha = ps * prev_season_cover_rate, beta = ps * (1 - ...),
ps=32, no forgetting (best per v1 validation). League fallback for new teams.

Reports per-bucket ROI to see if the late-season hole is closed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features.parquet"

UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
ID_COLS = ["season", "week", "team", "is_cover"]
PRIOR_STRENGTH = 32

XGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=7, n_jobs=4,
)


def fit_predict_xgb_a(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Baseline: XGB-A trained once per season on prior seasons only."""
    parts = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        tr = df[df["season"] < s]
        te = df[df["season"] == s].copy()
        m = xgb.XGBClassifier(**XGB_KW)
        m.fit(tr[feat], tr["is_cover"])
        te["_p"] = m.predict_proba(te[feat])[:, 1]
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def fit_predict_weekly_retrain(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Z1: retrain each week with prior seasons + current season's prior weeks."""
    parts = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        season_df = df[df["season"] == s]
        weeks = sorted(season_df["week"].unique())
        for w in weeks:
            tr = df[(df["season"] < s) | ((df["season"] == s) & (df["week"] < w))]
            te = season_df[season_df["week"] == w].copy()
            m = xgb.XGBClassifier(**XGB_KW)
            m.fit(tr[feat], tr["is_cover"])
            te["_p"] = m.predict_proba(te[feat])[:, 1]
            parts.append(te)
    return pd.concat(parts, ignore_index=True)


def bayes_per_team_posterior(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """For each team-week row in `season`, compute the PRE-week Beta posterior mean.

    Init from previous season cover rate (league fallback). Updates after each week.
    """
    prev = df[df["season"] == season - 1]
    rates = prev.groupby("team")["is_cover"].mean().to_dict() if not prev.empty else {}
    league = float(prev["is_cover"].mean()) if not prev.empty else 0.5

    def init(t):
        p0 = rates.get(t, league)
        return PRIOR_STRENGTH * p0, PRIOR_STRENGTH * (1 - p0)

    state: dict[str, tuple[float, float]] = {}
    season_df = df[df["season"] == season].sort_values(["week", "team"]).reset_index(drop=True)
    p_bayes = np.full(len(season_df), np.nan)
    for w in sorted(season_df["week"].unique()):
        idxs = season_df.index[season_df["week"] == w].tolist()
        # 1) pre-week posterior mean per team
        for i in idxs:
            t = season_df.at[i, "team"]
            if t not in state:
                state[t] = init(t)
            a, b = state[t]
            p_bayes[i] = a / (a + b)
        # 2) post-week update
        for i in idxs:
            t = season_df.at[i, "team"]
            y = int(season_df.at[i, "is_cover"])
            a, b = state[t]
            state[t] = (a + y, b + (1 - y))
    season_df["_p_bayes"] = p_bayes
    return season_df


def blend_xgb_and_bayes(xgb_preds: pd.DataFrame, schedule: str) -> pd.DataFrame:
    """Z2: blend XGB output with per-team Beta posterior using a week-dependent weight."""
    out_parts = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        bay = bayes_per_team_posterior(xgb_preds, s)[["season", "week", "team", "_p_bayes"]]
        xgs = xgb_preds[xgb_preds["season"] == s].merge(bay, on=["season", "week", "team"], how="left")
        w = weight_schedule(xgs["week"].values, schedule)
        xgs["_p"] = (1 - w) * xgs["_p"] + w * xgs["_p_bayes"].fillna(xgs["_p"])
        out_parts.append(xgs)
    return pd.concat(out_parts, ignore_index=True)


def weight_schedule(week: np.ndarray, name: str) -> np.ndarray:
    if name == "linear":
        return np.clip(week / 17.0, 0, 1)
    if name == "late":
        return np.clip((week - 8.0) / 9.0, 0, 1)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-(week - 10.0) / 2.0))
    raise ValueError(name)


def pair_and_pick(predicted: pd.DataFrame) -> pd.DataFrame:
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
            if ya == 0 and yb == 0:
                continue
            if pa >= pb:
                pick_y = ya; pick_p = pa; pick_spread = ri["spread"]
            else:
                pick_y = yb; pick_p = pb; pick_spread = rj["spread"]
            rows.append({"season": int(s), "week": int(w),
                         "p": float(pick_p), "covered": int(pick_y),
                         "fav": int(pick_spread < 0)})
    return pd.DataFrame(rows)


def summarize(picks: pd.DataFrame, label: str) -> dict:
    n = len(picks); w = int(picks["covered"].sum())
    u = w * UNIT_PROFIT_WIN - (n - w) * UNIT_LOSS
    return dict(model=label, bets=n, win_pct=round(w / n, 4) if n else 0.0,
                units=round(u, 2), roi=round(u / n, 4) if n else 0.0)


def bucket_summary(picks: pd.DataFrame, label: str) -> pd.DataFrame:
    def b(wk):
        if wk <= 3: return "1) wks 1-3"
        if wk <= 6: return "2) wks 4-6"
        if wk <= 12: return "3) wks 7-12"
        return "4) wks 13+"
    p = picks.assign(bucket=picks["week"].apply(b))
    g = p.groupby("bucket").agg(bets=("covered", "count"), wins=("covered", "sum")).reset_index()
    g["win_pct"] = (g["wins"] / g["bets"]).round(4)
    g["units"] = g["wins"] * UNIT_PROFIT_WIN - (g["bets"] - g["wins"]) * UNIT_LOSS
    g["roi"] = (g["units"] / g["bets"]).round(4)
    g["model"] = label
    return g[["model", "bucket", "bets", "win_pct", "units", "roi"]]


def main():
    df = pd.read_parquet(FEATURES)
    feat = [c for c in df.columns if c not in ID_COLS]
    print(f"Loaded {len(df):,} rows x {len(feat)} features. Running XGB-A baseline...")

    # Baseline
    xgb_a = fit_predict_xgb_a(df, feat)
    baseline_picks = pair_and_pick(xgb_a)

    # Z1: weekly retrain
    print("Running Z1: weekly retraining (102 fits, ~3-5 min)...")
    z1 = fit_predict_weekly_retrain(df, feat)
    z1_picks = pair_and_pick(z1)

    # Z2: three blend schedules
    print("Running Z2: Bayesian post-hoc adjustment, 3 schedules...")
    z2_runs = {}
    for sch in ["linear", "late", "sigmoid"]:
        blended = blend_xgb_and_bayes(xgb_a, sch)
        z2_runs[sch] = pair_and_pick(blended)

    # Aggregate scoreboard
    print("\n=== AGGREGATE (bet every non-push game) ===")
    aggs = [
        summarize(baseline_picks, "XGB-A baseline"),
        summarize(z1_picks,        "Z1 weekly retrain"),
        summarize(z2_runs["linear"],  "Z2 blend, linear w"),
        summarize(z2_runs["late"],    "Z2 blend, late w"),
        summarize(z2_runs["sigmoid"], "Z2 blend, sigmoid w"),
    ]
    print(pd.DataFrame(aggs).to_string(index=False))

    # Week-bucket comparison
    print("\n=== PER-WEEK-BUCKET ROI (the diagnostic) ===")
    buckets = pd.concat([
        bucket_summary(baseline_picks, "XGB-A baseline"),
        bucket_summary(z1_picks,        "Z1 weekly retrain"),
        bucket_summary(z2_runs["linear"],  "Z2 linear"),
        bucket_summary(z2_runs["late"],    "Z2 late"),
        bucket_summary(z2_runs["sigmoid"], "Z2 sigmoid"),
    ], ignore_index=True)
    pivot_roi = buckets.pivot(index="bucket", columns="model", values="roi")
    pivot_w   = buckets.pivot(index="bucket", columns="model", values="win_pct")
    print("\nROI per bucket:")
    print(pivot_roi.to_string())
    print("\nWin% per bucket:")
    print(pivot_w.to_string())

    # Per season for late bucket only
    print("\n=== LATE BUCKET (wks 13+) PER SEASON ===")
    late_compare = []
    for name, picks in [("XGB-A", baseline_picks), ("Z1", z1_picks),
                        ("Z2 lin", z2_runs["linear"]),
                        ("Z2 late", z2_runs["late"]),
                        ("Z2 sig", z2_runs["sigmoid"])]:
        late = picks[picks["week"] >= 13]
        for s in [2019, 2020, 2021, 2022, 2023, 2024]:
            sub = late[late["season"] == s]
            n = len(sub); w = int(sub["covered"].sum())
            roi = (w * UNIT_PROFIT_WIN - (n - w) * UNIT_LOSS) / n if n else 0
            late_compare.append({"model": name, "season": s, "bets": n,
                                 "win_pct": round(w / n, 4) if n else 0,
                                 "roi": round(roi, 4)})
    late_df = pd.DataFrame(late_compare)
    print(late_df.pivot(index="season", columns="model", values="roi").to_string())


if __name__ == "__main__":
    main()
