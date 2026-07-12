"""
Lockbox 2024 -- final clean OOS validation of the margin model.

Mimics exactly what the production training job will do at the start of 2025:
  1. Train XGBRegressor on all available data (2018-2023) with target = margin.
  2. Calibrate Student-t (scale, nu) on OOS residuals from the 2023 holdout.
  3. Predict mu_hat on every 2024 team-week.
  4. Compute P(cover) at the market spread and write a frozen pick log.
  5. Score against actual covers.

No hyperparameter tuning. No multi-fold averaging. No peeking at 2024.
This is the number we'd see if we'd deployed this model for the 2024 season.

Output:
  research/xgboost/lockbox_2024_picks.csv  -- every team-week pick (frozen)
  research/xgboost/lockbox_2024_summary.json -- aggregate + per-week stats
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features.parquet"
PICKS_OUT = ROOT / "lockbox_2024_picks.csv"
SUMMARY_OUT = ROOT / "lockbox_2024_summary.json"

UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
ID_COLS = ["season", "week", "team", "is_cover"]
DROP_FEATS = ["spread", "spread_odds", "moneyline", "exp_total_points",
              "over_odds", "under_odds", "is_fav"]

XGB_REG_KW = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="reg:squarederror", eval_metric="rmse",
    tree_method="hist", random_state=7, n_jobs=4,
)


def load_margin_target() -> pd.DataFrame:
    import nfl_data_py as nfl
    sched = nfl.import_schedules(list(range(2018, 2025)))
    sched = sched.dropna(subset=["home_score", "away_score"])
    home = sched.rename(columns={"home_team": "team"})
    home["margin"] = home["home_score"] - home["away_score"]
    away = sched.rename(columns={"away_team": "team"})
    away["margin"] = away["away_score"] - away["home_score"]
    out = pd.concat([
        home[["season", "week", "team", "margin"]],
        away[["season", "week", "team", "margin"]],
    ], ignore_index=True)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def main():
    df = pd.read_parquet(FEATURES).merge(
        load_margin_target(), on=["season", "week", "team"], how="inner"
    )
    feat = [c for c in df.columns if c not in ID_COLS + ["margin"] and c not in DROP_FEATS]
    print(f"Loaded {len(df):,} rows.  Margin-model features: {len(feat)}")

    # ----- 1. Calibration: train on 2018-2022, predict on 2023 -> get residuals
    print("\n[1/3] OOS calibration (train 2018-2022, residuals on 2023)...")
    tr_cal = df[df["season"] < 2023]
    cv = df[df["season"] == 2023]
    m_cal = xgb.XGBRegressor(**XGB_REG_KW)
    m_cal.fit(tr_cal[feat], tr_cal["margin"])
    resid = cv["margin"].values - m_cal.predict(cv[feat])
    nu, _, scale = stats.t.fit(resid, floc=0)
    nu = float(np.clip(nu, 3.0, 50.0))
    sigma_norm = float(np.std(resid))
    print(f"  Calibration: nu={nu:.3f}, scale_t={scale:.3f}, sigma_normal={sigma_norm:.3f}")

    # ----- 2. Final training: 2018-2023
    print("\n[2/3] Final training on 2018-2023...")
    tr_final = df[df["season"] < 2024]
    m_final = xgb.XGBRegressor(**XGB_REG_KW)
    m_final.fit(tr_final[feat], tr_final["margin"])

    # ----- 3. Predict on 2024
    print("\n[3/3] Predicting on 2024 (frozen)...")
    test = df[df["season"] == 2024].copy()
    test["mu_hat"] = m_final.predict(test[feat])
    test["scale_t"] = scale
    test["nu_t"] = nu
    test["sigma_normal"] = sigma_norm
    # P(cover) at market spread, both Normal and Student-t
    test["p_cover_normal"] = stats.norm.cdf((test["mu_hat"] + test["spread"]) / sigma_norm)
    test["p_cover_t"] = stats.t.cdf((test["mu_hat"] + test["spread"]) / scale, df=nu)
    test["edge_pts"] = test["mu_hat"] + test["spread"]

    # Pair per game, pick higher-prob side, score
    pick_rows = []
    pushes = 0
    for (s, w), wk in test.groupby(["season", "week"], sort=False):
        wk = wk.reset_index(drop=True)
        used = set()
        for i in range(len(wk)):
            if i in used: continue
            ri = wk.iloc[i]
            cand = [j for j in range(len(wk)) if j != i and j not in used
                    and np.isclose(wk.iloc[j]["spread"], -ri["spread"])
                    and wk.iloc[j]["is_home"] == 1 - ri["is_home"]]
            if not cand: continue
            j = cand[0]; used.add(i); used.add(j); rj = wk.iloc[j]
            ya, yb = int(ri["is_cover"]), int(rj["is_cover"])
            pa, pb = float(ri["p_cover_t"]), float(rj["p_cover_t"])
            if ya == 0 and yb == 0:
                pushes += 1
                pick_rows.append({"season": int(s), "week": int(w),
                                  "team_a": ri["team"], "team_b": rj["team"],
                                  "spread_a": ri["spread"], "is_home_a": ri["is_home"],
                                  "mu_a": ri["mu_hat"], "mu_b": rj["mu_hat"],
                                  "p_t_a": pa, "p_t_b": pb,
                                  "edge_a": float(ri["edge_pts"]),
                                  "pick": ri["team"] if pa >= pb else rj["team"],
                                  "p_pick": max(pa, pb),
                                  "covered": None, "result": "PUSH", "units": 0.0})
                continue
            if pa >= pb:
                pick, pick_y, pick_p = ri["team"], ya, pa
            else:
                pick, pick_y, pick_p = rj["team"], yb, pb
            units = UNIT_PROFIT_WIN if pick_y == 1 else -UNIT_LOSS
            pick_rows.append({"season": int(s), "week": int(w),
                              "team_a": ri["team"], "team_b": rj["team"],
                              "spread_a": ri["spread"], "is_home_a": ri["is_home"],
                              "mu_a": ri["mu_hat"], "mu_b": rj["mu_hat"],
                              "p_t_a": pa, "p_t_b": pb,
                              "edge_a": float(ri["edge_pts"]),
                              "pick": pick, "p_pick": pick_p,
                              "covered": pick_y,
                              "result": "WIN" if pick_y == 1 else "LOSS",
                              "units": units})

    picks = pd.DataFrame(pick_rows)
    picks.to_csv(PICKS_OUT, index=False)

    # ----- Aggregate
    bets_only = picks[picks["result"].isin(["WIN", "LOSS"])]
    n = len(bets_only); w = int((bets_only["result"] == "WIN").sum())
    units = float(bets_only["units"].sum())
    roi = units / n if n else 0.0
    win_pct = w / n if n else 0.0

    # Confidence interval on win rate via Wilson
    from math import sqrt
    if n > 0:
        z = 1.96
        denom = 1 + z**2 / n
        center = (w/n + z**2/(2*n)) / denom
        margin = z * sqrt((w/n)*(1 - w/n)/n + z**2/(4*n**2)) / denom
        wilson_lo, wilson_hi = center - margin, center + margin
    else:
        wilson_lo = wilson_hi = 0

    # Per-week breakdown
    per_week = []
    for wk, g in bets_only.groupby("week"):
        nw = len(g); ww = int((g["result"] == "WIN").sum())
        uw = float(g["units"].sum())
        per_week.append({"week": int(wk), "bets": nw, "wins": ww,
                         "win_pct": round(ww/nw, 4) if nw else 0,
                         "units": round(uw, 3),
                         "roi": round(uw/nw, 4) if nw else 0})

    # Confidence buckets
    edges = bets_only["p_pick"].values
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]
    by_conf = []
    for lo, hi in buckets:
        mask = (edges >= lo) & (edges < hi)
        nb = int(mask.sum())
        wb = int((bets_only["result"][mask] == "WIN").sum())
        ub = float(bets_only["units"][mask].sum())
        by_conf.append({"p_range": f"[{lo:.2f}, {hi:.2f})",
                        "bets": nb, "wins": wb,
                        "win_pct": round(wb/nb, 4) if nb else 0,
                        "roi": round(ub/nb, 4) if nb else 0})

    summary = {
        "model_version": "margin_v2_lockbox_2024",
        "trained_on": "2018-2023",
        "test_on": "2024",
        "features_used": len(feat),
        "dropped": DROP_FEATS,
        "calibration": {"nu_t": round(nu, 3), "scale_t": round(scale, 3),
                        "sigma_normal": round(sigma_norm, 3),
                        "resid_n": int(len(resid)),
                        "resid_source": "2023 OOS"},
        "aggregate": {
            "bets": n, "wins": w, "pushes": pushes,
            "win_pct": round(win_pct, 4),
            "units": round(units, 3),
            "roi": round(roi, 4),
            "wilson_95_ci": [round(wilson_lo, 4), round(wilson_hi, 4)],
            "breakeven_winpct_at_-110": round(110/(100+110), 4),
        },
        "per_week": per_week,
        "by_confidence": by_conf,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2))

    print(f"\n=== LOCKBOX 2024 RESULT ===")
    print(f"Bets: {n}, Wins: {w}, Pushes: {pushes}")
    print(f"Win%: {win_pct*100:.2f}%  (Wilson 95%: [{wilson_lo*100:.2f}, {wilson_hi*100:.2f}])")
    print(f"Breakeven at -110: 52.38%")
    print(f"Units: {units:+.2f}, ROI: {roi*100:+.2f}%")
    print(f"\nPer-week:")
    print(pd.DataFrame(per_week).to_string(index=False))
    print(f"\nBy confidence bucket:")
    print(pd.DataFrame(by_conf).to_string(index=False))
    print(f"\nWritten: {PICKS_OUT.name}, {SUMMARY_OUT.name}")


if __name__ == "__main__":
    main()
