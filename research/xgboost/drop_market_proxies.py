"""
Drop spread-/market-proxy features and re-test XGB-A.

Hypothesis: the model's flat spread sensitivity is because moneyline + total-market
features encode the same info as spread. Dropping them forces the model to use spread.

X. Refit XGB-A on the reduced feature set (51 - 5 = 46 features).
Y. Backtest ROI (bet every non-push game) -- does the edge survive?
Z. Re-run spread sensitivity -- does the curve actually move now?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features.parquet"

UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
ID_COLS = ["season", "week", "team", "is_cover"]
DROP = ["moneyline", "spread_odds", "exp_total_points", "over_odds", "under_odds"]

XGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=7, n_jobs=4,
)


def fit_predict(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    parts = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        tr = df[df["season"] < s]
        te = df[df["season"] == s].copy()
        m = xgb.XGBClassifier(**XGB_KW)
        m.fit(tr[feat], tr["is_cover"])
        te["_p"] = m.predict_proba(te[feat])[:, 1]
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def pair_and_score(predicted: pd.DataFrame) -> dict:
    bets = wins = pushes = 0
    units = 0.0
    per_season: dict[int, list[int]] = {}
    for (s, w), wk in predicted.groupby(["season", "week"], sort=False):
        wk = wk.reset_index(drop=True); used = set()
        for i in range(len(wk)):
            if i in used: continue
            ri = wk.iloc[i]
            cand = [j for j in range(len(wk)) if j != i and j not in used
                    and np.isclose(wk.iloc[j]["spread"], -ri["spread"])
                    and wk.iloc[j]["is_home"] == 1 - ri["is_home"]]
            if not cand: continue
            j = cand[0]; used.add(i); used.add(j); rj = wk.iloc[j]
            pa, pb, ya, yb = ri["_p"], rj["_p"], int(ri["is_cover"]), int(rj["is_cover"])
            if ya == 0 and yb == 0:
                pushes += 1; continue
            pick_y = ya if pa >= pb else yb
            bets += 1
            if pick_y == 1:
                wins += 1; units += UNIT_PROFIT_WIN
                per_season.setdefault(int(s), [0, 0])[0] += 1
                per_season[int(s)][1] += 1
            else:
                units -= UNIT_LOSS
                per_season.setdefault(int(s), [0, 0])[1] += 1
    out = dict(bets=bets, pushes=pushes, wins=wins,
               win_pct=round(wins/bets, 4) if bets else 0,
               units=round(units, 2),
               roi=round(units/bets, 4) if bets else 0)
    out["per_season"] = {s: {"wins": wb[0], "bets": wb[1],
                              "win_pct": round(wb[0]/wb[1], 4) if wb[1] else 0,
                              "roi": round((wb[0]*UNIT_PROFIT_WIN - (wb[1]-wb[0])*UNIT_LOSS)/wb[1], 4) if wb[1] else 0}
                          for s, wb in sorted(per_season.items())}
    return out


def sweep_spread(model, row: pd.Series, feat: list[str], grid: np.ndarray) -> np.ndarray:
    X = pd.DataFrame([row[feat].copy()] * len(grid))
    X["spread"] = grid
    if "is_fav" in X.columns:
        X["is_fav"] = (grid < 0).astype(int)
    return model.predict_proba(X)[:, 1]


def main():
    df = pd.read_parquet(FEATURES)
    all_feat = [c for c in df.columns if c not in ID_COLS]
    feat = [c for c in all_feat if c not in DROP]
    print(f"Full feature set: {len(all_feat)}.  Dropped {DROP}.  Using {len(feat)}.")

    # Y. Backtest on reduced set
    print("\n=== Y. Backtest XGB-A (no market proxies) ===")
    pred = fit_predict(df, feat)
    r = pair_and_score(pred)
    print(f"Aggregate: bets={r['bets']}, wins={r['wins']}, pushes={r['pushes']}")
    print(f"           win%={r['win_pct']*100:.2f}%, ROI={r['roi']*100:+.2f}%, units={r['units']}")
    print(f"\nPer season:")
    print(pd.DataFrame.from_dict(r["per_season"], orient="index").to_string())

    # Z. Sensitivity check (refit a 2024-target model)
    print("\n=== Z. Spread sensitivity (model trained on <2024, sweep on 2024) ===")
    tr = df[df["season"] < 2024]
    m = xgb.XGBClassifier(**XGB_KW); m.fit(tr[feat], tr["is_cover"])
    season_df = df[df["season"] == 2024].reset_index(drop=True)
    grid = np.arange(-14, 14.5, 0.5)

    # monotonicity + range
    ranges = []
    mono_dec = mono_inc = 0
    for _, row in season_df.iterrows():
        p = sweep_spread(m, row, feat, grid)
        diffs = np.diff(p)
        ranges.append(p.max() - p.min())
        if (diffs <= 1e-6).all(): mono_dec += 1
        elif (diffs >= -1e-6).all(): mono_inc += 1
    n = len(season_df)
    print(f"Curves: {n}.  Avg P(cover) range across [-14,+14]: {np.mean(ranges):.3f}  "
          f"(was 0.05 with proxies; should be >0.15 if model now uses spread)")
    print(f"Monotonic decreasing: {mono_dec} ({mono_dec/n*100:.1f}%)  "
          f"Monotonic increasing: {mono_inc} ({mono_inc/n*100:.1f}%)")

    # key-number slopes
    print(f"\nKey-number slopes (dP per 2-point spread move, avg across 2024 games):")
    slopes = {}
    for key in [-7, -3, 0, 3, 7]:
        s_below, s_above = [], []
        i_below = int(np.where(np.isclose(grid, key - 1.0))[0][0])
        i_above = int(np.where(np.isclose(grid, key + 1.0))[0][0])
        for _, row in season_df.iterrows():
            p = sweep_spread(m, row, feat, grid)
            s_below.append(p[i_below]); s_above.append(p[i_above])
        slope = float(np.mean(s_above) - np.mean(s_below))
        slopes[key] = slope
        print(f"  spread={key:+d}: dP/d(2pt) = {slope:+.4f}")

    # a few examples
    print(f"\nExample curves (3 random 2024 games):")
    for _, row in season_df.sample(3, random_state=11).iterrows():
        p = sweep_spread(m, row, feat, grid)
        print(f"\n{row['team']} wk{int(row['week'])} (actual spread = {row['spread']:+.1f}):")
        for sp, pv in zip(grid[::4], p[::4]):
            mk = " <-- actual" if abs(sp - row["spread"]) < 0.25 else ""
            print(f"  spread={sp:+.1f}: P(cover)={pv:.3f}{mk}")


if __name__ == "__main__":
    main()
