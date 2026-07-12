"""
Spread sensitivity analysis -- prototype for the sliding-scale UI + diagnostic.

For each game in 2024, sweep the spread input from -14 to +14 in 0.5-point increments,
holding all other features constant. Plot/print the resulting P(cover) curve.

What we're checking:
1. SHAPE: is the curve smooth and monotonic? Pathological models (flat / wildly bouncing)
   would tell us spread isn't being used coherently.
2. KEY NUMBERS: NFL key numbers are 3 and 7 (the most common margins). A sensible model
   should show steeper P(cover) gradient near those.
3. CONSISTENCY: at the actual market spread, does the prediction match the
   already-reported per-game P? (sanity check the rebuild.)
4. PRODUCT: gives us the data the slider UI needs.

Output:
- spread_sensitivity_curves.csv: spread x p_cover for every (season, week, team) row
- Summary stats: avg slope, % of curves that are monotonic, key-number behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features.parquet"
ID_COLS = ["season", "week", "team", "is_cover"]

XGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=7, n_jobs=4,
)


def train_for(df: pd.DataFrame, feat: list[str], target_season: int) -> xgb.XGBClassifier:
    tr = df[df["season"] < target_season]
    m = xgb.XGBClassifier(**XGB_KW)
    m.fit(tr[feat], tr["is_cover"])
    return m


def sweep_spread(model, row: pd.Series, feat: list[str], grid: np.ndarray) -> np.ndarray:
    """For one team-week row, vary `spread` across grid, return P(cover) at each."""
    X = pd.DataFrame([row[feat].copy()] * len(grid))
    X["spread"] = grid
    # is_fav is derived from spread (spread < 0); update it for consistency
    if "is_fav" in X.columns:
        X["is_fav"] = (grid < 0).astype(int)
    return model.predict_proba(X)[:, 1]


def main():
    df = pd.read_parquet(FEATURES)
    feat = [c for c in df.columns if c not in ID_COLS]
    print(f"Loaded {len(df):,} rows x {len(feat)} features")

    target = 2024
    model = train_for(df, feat, target)
    season_df = df[df["season"] == target].reset_index(drop=True)

    grid = np.arange(-14, 14.5, 0.5)
    curves = []
    for i, row in season_df.iterrows():
        probs = sweep_spread(model, row, feat, grid)
        for sp, p in zip(grid, probs):
            curves.append({
                "season": int(row["season"]), "week": int(row["week"]),
                "team": row["team"], "actual_spread": float(row["spread"]),
                "queried_spread": float(sp), "p_cover": float(p),
            })
    out = pd.DataFrame(curves)
    out.to_csv(ROOT / "spread_sensitivity_curves.csv", index=False)
    print(f"Wrote {len(out):,} rows -> spread_sensitivity_curves.csv")

    # Diagnostic 1: monotonicity of P(cover) in spread (should be monotonic decreasing —
    # bigger spread means team is bigger dog/smaller favorite, P(cover) should fall as
    # spread increases for the side that was a favorite, rise for the dog... actually
    # it's about the team's spread directly. If team's spread = -7 they're favored by 7;
    # making it -10 makes them MORE favored = harder to cover = lower P(cover).
    # So P(cover) should be monotonically DECREASING in queried_spread for each row.)
    print("\n=== Monotonicity check ===")
    mono = []
    for (s, w, t), g in out.groupby(["season", "week", "team"], sort=False):
        p = g.sort_values("queried_spread")["p_cover"].values
        diffs = np.diff(p)
        mono.append({
            "team": t, "week": int(w),
            "monotonic_dec": bool((diffs <= 1e-6).all()),
            "monotonic_inc": bool((diffs >= -1e-6).all()),
            "max_p": float(p.max()), "min_p": float(p.min()), "range": float(p.max() - p.min()),
        })
    mdf = pd.DataFrame(mono)
    print(f"Total team-week curves: {len(mdf)}")
    print(f"Monotonic decreasing (expected for cover prob vs spread): "
          f"{mdf['monotonic_dec'].sum()} ({mdf['monotonic_dec'].mean()*100:.1f}%)")
    print(f"Monotonic increasing (unexpected):                       "
          f"{mdf['monotonic_inc'].sum()} ({mdf['monotonic_inc'].mean()*100:.1f}%)")
    print(f"Non-monotonic: {(~mdf['monotonic_dec'] & ~mdf['monotonic_inc']).sum()}")
    print(f"Avg P(cover) range across spreads: {mdf['range'].mean():.3f}")
    print(f"  (if model uses spread meaningfully, this should be substantial, e.g. >0.10)")

    # Diagnostic 2: avg slope at key numbers
    print("\n=== Spread sensitivity around key numbers (-7, -3, 0, +3, +7) ===")
    for key in [-7, -3, 0, 3, 7]:
        # slope across +/- 1 point of key
        below = out[np.isclose(out["queried_spread"], key - 1.0)].groupby(
            ["season", "week", "team"])["p_cover"].first()
        above = out[np.isclose(out["queried_spread"], key + 1.0)].groupby(
            ["season", "week", "team"])["p_cover"].first()
        slope = (above - below).mean()  # change in P over 2-point spread move
        print(f"  spread={key:+d}: avg dP(cover)/d(2pt) = {slope:+.4f}  "
              f"(more negative = stronger spread reaction)")

    # Diagnostic 3: a few example curves
    print("\n=== Example curves (3 random 2024 games) ===")
    sample = season_df.sample(3, random_state=11)
    for _, row in sample.iterrows():
        probs = sweep_spread(model, row, feat, grid)
        print(f"\n{row['team']} wk{int(row['week'])} (actual spread = {row['spread']:+.1f}):")
        for sp, p in zip(grid[::4], probs[::4]):  # every 2 points
            marker = " <-- actual" if abs(sp - row["spread"]) < 0.25 else ""
            print(f"  spread={sp:+.1f}: P(cover)={p:.3f}{marker}")


if __name__ == "__main__":
    main()
