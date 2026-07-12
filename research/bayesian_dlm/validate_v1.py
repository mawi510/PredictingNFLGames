"""
Multi-season validation of spike v1 (Beta-Binomial with forgetting factor lambda).

Closes the open loop: spike.py was only ever run on 2024. Per the new sample-size rule
[[feedback-sports-betting-sample-size]], any single-season ROI is unvalidated.

Same sweep as the original spike: lambda x prior_strength x threshold.
Run on each of 2019..2024 (2018 has no prior season available).
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from spike import backtest_one, load_labels, BREAKEVEN  # type: ignore

OUT_DIR = Path(__file__).resolve().parent


def main():
    df = load_labels()
    seasons = [s for s in sorted(df["season"].unique()) if s >= 2019]
    print(f"Validating spike v1 across seasons: {seasons}")

    lambdas = [1.00, 0.95, 0.90, 0.80, 0.60]
    prior_strengths = [4, 8, 16, 32]
    thresholds = [BREAKEVEN]  # focus on the -110 breakeven threshold

    rows = []
    for season in seasons:
        for lam, ps, thr in product(lambdas, prior_strengths, thresholds):
            out = backtest_one(df, season, lam, ps, thr)
            rows.append({k: v for k, v in out.items() if k != "_weekly"})

    full = pd.DataFrame(rows)
    full.to_csv(OUT_DIR / "validate_v1_per_season.csv", index=False)

    agg = (
        full.groupby(["lambda", "prior_strength", "threshold"], as_index=False)
        .agg(bets=("bets", "sum"), wins=("wins", "sum"),
             losses=("losses", "sum"), units=("units", "sum"))
    )
    agg["win_pct"] = (agg["wins"] / agg["bets"]).round(4)
    agg["roi"] = (agg["units"] / agg["bets"]).round(4)
    agg = agg.sort_values("roi", ascending=False)
    agg.to_csv(OUT_DIR / "validate_v1_aggregate.csv", index=False)

    print("\n=== Aggregate across 2019-2024 (pooled), top 10 ===")
    print(agg.head(10).to_string(index=False))
    print("\n=== Bottom 5 ===")
    print(agg.tail(5).to_string(index=False))

    # The 2024-best from spike.py was lambda=0.6, prior_strength=4, threshold=BE
    target = {"lambda": 0.6, "prior_strength": 4, "threshold": BREAKEVEN}
    print(f"\n=== Original 2024-best config across each season ===")
    print("(lambda=0.6, prior_strength=4, threshold=BE ~ 0.5238)\n")
    mask = (
        (full["lambda"] == target["lambda"])
        & (full["prior_strength"] == target["prior_strength"])
        & (full["threshold"] == target["threshold"])
    )
    print(full[mask].sort_values("season").to_string(index=False))

    # Which lambda dominates on average?
    print("\n=== Avg per-season ROI rank by lambda (lower=better) ===")
    full["per_season_rank"] = full.groupby("season")["roi"].rank(ascending=False)
    print(full.groupby("lambda")["per_season_rank"].mean().round(2).sort_values().to_string())

    print("\n=== Avg per-season ROI rank by prior_strength ===")
    print(full.groupby("prior_strength")["per_season_rank"].mean().round(2).sort_values().to_string())


if __name__ == "__main__":
    main()
