"""
Validate the v2 (matchup-aware Elo) spike across multiple seasons.

Runs the same hyperparam sweep as spike_v2.py on each of 2019..2024 (2018 has no
prior season available in the data we hold). Reports:

1. Per-season top configs (sanity check — does any config dominate everywhere?).
2. Aggregate ROI summed across all seasons per config (is there a config that's
   net-profitable when you back-test it identically across every year?).
3. Specifically, how the 2024-best config (K=0.05, init_scale=0.0, h=0.0,
   threshold=BE) performs on each other season.

Question this answers: is init_scale=0 (ignore last-season prior) a real signal
or a 2024 fluke?
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from spike_v2 import backtest, load_games, BREAKEVEN  # type: ignore

OUT_DIR = Path(__file__).resolve().parent


def main():
    df = load_games()
    seasons = [s for s in sorted(df["season"].unique()) if s >= 2019]
    print(f"Validating across seasons: {seasons}")

    Ks = [0.05, 0.10, 0.20]
    init_scales = [0.0, 0.25, 0.5, 1.0]
    hs = [0.0, 0.05]
    thresholds = [BREAKEVEN]  # focus on the breakeven rule (cleaner story)

    rows = []
    for season in seasons:
        for K, s, h, thr in product(Ks, init_scales, hs, thresholds):
            out = backtest(df, season, K, s, h, thr)
            rows.append({k: v for k, v in out.items() if k != "_weekly"})

    full = pd.DataFrame(rows)
    full.to_csv(OUT_DIR / "validate_v2_per_season.csv", index=False)

    # 1) Per-season top 3
    print("\n=== Per-season top 3 by ROI (min 20 bets) ===")
    for season in seasons:
        sub = full[(full["season"] == season) & (full["bets"] >= 20)].sort_values("roi", ascending=False)
        print(f"\n-- {season} --")
        print(sub.head(3).to_string(index=False))

    # 2) Aggregate across seasons (same config, pooled bets/units)
    agg = (
        full.groupby(["K", "init_scale", "h", "threshold"], as_index=False)
        .agg(bets=("bets", "sum"), wins=("wins", "sum"), losses=("losses", "sum"), units=("units", "sum"))
    )
    agg["win_pct"] = (agg["wins"] / agg["bets"]).round(4)
    agg["roi"] = (agg["units"] / agg["bets"]).round(4)
    agg = agg.sort_values("roi", ascending=False)
    agg.to_csv(OUT_DIR / "validate_v2_aggregate.csv", index=False)
    print("\n=== Aggregate across 2019-2024 (pooled), top 10 ===")
    print(agg.head(10).to_string(index=False))
    print("\n=== Aggregate bottom 5 ===")
    print(agg.tail(5).to_string(index=False))

    # 3) 2024-best config across each year
    target = {"K": 0.05, "init_scale": 0.0, "h": 0.0, "threshold": BREAKEVEN}
    print(f"\n=== 2024-best config across each season ===")
    print("(K=0.05, init_scale=0.0, h=0.0, threshold=BE ~ 0.5238)\n")
    mask = (
        (full["K"] == target["K"])
        & (full["init_scale"] == target["init_scale"])
        & (full["h"] == target["h"])
        & (full["threshold"] == target["threshold"])
    )
    print(full[mask].sort_values("season").to_string(index=False))

    # 4) "init_scale=0 wins on average" check: average rank by ROI of each init_scale across seasons
    print("\n=== Does init_scale=0 dominate? Avg per-season rank by ROI (lower=better) ===")
    full["per_season_rank"] = full.groupby("season")["roi"].rank(ascending=False)
    rank_by_init = full.groupby("init_scale")["per_season_rank"].mean().round(2).sort_values()
    print(rank_by_init.to_string())


if __name__ == "__main__":
    main()
