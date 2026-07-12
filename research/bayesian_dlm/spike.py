"""
Bayesian dynamic prior — cover-probability spike.

Idea (per project plan, [[site-backlog]] item 5):
    Prior   = previous season's per-team cover behavior.
    Update  = each week's realized cover (Bernoulli).
    Weight  = shifts from the season prior toward recent weeks as the season
              progresses, via an exponential forgetting factor lambda.

Model (Beta-Binomial with forgetting):
    Season start, team T:
        alpha_T = prior_strength * cover_rate_prev_season(T)
        beta_T  = prior_strength * (1 - cover_rate_prev_season(T))

    Before week w, predicted P(cover) for team T:
        p_hat = alpha_T / (alpha_T + beta_T)

    After observing y in week w:
        alpha_T <- lambda * alpha_T + y
        beta_T  <- lambda * beta_T  + (1 - y)

    lambda in (0, 1]. lambda=1 -> Bayes with full memory; smaller lambda -> recent weeks dominate.

Bet rule (matches current honest backtest in [[site-revamp-plan]]):
    Take a team-week whenever p_hat > threshold. Threshold default 0.524 (-110 vig breakeven).
    Reported in parallel: threshold 0.50 (the current RF rule) for direct comparison.

Outputs:
    - Prints summary table per (lambda, prior_strength, threshold) sweep.
    - Writes research/bayesian_dlm/results.csv with all sweep rows.
    - Writes research/bayesian_dlm/weekly_equity.csv for the best config.
"""
from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CSV = REPO_ROOT / "nfl_training_data.csv"
CURRENT_CSV = REPO_ROOT / "nfl_current_data.csv"
OUT_DIR = Path(__file__).resolve().parent

# -110 American odds -> bet 110 to win 100. Breakeven cover rate = 110/210.
BREAKEVEN = 110 / 210  # ~0.5238
UNIT_PROFIT_WIN = 100 / 110  # +0.909u on a 1u bet at -110
UNIT_LOSS = 1.0

LABEL_COLS = ["season", "week", "team", "is_cover"]


def load_labels() -> pd.DataFrame:
    """Per team-week cover labels, 2018-2024. One row per team per played week."""
    hist = pd.read_csv(HISTORICAL_CSV, usecols=LABEL_COLS)
    curr = pd.read_csv(CURRENT_CSV, usecols=LABEL_COLS)
    df = pd.concat([hist, curr], ignore_index=True)
    df = df.dropna(subset=["is_cover"]).copy()
    df["is_cover"] = df["is_cover"].astype(int)
    df = df.sort_values(["season", "week", "team"]).reset_index(drop=True)
    return df


def prev_season_cover_rate(df: pd.DataFrame, season: int) -> dict[str, float]:
    """Map team -> prior-season cover rate. Falls back to league avg for new teams."""
    prev = df[df["season"] == season - 1]
    if prev.empty:
        return {}
    rates = prev.groupby("team")["is_cover"].mean().to_dict()
    league = prev["is_cover"].mean()
    return {**{"__league__": league}, **rates}


def backtest_one(
    df: pd.DataFrame,
    target_season: int,
    lam: float,
    prior_strength: float,
    threshold: float,
) -> dict:
    """Walk forward through target_season, return summary stats."""
    priors = prev_season_cover_rate(df, target_season)
    league = priors.get("__league__", 0.5)

    # init per-team Beta(alpha, beta)
    state: dict[str, tuple[float, float]] = {}

    def init_team(team: str) -> tuple[float, float]:
        p0 = priors.get(team, league)
        a = prior_strength * p0
        b = prior_strength * (1 - p0)
        return (a, b)

    season_df = df[df["season"] == target_season].copy()
    weeks = sorted(season_df["week"].unique())

    wins = losses = bets = 0
    units = 0.0
    weekly_rows = []

    for w in weeks:
        wk = season_df[season_df["week"] == w]
        # 1) predict for every team-row in this week using CURRENT state
        for _, row in wk.iterrows():
            team = row["team"]
            if team not in state:
                state[team] = init_team(team)
            a, b = state[team]
            p_hat = a / (a + b)
            if p_hat > threshold:
                bets += 1
                if row["is_cover"] == 1:
                    wins += 1
                    units += UNIT_PROFIT_WIN
                else:
                    losses += 1
                    units -= UNIT_LOSS
        # 2) AFTER predictions, update state with this week's observed covers
        for _, row in wk.iterrows():
            team = row["team"]
            a, b = state[team]
            y = row["is_cover"]
            state[team] = (lam * a + y, lam * b + (1 - y))
        weekly_rows.append(
            {"season": target_season, "week": w, "cum_bets": bets, "cum_units": units}
        )

    roi = units / bets if bets else 0.0
    win_pct = wins / bets if bets else 0.0
    return {
        "season": target_season,
        "lambda": lam,
        "prior_strength": prior_strength,
        "threshold": threshold,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_pct": round(win_pct, 4),
        "units": round(units, 2),
        "roi": round(roi, 4),
        "_weekly": weekly_rows,
    }


def main():
    df = load_labels()
    print(f"Loaded labels: {len(df)} team-weeks, seasons {sorted(df['season'].unique())}")

    target_season = 2024
    lambdas = [1.00, 0.95, 0.90, 0.80, 0.60]
    prior_strengths = [4, 8, 16, 32]
    thresholds = [0.50, BREAKEVEN]

    results = []
    weekly_keep = None
    best_key = None

    for lam, ps, thr in product(lambdas, prior_strengths, thresholds):
        out = backtest_one(df, target_season, lam, ps, thr)
        results.append({k: v for k, v in out.items() if k != "_weekly"})
        # keep best by ROI (min 30 bets to avoid trivial top)
        if out["bets"] >= 30:
            if best_key is None or out["roi"] > best_key["roi"]:
                best_key = out
                weekly_keep = out["_weekly"]

    res_df = pd.DataFrame(results).sort_values("roi", ascending=False)
    res_df.to_csv(OUT_DIR / "results.csv", index=False)

    print("\n=== Top 10 configs by ROI ===")
    print(res_df.head(10).to_string(index=False))

    print("\n=== Worst 5 configs by ROI ===")
    print(res_df.tail(5).to_string(index=False))

    # baselines for context
    season = df[df["season"] == target_season]
    base_rate = season["is_cover"].mean()
    print(
        f"\nBaseline 'bet every team-week' on {target_season}: "
        f"cover_rate={base_rate:.4f}, n={len(season)}, "
        f"units_at_-110 = {len(season)*(base_rate*UNIT_PROFIT_WIN - (1-base_rate)*UNIT_LOSS):.2f}"
    )
    print("Current RF backtest (2024 wks 4-15, p>0.5): 125-162, -48.36u, ROI -16.9% (from site-revamp-plan)")

    if weekly_keep is not None:
        wk_df = pd.DataFrame(weekly_keep)
        wk_df.to_csv(OUT_DIR / "weekly_equity.csv", index=False)
        print(
            f"\nBest config: lambda={best_key['lambda']}, prior_strength={best_key['prior_strength']}, "
            f"threshold={best_key['threshold']:.4f} -> {best_key['wins']}-{best_key['losses']}, "
            f"units={best_key['units']}, ROI={best_key['roi']*100:.2f}%"
        )
        print(f"Weekly equity curve -> {OUT_DIR / 'weekly_equity.csv'}")


if __name__ == "__main__":
    main()
