"""
Bayesian dynamic prior — v2, matchup-aware (Elo-style on covers).

Why v2: the v1 spike (spike.py) modeled each team's cover skill in isolation. It
ignored WHO they were playing, so a team that covered 55% vs weak defenses had its
55% taken at face value next week against a strong defense.

This version puts each team on a single latent skill axis in logit space and predicts
P(team T covers vs opponent O) as a paired comparison:

    p_T = sigmoid( theta_T - theta_O + h * is_home_T )

After observing the game outcome y_T in {0,1} (did T cover):

    theta_T <- theta_T + K * (y_T - p_T)
    theta_O <- theta_O - K * (y_T - p_T)

Initialization at season start:
    theta_T = init_scale * logit( prev_season_cover_rate(T) )
    (init_scale shrinks/inflates how strong the prior is in logit units.)

Hyperparams:
    K          : learning rate (analog of (1 - lambda) from v1)
    init_scale : prior strength in logit units
    h          : home-field cover bonus in logit units
    threshold  : bet rule, p > threshold

Bet rule + accounting identical to v1 (each side of every game is its own bet candidate,
matching the production "bet every team-week where p>thr" honest backtest).
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CSV = REPO_ROOT / "nfl_training_data.csv"
CURRENT_CSV = REPO_ROOT / "nfl_current_data.csv"
OUT_DIR = Path(__file__).resolve().parent

BREAKEVEN = 110 / 210
UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0

COLS = ["season", "week", "team", "spread", "is_home", "is_cover"]


def load_games() -> pd.DataFrame:
    hist = pd.read_csv(HISTORICAL_CSV, usecols=COLS)
    curr = pd.read_csv(CURRENT_CSV, usecols=COLS)
    df = pd.concat([hist, curr], ignore_index=True)
    df = df.dropna(subset=["is_cover", "spread", "is_home"]).copy()
    df["is_cover"] = df["is_cover"].astype(int)
    df["is_home"] = df["is_home"].astype(int)
    return df.sort_values(["season", "week", "team"]).reset_index(drop=True)


def pair_games(week_df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    """Pair team-rows into games by matching opposite spread + opposite is_home."""
    pairs = []
    used = set()
    rows = week_df.reset_index(drop=True)
    by_team = {r["team"]: i for i, r in rows.iterrows()}
    for i, r in rows.iterrows():
        if i in used:
            continue
        # find the row with -spread and opposite is_home, not yet used
        cands = rows[
            (~rows.index.isin(used))
            & (rows.index != i)
            & (np.isclose(rows["spread"], -r["spread"]))
            & (rows["is_home"] == 1 - r["is_home"])
        ]
        if cands.empty:
            continue
        j = cands.index[0]
        pairs.append((r, rows.loc[j]))
        used.add(i)
        used.add(j)
    return pairs


def prev_season_cover_rate(df: pd.DataFrame, season: int) -> tuple[dict, float]:
    prev = df[df["season"] == season - 1]
    if prev.empty:
        return {}, 0.5
    rates = prev.groupby("team")["is_cover"].mean().to_dict()
    league = float(prev["is_cover"].mean())
    return rates, league


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float, clip: float = 0.05) -> float:
    p = min(max(p, clip), 1 - clip)
    return float(np.log(p / (1 - p)))


def backtest(
    df: pd.DataFrame,
    season: int,
    K: float,
    init_scale: float,
    h: float,
    threshold: float,
) -> dict:
    rates, league = prev_season_cover_rate(df, season)
    theta: dict[str, float] = {}

    def init(team: str) -> float:
        p0 = rates.get(team, league)
        return init_scale * logit(p0)

    season_df = df[df["season"] == season]
    weeks = sorted(season_df["week"].unique())

    bets = wins = losses = 0
    units = 0.0
    weekly = []

    for w in weeks:
        wk = season_df[season_df["week"] == w]
        pairs = pair_games(wk)
        # 1) predict + place bets using current theta
        post_updates = []  # defer updates till after all preds in the week
        for a, b in pairs:
            ta, tb = a["team"], b["team"]
            if ta not in theta:
                theta[ta] = init(ta)
            if tb not in theta:
                theta[tb] = init(tb)
            # P(a covers) and P(b covers); they should sum to ~1 in this model
            pa = sigmoid(theta[ta] - theta[tb] + h * a["is_home"])
            pb = sigmoid(theta[tb] - theta[ta] + h * b["is_home"])
            for row, p in [(a, pa), (b, pb)]:
                if p > threshold:
                    bets += 1
                    if row["is_cover"] == 1:
                        wins += 1
                        units += UNIT_PROFIT_WIN
                    else:
                        losses += 1
                        units -= UNIT_LOSS
            # store deferred update (use a's outcome — b's is just 1-a's in cover terms,
            # ignoring pushes which are rare)
            post_updates.append((ta, tb, a["is_cover"], pa))
        # 2) apply updates
        for ta, tb, ya, pa in post_updates:
            err = ya - pa
            theta[ta] += K * err
            theta[tb] -= K * err
        weekly.append({"season": season, "week": w, "cum_bets": bets, "cum_units": units})

    roi = units / bets if bets else 0.0
    return {
        "season": season,
        "K": K,
        "init_scale": init_scale,
        "h": h,
        "threshold": threshold,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_pct": round(wins / bets, 4) if bets else 0.0,
        "units": round(units, 2),
        "roi": round(roi, 4),
        "_weekly": weekly,
    }


def main():
    df = load_games()
    season = 2024
    print(f"Loaded {len(df)} team-weeks, seasons {sorted(df['season'].unique())}")

    Ks = [0.05, 0.10, 0.20, 0.40]
    init_scales = [0.0, 0.25, 0.5, 1.0]   # 0.0 = ignore prior, start at theta=0
    hs = [0.0, 0.05, 0.10]                 # logit-space home bump
    thresholds = [0.50, BREAKEVEN]

    results = []
    best = None
    best_weekly = None
    for K, s, h, thr in product(Ks, init_scales, hs, thresholds):
        out = backtest(df, season, K, s, h, thr)
        results.append({k: v for k, v in out.items() if k != "_weekly"})
        if out["bets"] >= 30 and (best is None or out["roi"] > best["roi"]):
            best = out
            best_weekly = out["_weekly"]

    res_df = pd.DataFrame(results).sort_values("roi", ascending=False)
    res_df.to_csv(OUT_DIR / "results_v2.csv", index=False)

    print("\n=== Top 10 configs (v2, matchup-aware) ===")
    print(res_df.head(10).to_string(index=False))

    print("\n=== Bottom 5 ===")
    print(res_df.tail(5).to_string(index=False))

    print(
        f"\nBest v2: K={best['K']}, init_scale={best['init_scale']}, h={best['h']}, "
        f"threshold={best['threshold']:.4f} -> {best['wins']}-{best['losses']}, "
        f"units={best['units']}, ROI={best['roi']*100:.2f}%"
    )

    if best_weekly:
        pd.DataFrame(best_weekly).to_csv(OUT_DIR / "weekly_equity_v2.csv", index=False)

    # context
    season_df = df[df["season"] == season]
    print(
        f"\nContext - 2024: cover_base_rate={season_df['is_cover'].mean():.4f}, "
        f"n_team_weeks={len(season_df)}"
    )
    print("Baselines: prod RF -16.9%, v1 spike best -6.44%, breakeven -110 = +0.00%")


if __name__ == "__main__":
    main()
