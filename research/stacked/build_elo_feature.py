"""
Build the Elo theta-diff stacked feature.

Walks every season independently with the best v2 Elo config (validated across 2019-2024):
    K=0.10, init_scale=1.0, h=0.05

For each team-week row, emits:
    theta              -- the team's pre-week skill in logit space
    theta_diff         -- theta_T - theta_O (positive = team favored over opponent on cover skill)
                          NaN when the opponent can't be paired (rare; e.g. unmatched spreads).

State is reset at the start of each season (theta_T = init_scale * logit(prev_season_cover_rate)).
The theta values written are PRE-week — i.e., they use ONLY information from prior weeks
(and prior seasons). Safe to use as a feature without leakage.

Output: research/stacked/elo_features.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "research" / "bayesian_dlm"))
from spike_v2 import load_games, pair_games, prev_season_cover_rate, sigmoid, logit  # type: ignore

OUT = Path(__file__).resolve().parent / "elo_features.parquet"

K = 0.10
INIT_SCALE = 1.0
H = 0.05


def build():
    df = load_games()
    out_rows = []

    for season in sorted(df["season"].unique()):
        rates, league = prev_season_cover_rate(df, season)
        theta: dict[str, float] = {}

        def init(team: str) -> float:
            p0 = rates.get(team, league)
            return INIT_SCALE * logit(p0)

        season_df = df[df["season"] == season]
        for w in sorted(season_df["week"].unique()):
            wk = season_df[season_df["week"] == w]
            pairs = pair_games(wk)
            # 1) record PRE-week theta for every row in this week
            team_to_opp: dict[str, str] = {}
            for a, b in pairs:
                team_to_opp[a["team"]] = b["team"]
                team_to_opp[b["team"]] = a["team"]
            for _, row in wk.iterrows():
                t = row["team"]
                if t not in theta:
                    theta[t] = init(t)
                opp = team_to_opp.get(t)
                if opp is not None:
                    if opp not in theta:
                        theta[opp] = init(opp)
                    diff = theta[t] - theta[opp]
                else:
                    diff = np.nan
                out_rows.append({
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "team": row["team"],
                    "theta": theta[t],
                    "theta_diff": diff,
                })
            # 2) apply updates after recording pre-week state
            for a, b in pairs:
                ta, tb = a["team"], b["team"]
                pa = sigmoid(theta[ta] - theta[tb] + H * a["is_home"])
                err = a["is_cover"] - pa
                theta[ta] += K * err
                theta[tb] -= K * err

    out = pd.DataFrame(out_rows)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} -- shape {out.shape}")
    print(f"theta_diff coverage: {out['theta_diff'].notna().sum()}/{len(out)} "
          f"({out['theta_diff'].notna().mean()*100:.1f}%)")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    build()
