"""
Re-run prior baselines under the CORRECTED accounting (per-game pick + push exclusion).

Why: backtest_v3.py revealed the per-team-week bet rule was double-betting hedged games
and counting pushes as losses. Every prior scoreboard entry was biased downward. Need a
fair scoreboard.

Models re-run (all with per-game pick + push-excluded ROI):

  XGB-A   XGB on 51 curated features, no calibration
  XGB-B   XGB on 54 features (+ prior + blended), no calibration
  XGB-C   XGB on 51 + isotonic calibration
  XGB-D   XGB on 54 + isotonic calibration  (was the prior leader at +9.93%)
  ELO-V2  Matchup-aware Elo (K=0.10, init_scale=1.0, h=0.05) -- best v2 agg
  BETA-V1 Beta-Binomial with forgetting (lambda=1.0, prior_strength=32) -- best v1 agg

Plus a favoritism diagnostic per model: of the bets it places, what % are on the favorite
vs underdog. Context (user): historically underdogs cover ~50% but in specific eras up to
60%; a model that only bets favorites blindly is probably overfitting to recency.

Output: a single scoreboard table.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "research" / "bayesian_dlm"))
from spike_v2 import pair_games, sigmoid, logit  # type: ignore

ROOT = Path(__file__).resolve().parent
FEATURES_V1 = ROOT / "features.parquet"
FEATURES_V2 = ROOT / "features_v2.parquet"

BREAKEVEN = 110 / 210
UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
THRESHOLD = BREAKEVEN
ID_COLS = ["season", "week", "team", "is_cover"]


def per_game_pick_and_score(test: pd.DataFrame, probs: np.ndarray) -> dict:
    """Apply corrected accounting: per-game argmax pick + push exclusion.

    Also returns favoritism diagnostic: how many bets on favorites vs underdogs.
    A team is the favorite if its spread < 0.
    """
    t = test.reset_index(drop=True).copy()
    t["_p"] = probs

    bets = wins = pushes = both_over = no_pick = games = 0
    fav_bets = dog_bets = pickem = 0
    units = 0.0

    for (s, w), wk in t.groupby(["season", "week"], sort=False):
        rows = wk.reset_index(drop=True)
        used = set()
        for i in range(len(rows)):
            if i in used:
                continue
            ri = rows.iloc[i]
            cand = [
                j for j in range(len(rows))
                if j != i and j not in used
                and np.isclose(rows.iloc[j]["spread"], -ri["spread"])
                and rows.iloc[j]["is_home"] == 1 - ri["is_home"]
            ]
            if not cand:
                continue
            j = cand[0]
            used.add(i); used.add(j)
            games += 1
            rj = rows.iloc[j]
            pa, pb, ya, yb = ri["_p"], rj["_p"], int(ri["is_cover"]), int(rj["is_cover"])
            if pa > THRESHOLD and pb > THRESHOLD:
                both_over += 1
            if ya == 0 and yb == 0:
                pushes += 1
                continue
            if pa >= pb:
                pick = (ri["spread"], pa, ya)
            else:
                pick = (rj["spread"], pb, yb)
            sp, pick_p, pick_y = pick
            if pick_p <= THRESHOLD:
                no_pick += 1
                continue
            bets += 1
            if sp < 0:
                fav_bets += 1
            elif sp > 0:
                dog_bets += 1
            else:
                pickem += 1
            if pick_y == 1:
                wins += 1
                units += UNIT_PROFIT_WIN
            else:
                units -= UNIT_LOSS

    return dict(
        games=games, pushes_excluded=pushes,
        bets=bets, wins=wins, losses=bets - wins,
        win_pct=round(wins / bets, 4) if bets else 0.0,
        units=round(units, 2),
        roi=round(units / bets, 4) if bets else 0.0,
        fav_pct=round(fav_bets / bets, 4) if bets else 0.0,
        dog_pct=round(dog_bets / bets, 4) if bets else 0.0,
        pickem=pickem,
    )


# ----------------- model fitters -----------------

XGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=7, n_jobs=4,
)


def xgb_predict(df: pd.DataFrame, feat: list[str], calibrate: bool, season: int) -> tuple[pd.DataFrame, np.ndarray]:
    tr = df[df["season"] < season]; te = df[df["season"] == season]
    X_tr, y_tr = tr[feat], tr["is_cover"]; X_te = te[feat]
    model = xgb.XGBClassifier(**XGB_KW)
    if not calibrate:
        model.fit(X_tr, y_tr)
        return te, model.predict_proba(X_te)[:, 1]
    n = len(X_tr); cut = int(n * 0.8)
    model.fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
    p_cal = model.predict_proba(X_tr.iloc[cut:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_tr.iloc[cut:])
    return te, iso.transform(model.predict_proba(X_te)[:, 1])


def elo_v2_predict(df_labels: pd.DataFrame, season: int, K=0.10, init_scale=1.0, h=0.05) -> tuple[pd.DataFrame, np.ndarray]:
    """Run matchup-aware Elo on `season`, returning per-row P(cover) using pre-week thetas."""
    prev = df_labels[df_labels["season"] == season - 1]
    rates = prev.groupby("team")["is_cover"].mean().to_dict() if not prev.empty else {}
    league = float(prev["is_cover"].mean()) if not prev.empty else 0.5
    theta: dict[str, float] = {}

    def init(t):
        return init_scale * logit(rates.get(t, league))

    sd = df_labels[df_labels["season"] == season].sort_values(["week", "team"]).reset_index(drop=True)
    probs = np.full(len(sd), np.nan)
    for w in sorted(sd["week"].unique()):
        wk = sd[sd["week"] == w]
        pairs = pair_games(wk)
        idx_map = {r["team"]: r.name for _, r in wk.iterrows()}
        # 1) record pre-week prob for every row
        team_opp = {}
        for a, b in pairs:
            team_opp[a["team"]] = b
            team_opp[b["team"]] = a
        for _, row in wk.iterrows():
            t = row["team"]
            if t not in theta:
                theta[t] = init(t)
            opp = team_opp.get(t)
            if opp is None:
                continue
            if opp["team"] not in theta:
                theta[opp["team"]] = init(opp["team"])
            p = sigmoid(theta[t] - theta[opp["team"]] + h * row["is_home"])
            probs[idx_map[t]] = p
        # 2) post-week updates
        for a, b in pairs:
            pa = sigmoid(theta[a["team"]] - theta[b["team"]] + h * a["is_home"])
            err = a["is_cover"] - pa
            theta[a["team"]] += K * err
            theta[b["team"]] -= K * err
    sd["_p"] = probs
    return sd, sd["_p"].fillna(0.5).values


def beta_v1_predict(df_labels: pd.DataFrame, season: int, lam=1.0, prior_strength=32) -> tuple[pd.DataFrame, np.ndarray]:
    prev = df_labels[df_labels["season"] == season - 1]
    rates = prev.groupby("team")["is_cover"].mean().to_dict() if not prev.empty else {}
    league = float(prev["is_cover"].mean()) if not prev.empty else 0.5
    state: dict[str, tuple[float, float]] = {}

    def init(t):
        p0 = rates.get(t, league)
        return (prior_strength * p0, prior_strength * (1 - p0))

    sd = df_labels[df_labels["season"] == season].sort_values(["week", "team"]).reset_index(drop=True)
    probs = np.full(len(sd), np.nan)
    for w in sorted(sd["week"].unique()):
        wk = sd[sd["week"] == w]
        for idx, row in wk.iterrows():
            t = row["team"]
            if t not in state:
                state[t] = init(t)
            a, b = state[t]
            probs[idx] = a / (a + b)
        for _, row in wk.iterrows():
            t = row["team"]; y = row["is_cover"]
            a, b = state[t]
            state[t] = (lam * a + y, lam * b + (1 - y))
    sd["_p"] = probs
    return sd, probs


# ----------------- driver -----------------

def run_xgb(label, parquet, feat_cols, calibrate):
    df = pd.read_parquet(parquet)
    per_season = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        te, p = xgb_predict(df, feat_cols, calibrate, s)
        res = per_game_pick_and_score(te, p)
        per_season.append({"season": s, **res})
    return label, pd.DataFrame(per_season)


def run_bayes(label, predictor):
    """predictor: callable(df_labels, season) -> (test_df, probs)."""
    df_labels = pd.concat([
        pd.read_csv(REPO_ROOT / "nfl_training_data.csv", usecols=["season", "week", "team", "spread", "is_home", "is_cover"]),
        pd.read_csv(REPO_ROOT / "nfl_current_data.csv", usecols=["season", "week", "team", "spread", "is_home", "is_cover"]),
    ])
    df_labels = df_labels.dropna(subset=["is_cover", "spread", "is_home"])
    df_labels["is_cover"] = df_labels["is_cover"].astype(int)
    df_labels["is_home"] = df_labels["is_home"].astype(int)
    per_season = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        te, p = predictor(df_labels, s)
        res = per_game_pick_and_score(te, p)
        per_season.append({"season": s, **res})
    return label, pd.DataFrame(per_season)


def aggregate(per: pd.DataFrame, label: str) -> dict:
    bets = per["bets"].sum(); wins = per["wins"].sum(); units = per["units"].sum()
    fav_bets = (per["bets"] * per["fav_pct"]).sum()
    return {
        "model": label,
        "games": per["games"].sum(),
        "pushes_excl": per["pushes_excluded"].sum(),
        "bets": bets,
        "win_pct": round(wins / bets, 4) if bets else 0.0,
        "units": round(units, 2),
        "roi": round(units / bets, 4) if bets else 0.0,
        "fav_pct": round(fav_bets / bets, 4) if bets else 0.0,
    }


def main():
    df_v1 = pd.read_parquet(FEATURES_V1)
    df_v2 = pd.read_parquet(FEATURES_V2)
    base51 = [c for c in df_v1.columns if c not in ID_COLS]
    base54 = [c for c in df_v2.columns if c not in ID_COLS]

    runs = []
    runs.append(run_xgb("XGB-A 51 feat, no cal",     FEATURES_V1, base51, False))
    runs.append(run_xgb("XGB-B 54 feat, no cal",     FEATURES_V2, base54, False))
    runs.append(run_xgb("XGB-C 51 + isotonic",       FEATURES_V1, base51, True))
    runs.append(run_xgb("XGB-D 54 + isotonic",       FEATURES_V2, base54, True))
    runs.append(run_bayes("ELO-V2 K=0.10, init=1.0, h=0.05",
                          lambda d, s: elo_v2_predict(d, s)))
    runs.append(run_bayes("BETA-V1 lambda=1.0, ps=32",
                          lambda d, s: beta_v1_predict(d, s)))

    aggs = [aggregate(per, label) for label, per in runs]
    agg_df = pd.DataFrame(aggs).sort_values("roi", ascending=False)
    agg_df.to_csv(ROOT / "rebaseline_scoreboard.csv", index=False)

    print("\n=== Re-baseline scoreboard (corrected accounting: per-game pick + push exclusion) ===")
    print(agg_df.to_string(index=False))

    # Per-season ROI table
    print("\n=== Per-season ROI ===")
    per_pivot = pd.concat(
        [per.assign(model=label) for label, per in runs], ignore_index=True
    ).pivot_table(index="season", columns="model", values="roi")
    print(per_pivot.to_string())

    # Per-season bet count
    print("\n=== Per-season bet count ===")
    bets_pivot = pd.concat(
        [per.assign(model=label) for label, per in runs], ignore_index=True
    ).pivot_table(index="season", columns="model", values="bets")
    print(bets_pivot.to_string())

    print("\nNote: 'fav_pct' is the share of bets the model placed on the spread favorite (spread<0).")
    print("Context: historical NFL underdogs cover ~50%, occasionally up to 60% in specific eras.")


if __name__ == "__main__":
    main()
