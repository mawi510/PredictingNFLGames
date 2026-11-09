"""
Issue #25: the production margin model's line is about half as wide as the
market's, so its picks are mechanically underdogs. This compares ways to fix
the line's scale in a strict walk-forward (train on seasons < s, predict s).

Candidates (all without market inputs):
  team        current production form: one XGBoost row per team (own features),
              game line = (mu_home - mu_away) / 2
  team+opp    same, with the opponent's features appended to each row
  diff_xgb    one XGBoost row per game on home-minus-away feature differences,
              trained on both orientations, prediction antisymmetrized
  diff_ridge  ridge regression on the same standardized differences
  *_cal       the same line times a slope fitted on EARLIER out-of-sample
              seasons only (nested; the first test season uses slope 1)

Selection rule, fixed before looking at results: lowest pooled out-of-sample
RMSE of the margin on 2019-2025, slope nearest 1 as the tie-break. The
against-the-spread record is reported but NOT used to choose: a season of
picks is too noisy to select on. 2026 is reported separately.

Run:  python research/margin_v3/walkforward.py
Out:  research/margin_v3/walkforward_summary.csv, walkforward_games.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pipeline.margin_model import (  # noqa: E402
    attach_matchup_features,
    attach_opponent,
    load_schedule,
)

warnings.filterwarnings("ignore")

_PARQUET = REPO / "research" / "xgboost" / "features_2025.parquet"
_NOT_FEATURES = {"season", "week", "team", "is_cover", "spread", "spread_odds", "moneyline",
                 "exp_total_points", "over_odds", "under_odds", "is_fav"}
FEATS = [c for c in pd.read_parquet(_PARQUET).columns if c not in _NOT_FEATURES]
KEYS = ["season", "week", "team"]
XGB_KW = dict(  # production hyperparameters, unchanged
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="reg:squarederror", tree_method="hist", random_state=7, n_jobs=4,
)
TEST_SEASONS = list(range(2019, 2027))


def load_team_rows() -> pd.DataFrame:
    """One row per team-game, 2018 to date, with opponent, home flag and margin."""
    hist = attach_opponent(pd.read_parquet(_PARQUET))
    cur = pd.read_csv(REPO / "nfl_current_data.csv", low_memory=False)
    cur.columns = cur.columns.str.lower()
    cur = attach_matchup_features(cur)
    cols = KEYS + ["opponent", "spread"] + FEATS
    df = pd.concat([hist[cols], cur[cols]], ignore_index=True)
    sched = load_schedule(df["season"].unique())[KEYS + ["is_home", "margin"]]
    df = df.merge(sched.rename(columns={"is_home": "sched_home"}), on=KEYS, how="left")
    return df.dropna(subset=["sched_home"]).reset_index(drop=True)


def build_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home and away feature blocks side by side."""
    home = df[df.sched_home == 1]
    away = df[df.sched_home == 0]
    g = home.merge(
        away, left_on=["season", "week", "opponent"], right_on=KEYS, suffixes=("_h", "_a")
    )
    out = pd.DataFrame({
        "season": g["season"], "week": g["week"],
        "home": g["team_h"], "away": g["team_a"],
        "spread_home": g["spread_h"], "margin_home": g["margin_h"],
    })
    out["_h"] = list(g[[f + "_h" for f in FEATS]].to_numpy(dtype=float))
    out["_a"] = list(g[[f + "_a" for f in FEATS]].to_numpy(dtype=float))
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit every candidate on `train` games, return home-margin lines for `test`."""
    Fh, Fa = np.stack(train["_h"]), np.stack(train["_a"])
    y = train["margin_home"].to_numpy(dtype=float)
    Th, Ta = np.stack(test["_h"]), np.stack(test["_a"])
    lines = {}

    # team: each side from its own features
    m = xgb.XGBRegressor(**XGB_KW).fit(np.vstack([Fh, Fa]), np.concatenate([y, -y]))
    lines["team"] = (m.predict(Th) - m.predict(Ta)) / 2

    # team+opp: each side sees the opponent's features too
    m = xgb.XGBRegressor(**XGB_KW).fit(
        np.vstack([np.hstack([Fh, Fa]), np.hstack([Fa, Fh])]), np.concatenate([y, -y])
    )
    lines["team+opp"] = (m.predict(np.hstack([Th, Ta])) - m.predict(np.hstack([Ta, Th]))) / 2

    # diff_xgb: one row per game on feature differences, both orientations
    D, Dt = Fh - Fa, Th - Ta
    m = xgb.XGBRegressor(**XGB_KW).fit(np.vstack([D, -D]), np.concatenate([y, -y]))
    lines["diff_xgb"] = (m.predict(Dt) - m.predict(-Dt)) / 2

    # diff_ridge: linear on standardized differences; missing -> 0 (no information)
    sd = np.nanstd(np.vstack([D, -D]), axis=0)
    sd[sd == 0] = 1.0
    Z, Zt = np.nan_to_num(D / sd), np.nan_to_num(Dt / sd)
    r = RidgeCV(alphas=np.logspace(0, 4, 13), fit_intercept=False).fit(
        np.vstack([Z, -Z]), np.concatenate([y, -y])
    )
    lines["diff_ridge"] = r.predict(Zt)
    return lines


def slope_of(line: np.ndarray, margin: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(line)), line])
    return float(np.linalg.lstsq(X, margin, rcond=None)[0][1])


def metrics(g: pd.DataFrame, col: str) -> dict:
    line, margin, spread = g[col].to_numpy(), g["margin_home"].to_numpy(), g["spread_home"].to_numpy()
    res = np.sign(margin + spread)            # +1 home covers
    edge = line + spread                       # >0: model takes the home side
    pick_home = edge > 0
    ok = (res != 0) & (edge != 0)
    win = np.where(pick_home, res > 0, res < 0)[ok]
    has_fav = ok & (spread != 0)
    dog_pick = np.where(pick_home, spread > 0, spread < 0)[has_fav]
    return {
        "n": len(g),
        "rmse": float(np.sqrt(np.mean((margin - line) ** 2))),
        "rmse_market": float(np.sqrt(np.mean((margin + spread) ** 2))),
        "slope": slope_of(line, margin),
        "line_sd": float(np.std(line)),
        "market_sd": float(np.std(-spread)),
        "wins": int(win.sum()), "losses": int((~win).sum()),
        "win_pct": float(win.mean()),
        "dog_share": float(dog_pick.mean()),
        "inside_spread": float(np.mean(np.abs(line) < np.abs(spread))),
    }


def main() -> None:
    games = build_games(load_team_rows())
    played = games.dropna(subset=["margin_home"]).reset_index(drop=True)
    print(f"{len(played):,} played games, {played.season.min()}-{played.season.max()}")

    parts = []
    for s in TEST_SEASONS:
        train, test = played[played.season < s], played[played.season == s].copy()
        if test.empty:
            continue
        for name, line in fit_predict(train, test).items():
            test[name] = line
        parts.append(test)
        print(f"  season {s}: trained on {len(train):,}, predicted {len(test):,}")
    oos = pd.concat(parts, ignore_index=True)
    base = [c for c in ("team", "team+opp", "diff_xgb", "diff_ridge")]

    # Nested slope calibration: slope for season s from OOS seasons before s.
    for name in base:
        cal = np.empty(len(oos))
        for s in sorted(oos.season.unique()):
            prior = oos[oos.season < s]
            k = slope_of(prior[name].to_numpy(), prior["margin_home"].to_numpy()) if len(prior) else 1.0
            cal[(oos.season == s).to_numpy()] = k
        oos[name + "_cal"] = oos[name] * cal
        oos[name + "_k"] = cal

    models = base + [b + "_cal" for b in base]
    pooled = oos[oos.season <= 2025]
    rows = []
    for scope, sub in (("pooled 2019-2025", pooled), ("2026 to date", oos[oos.season == 2026])):
        for name in models:
            rows.append({"scope": scope, "model": name, **metrics(sub, name)})
    for s in sorted(oos.season.unique()):
        for name in models:
            rows.append({"scope": f"season {s}", "model": name, **metrics(oos[oos.season == s], name)})
    summary = pd.DataFrame(rows)
    summary.to_csv(HERE / "walkforward_summary.csv", index=False)
    oos.drop(columns=["_h", "_a"]).to_csv(HERE / "walkforward_games.csv", index=False)

    show = ["model", "rmse", "slope", "line_sd", "wins", "losses", "win_pct", "dog_share", "inside_spread"]
    for scope in ("pooled 2019-2025", "2026 to date"):
        sub = summary[summary.scope == scope]
        print(f"\n== {scope}  (n={int(sub.n.iloc[0])}, market RMSE {sub.rmse_market.iloc[0]:.2f}, "
              f"market line sd {sub.market_sd.iloc[0]:.2f})")
        print(sub[show].round(3).to_string(index=False))

    by_season = summary[summary.scope.str.startswith("season")]
    for value in ("rmse", "slope"):
        piv = by_season.pivot(index="scope", columns="model", values=value)[models]
        if value == "rmse":
            piv.insert(0, "market", by_season.groupby("scope").rmse_market.first())
        print(f"\n== {value} by season")
        print(piv.round(2).to_string())
    print("\n== slopes used by the calibrated versions (fitted on earlier OOS seasons)")
    print(oos.groupby("season")[[b + "_k" for b in base]].first().round(2).to_string())

    # Early weeks have few rolling features; the site only publishes picks from week 4.
    print("\n== RMSE by part of season, pooled 2019-2025")
    for label, sub in (("weeks 1-3", pooled[pooled.week <= 3]), ("weeks 4+", pooled[pooled.week >= 4])):
        m = {name: metrics(sub, name)["rmse"] for name in ("team", "diff_ridge")}
        print(f"  {label}: n={len(sub)} market {metrics(sub, 'team')['rmse_market']:.2f} "
              f"team {m['team']:.2f} diff_ridge {m['diff_ridge']:.2f}")

    # Are the cover probabilities honest? Normal error with the sd of EARLIER
    # out-of-sample residuals (13.5 for the first season), picks from week 4.
    from scipy import stats
    print("\n== cover probability calibration, weeks 4+ (pick probability bucket -> hit rate)")
    for name in ("team", "diff_ridge"):
        p_home = np.empty(len(oos))
        for s in sorted(oos.season.unique()):
            prior = oos[oos.season < s]
            sd = float(np.std(prior["margin_home"] - prior[name])) if len(prior) else 13.5
            idx = (oos.season == s).to_numpy()
            p_home[idx] = stats.norm.cdf((oos.loc[idx, name] + oos.loc[idx, "spread_home"]) / sd)
        res = np.sign(oos["margin_home"] + oos["spread_home"]).to_numpy()
        p_pick = np.maximum(p_home, 1 - p_home)
        win = np.where(p_home >= 0.5, res > 0, res < 0)
        for scope, mask in (("2019-2025", (oos.season <= 2025)), ("2026", (oos.season == 2026))):
            ok = mask.to_numpy() & (res != 0) & (oos.week >= 4).to_numpy()
            cells = []
            for lo, hi in ((0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)):
                b = ok & (p_pick >= lo) & (p_pick < hi)
                cells.append(f"{lo:.2f}-{min(hi, 1):.2f}: {win[b].mean():.3f} (n={b.sum()})" if b.sum() else f"{lo:.2f}+: none")
            print(f"  {name:<10} {scope}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
