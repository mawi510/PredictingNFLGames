"""
Production training job for the margin model (v3).

Model: ridge regression on standardized home-minus-away feature differences,
predicting the home team's final margin. Chosen in research/margin_v3 by a
rule fixed in advance (lowest out-of-sample margin RMSE, walk-forward
2019-2025) over the v2 form (one XGBoost row per team), whose line was about
half as wide as the market's.

Steps:
  1. Build one row per game from the curated team-week features.
  2. Walk forward (train on seasons < s, predict s) to get out-of-sample lines.
     From those, fit the cover-probability scale on actual cover outcomes
     (weeks >= MIN_WEEK, pushes excluded) and record the margin residual sd.
  3. Refit on every season and write the bundle.

Output: api/model_margin_v3.json, a plain JSON bundle the API reads directly
(feature order, coefficients, feature sds and means, scale_t, nu_t, version).

Run:
    python -m pipeline.train_margin
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.margin_model import (  # noqa: E402
    DEFAULT_BUNDLE_PATH,
    attach_opponent,
    game_margins,
    load_schedule,
)

FEATURES_PARQUET = REPO / "research" / "xgboost" / "features_2025.parquet"

ID_COLS = ["season", "week", "team", "is_cover", "opponent"]
# Market-derived columns are never model inputs. The spread is used only to
# turn a predicted margin into a cover probability.
DROP_FEATS = [
    "spread", "spread_odds", "moneyline", "exp_total_points",
    "over_odds", "under_odds", "is_fav",
]
ALPHAS = np.logspace(0, 4, 13)
NU_T = 50.0          # near-normal tails; kept as a parameter of the cover formula
MIN_WEEK = 4         # the site publishes picks from week 4


def load_games() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """One row per played game with home/away feature blocks.

    Returns (games, feature names, team-level feature means).
    """
    df = attach_opponent(pd.read_parquet(FEATURES_PARQUET))
    feats = [c for c in df.columns if c not in ID_COLS and c not in DROP_FEATS]
    sched = load_schedule(df["season"].unique())[["season", "week", "team", "margin"]]
    df = df.merge(sched, on=["season", "week", "team"], how="inner").dropna(subset=["margin"])
    home, away = df[df["is_home"] == 1], df[df["is_home"] == 0]
    g = home.merge(
        away, left_on=["season", "week", "opponent"], right_on=["season", "week", "team"],
        suffixes=("_h", "_a"),
    )
    games = pd.DataFrame({
        "season": g["season"].astype(int), "week": g["week"].astype(int),
        "spread_home": g["spread_h"].astype(float), "margin_home": g["margin_h"].astype(float),
    })
    games["_h"] = list(g[[f + "_h" for f in feats]].to_numpy(dtype=float))
    games["_a"] = list(g[[f + "_a" for f in feats]].to_numpy(dtype=float))
    means = np.nanmean(df[feats].to_numpy(dtype=float), axis=0)
    return games, feats, means


def fit_ridge(games: pd.DataFrame) -> dict:
    """Fit on both orientations of every game so the model is antisymmetric."""
    D = np.stack(games["_h"]) - np.stack(games["_a"])
    y = games["margin_home"].to_numpy()
    sd = np.nanstd(np.vstack([D, -D]), axis=0)
    sd[sd == 0] = 1.0
    Z = np.nan_to_num(D / sd)
    r = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(np.vstack([Z, -Z]), np.concatenate([y, -y]))
    return {"coef": r.coef_.tolist(), "feature_sd": sd.tolist(), "alpha": float(r.alpha_)}


def out_of_sample_lines(games: pd.DataFrame) -> pd.DataFrame:
    """Walk forward: each season's lines come from a fit on earlier seasons."""
    seasons = sorted(games["season"].unique())
    parts = []
    for s in seasons[1:]:
        fit = fit_ridge(games[games["season"] < s])
        test = games[games["season"] == s].copy()
        test["line"] = game_margins(np.stack(test["_h"]), np.stack(test["_a"]), fit)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def fit_cover_scale(oos: pd.DataFrame) -> tuple[float, int]:
    """Scale s that maximizes the likelihood of the actual cover outcomes under
    P(home covers) = t.cdf((line + spread) / s)."""
    d = oos[oos["week"] >= MIN_WEEK]
    res = np.sign(d["margin_home"] + d["spread_home"]).to_numpy()
    gap = (d["line"] + d["spread_home"]).to_numpy()[res != 0]
    y = (res[res != 0] > 0).astype(float)

    def nll(s: float) -> float:
        p = np.clip(stats.t.cdf(gap / s, NU_T), 1e-9, 1 - 1e-9)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    best = optimize.minimize_scalar(nll, bounds=(5.0, 400.0), method="bounded")
    print(f"Cover scale {best.x:.1f} on {len(y):,} out-of-sample games "
          f"(log loss {best.fun:.4f}; coin flip {np.log(2):.4f})")
    return float(best.x), int(len(y))


def main() -> None:
    games, feats, means = load_games()
    seasons = sorted(int(s) for s in games["season"].unique())
    print(f"{len(games):,} games, seasons {seasons[0]}-{seasons[-1]}, {len(feats)} features")

    oos = out_of_sample_lines(games)
    resid = oos["margin_home"] - oos["line"]
    market_rmse = float(np.sqrt(np.mean((oos["margin_home"] + oos["spread_home"]) ** 2)))
    print(f"Out-of-sample {int(oos.season.min())}-{int(oos.season.max())}: RMSE {np.sqrt(np.mean(resid**2)):.2f} "
          f"(market {market_rmse:.2f}), line sd {oos['line'].std():.2f} "
          f"(market {oos['spread_home'].std():.2f})")
    scale, n_cal = fit_cover_scale(oos)

    fit = fit_ridge(games)
    now = datetime.now(timezone.utc)
    bundle = {
        "version": f"margin-v3.{now.strftime('%Y.%m')}",
        "model_type": "ridge_feature_diff",
        "description": (
            "Margin model v3. Ridge regression on standardized home-minus-away feature "
            "differences predicts the home margin; a feature missing for either team "
            "contributes 0. Cover probability = t.cdf((margin + spread) / scale_t, nu_t), "
            "with scale_t fitted on out-of-sample cover outcomes."
        ),
        "features": feats,
        "n_features": len(feats),
        "coef": fit["coef"],
        "feature_sd": fit["feature_sd"],
        "feature_mean": means.tolist(),
        "alpha": fit["alpha"],
        "scale_t": scale,
        "nu_t": NU_T,
        "margin_resid_sd": float(np.std(resid)),
        "dropped_features": DROP_FEATS,
        "spread_is_external": True,
        "training_seasons": seasons,
        "calibration": {
            "source": f"walk-forward out-of-sample lines, seasons {seasons[1]}-{seasons[-1]}, "
                      f"weeks {MIN_WEEK}+, pushes excluded",
            "n_games": n_cal,
        },
        "trained_at": now.isoformat(),
    }
    DEFAULT_BUNDLE_PATH.write_text(json.dumps(bundle, indent=1))
    print(f"Wrote {DEFAULT_BUNDLE_PATH}  (alpha {fit['alpha']:.0f}, version {bundle['version']})")
    top = sorted(zip(feats, fit["coef"]), key=lambda t: -abs(t[1]))[:8]
    print("Largest coefficients (points per sd of difference):")
    for name, c in top:
        print(f"  {c:+.2f}  {name}")


if __name__ == "__main__":
    main()
