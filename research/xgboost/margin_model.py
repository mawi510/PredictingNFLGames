"""
Margin-based cover prediction (the principled reformulation).

Instead of training a classifier to predict P(cover) directly, train a regressor
to predict the team's expected point margin (mu), then derive
    P(cover | spread=s) = Phi((mu + s) / sigma)
where sigma is the residual std on the training set and Phi is the Normal CDF.

This guarantees:
  - smooth, monotonic-increasing P(cover) vs spread (the slider problem solved by construction)
  - interpretability: mu + s = model's edge in points
  - spread CANNOT be ignored -- it enters analytically in the last step

Pipeline (per target season, walk-forward):
  1. Train XGBRegressor on prior seasons, target = margin = team_score - opp_score.
     Features = team-strength features only. NO spread, NO market features.
  2. Compute sigma = std of training residuals (constant for now).
  3. For each test team-week: mu_hat = regressor.predict(row); p_cover = Phi((mu_hat + spread) / sigma).
  4. Score with the same pair-and-pick rule as the classifier (bet every non-push game).

Diagnostics:
  - Residual histogram, mean, std, kurtosis (is Normal OK?)
  - QQ vs Normal -- if tails are heavy, swap to Student-t
  - Spread sensitivity (should now be a smooth S-curve)
  - Per-season ROI
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features.parquet"
REPO = ROOT.parents[1]
TRAIN_CSV = REPO / "nfl_training_data.csv"
CUR_CSV = REPO / "nfl_current_data.csv"

UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
ID_COLS = ["season", "week", "team", "is_cover"]
DROP_FEATS = ["spread", "spread_odds", "moneyline", "exp_total_points",
              "over_odds", "under_odds", "is_fav"]

XGB_REG_KW = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="reg:squarederror", eval_metric="rmse",
    tree_method="hist", random_state=7, n_jobs=4,
)


def load_margin_target() -> pd.DataFrame:
    """Build (season, week, team) -> margin from nfl_data_py schedules."""
    import nfl_data_py as nfl
    sched = nfl.import_schedules(list(range(2018, 2025)))
    sched = sched.dropna(subset=["home_score", "away_score"])
    sched = sched[["season", "week", "home_team", "away_team",
                   "home_score", "away_score"]].copy()
    # Long-format: one row per team-game with that team's margin
    home = sched.rename(columns={"home_team": "team"})
    home["margin"] = home["home_score"] - home["away_score"]
    away = sched.rename(columns={"away_team": "team"})
    away["margin"] = away["away_score"] - away["home_score"]
    out = pd.concat([
        home[["season", "week", "team", "margin"]],
        away[["season", "week", "team", "margin"]],
    ], ignore_index=True)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def fit_predict_margin(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Walk-forward train a margin regressor with OUT-OF-SAMPLE residual calibration.

    For target season s with training seasons T = [t1, ..., t_n]:
      1. Hold out the most recent training season t_n as a calibration set.
      2. Train a "calibration model" on T \\ {t_n} and predict on t_n -- these
         residuals are honest forward-error estimates (not train-fit residuals).
      3. Fit Student-t(nu, scale) AND record sigma_normal on those OOS residuals.
      4. Refit the FINAL model on the full T and use it for predictions on s.

    Falls back to in-sample residuals only if there is just one training season
    (which only happens for the 2019 fold here).
    """
    parts = []
    fold_params = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        train_seasons = sorted(df.loc[df["season"] < s, "season"].unique())
        tr_full = df[df["season"] < s]
        te = df[df["season"] == s].copy()

        # --- OOS calibration: train on T-minus-most-recent, score on most-recent ---
        if len(train_seasons) >= 2:
            calib_season = train_seasons[-1]
            tr_calib = df[df["season"] < calib_season]
            cv = df[df["season"] == calib_season]
            m_cal = xgb.XGBRegressor(**XGB_REG_KW)
            m_cal.fit(tr_calib[feat], tr_calib["margin"])
            resid = cv["margin"].values - m_cal.predict(cv[feat])
            resid_source = f"OOS on {calib_season}"
        else:
            # Cold-start: only one prior season, fall back to in-sample
            m_cal = xgb.XGBRegressor(**XGB_REG_KW)
            m_cal.fit(tr_full[feat], tr_full["margin"])
            resid = tr_full["margin"].values - m_cal.predict(tr_full[feat])
            resid_source = "in-sample (cold start)"

        nu, _, scale = stats.t.fit(resid, floc=0)
        nu = float(np.clip(nu, 3.0, 50.0))
        sigma_normal = float(np.std(resid))

        # --- Final model on all training seasons ---
        m_final = xgb.XGBRegressor(**XGB_REG_KW)
        m_final.fit(tr_full[feat], tr_full["margin"])
        te["mu_hat"] = m_final.predict(te[feat])
        te["sigma_hat"] = float(scale)
        te["nu_hat"] = nu
        te["sigma_normal"] = sigma_normal
        fold_params.append((s, resid_source, len(resid), nu, float(scale), sigma_normal))
        parts.append(te)

    print("\nPer-fold residual-distribution fits (OUT-OF-SAMPLE):")
    print(pd.DataFrame(fold_params,
        columns=["season", "resid_source", "n_resid", "nu_t", "scale_t", "sigma_normal"]
    ).to_string(index=False))
    return pd.concat(parts, ignore_index=True)


def p_cover_normal(mu, sigma, spread):
    """P(margin > -spread) under Normal(mu, sigma^2)."""
    return stats.norm.cdf((mu + spread) / sigma)


def p_cover_t(mu, scale, nu, spread):
    """P(margin > -spread) under Student-t with location=mu, scale=scale, df=nu."""
    return stats.t.cdf((mu + spread) / scale, df=nu)


def pair_and_score(df: pd.DataFrame) -> dict:
    bets = wins = pushes = 0; units = 0.0
    per_season: dict[int, list[int]] = {}
    for (s, w), wk in df.groupby(["season", "week"], sort=False):
        wk = wk.reset_index(drop=True); used = set()
        for i in range(len(wk)):
            if i in used: continue
            ri = wk.iloc[i]
            cand = [j for j in range(len(wk)) if j != i and j not in used
                    and np.isclose(wk.iloc[j]["spread"], -ri["spread"])
                    and wk.iloc[j]["is_home"] == 1 - ri["is_home"]]
            if not cand: continue
            j = cand[0]; used.add(i); used.add(j); rj = wk.iloc[j]
            pa, pb = ri["p_cover"], rj["p_cover"]
            ya, yb = int(ri["is_cover"]), int(rj["is_cover"])
            if ya == 0 and yb == 0:
                pushes += 1; continue
            pick_y = ya if pa >= pb else yb
            bets += 1
            wb = per_season.setdefault(int(s), [0, 0])
            wb[1] += 1
            if pick_y == 1:
                wins += 1; units += UNIT_PROFIT_WIN; wb[0] += 1
            else:
                units -= UNIT_LOSS
    return dict(
        bets=bets, pushes=pushes, wins=wins,
        win_pct=round(wins/bets, 4) if bets else 0,
        units=round(units, 2),
        roi=round(units/bets, 4) if bets else 0,
        per_season={s: {"wins": wb[0], "bets": wb[1],
                        "win_pct": round(wb[0]/wb[1], 4) if wb[1] else 0,
                        "roi": round((wb[0]*UNIT_PROFIT_WIN - (wb[1]-wb[0])*UNIT_LOSS)/wb[1], 4) if wb[1] else 0}
                    for s, wb in sorted(per_season.items())},
    )


def main():
    feat_df = pd.read_parquet(FEATURES)
    print(f"Loaded features.parquet: {len(feat_df):,} rows x {len(feat_df.columns)} cols")
    # Attach margin target
    margins = load_margin_target()
    print(f"Loaded margin target: {len(margins):,} (season, week, team) rows")
    df = feat_df.merge(margins, on=["season", "week", "team"], how="inner")
    print(f"After merge: {len(df):,} rows (dropped {len(feat_df)-len(df)} unmatched)")

    feat = [c for c in df.columns if c not in ID_COLS + ["margin"] and c not in DROP_FEATS]
    print(f"Margin-model feature set: {len(feat)} features (dropped {DROP_FEATS})")

    # 1. Walk-forward train + predict
    preds = fit_predict_margin(df, feat)

    # 2. Residual diagnostics on the 2024 training fold (gives a current-era read)
    print("\n=== Residual diagnostics (Normal assumption check) ===")
    tr2024 = df[df["season"] < 2024]
    m = xgb.XGBRegressor(**XGB_REG_KW); m.fit(tr2024[feat], tr2024["margin"])
    resid = tr2024["margin"].values - m.predict(tr2024[feat])
    print(f"Train residuals: n={len(resid):,}, mean={resid.mean():+.3f}, "
          f"std={resid.std():.3f}, skew={stats.skew(resid):+.3f}, "
          f"excess_kurtosis={stats.kurtosis(resid):+.3f}")
    # Shapiro is unreliable at large n; report Anderson-Darling stat instead
    ad = stats.anderson(resid, dist="norm")
    print(f"Anderson-Darling stat: {ad.statistic:.3f}  "
          f"(critical at 5% = {ad.critical_values[2]:.3f}; "
          f"{'NOT normal' if ad.statistic > ad.critical_values[2] else 'normal-ish'})")
    # Heavy-tail check: empirical vs Normal at 2 / 3 sigma
    s = resid.std()
    for k in [2, 3]:
        emp = float(np.mean(np.abs(resid) > k * s))
        nrm = 2 * (1 - stats.norm.cdf(k))
        print(f"  |resid|>{k}sigma: empirical={emp*100:.2f}%, normal={nrm*100:.2f}%  "
              f"({'fat tail' if emp > 1.5*nrm else 'ok'})")

    # 3. Cover probability + backtest -- Normal AND Student-t
    preds_n = preds.copy()
    preds_n["p_cover"] = p_cover_normal(
        preds_n["mu_hat"].values, preds_n["sigma_normal"].values, preds_n["spread"].values
    )
    preds_t = preds.copy()
    preds_t["p_cover"] = p_cover_t(
        preds_t["mu_hat"].values, preds_t["sigma_hat"].values,
        preds_t["nu_hat"].values, preds_t["spread"].values
    )
    print(f"\n=== Backtest comparison (bet every non-push pick) ===")
    for label, p in [("Normal", preds_n), ("Student-t", preds_t)]:
        r = pair_and_score(p)
        print(f"\n[{label}] bets={r['bets']}, wins={r['wins']}, pushes={r['pushes']}, "
              f"win%={r['win_pct']*100:.2f}%, ROI={r['roi']*100:+.2f}%, units={r['units']}")
        print(pd.DataFrame.from_dict(r["per_season"], orient="index").to_string())

    # NOTE: sigma cancels when comparing two sides of the same game, so picks may be
    # identical under Normal vs Student-t. The difference shows up in stated P(cover),
    # not the pick. Quantify how often the pick disagrees.
    pick_match = (preds_n["p_cover"] >= 0.5).astype(int) == (preds_t["p_cover"] >= 0.5).astype(int)
    print(f"\nPick agreement (p>=0.5) between Normal and Student-t: "
          f"{pick_match.mean()*100:.2f}%  ({(~pick_match).sum()} disagreements out of {len(preds_n)})")

    # 4. Spread sensitivity -- Student-t slider behavior
    print("\n=== Spread sensitivity (2024 fold, Student-t) ===")
    s24 = preds[preds["season"] == 2024].copy()
    scale24 = float(s24["sigma_hat"].iloc[0])
    nu24 = float(s24["nu_hat"].iloc[0])
    print(f"2024 fold params: scale={scale24:.3f}, nu={nu24:.2f}")
    grid = np.arange(-14, 14.5, 0.5)
    ranges = []
    mono_inc = 0
    for _, row in s24.iterrows():
        curve = p_cover_t(np.full_like(grid, row["mu_hat"]), scale24, nu24, grid)
        ranges.append(curve.max() - curve.min())
        if (np.diff(curve) >= -1e-9).all(): mono_inc += 1
    print(f"Curves: {len(s24)}.  Avg P(cover) range across [-14,+14]: {np.mean(ranges):.3f}")
    print(f"Monotonic increasing: {mono_inc}/{len(s24)} ({mono_inc/len(s24)*100:.1f}%)")
    for key in [-7, -3, 0, 3, 7]:
        below = p_cover_t(s24["mu_hat"].values, scale24, nu24, np.full(len(s24), key - 1.0))
        above = p_cover_t(s24["mu_hat"].values, scale24, nu24, np.full(len(s24), key + 1.0))
        print(f"  spread={key:+d}: dP/d(2pt) = {(above - below).mean():+.4f}")

    # Tail comparison Normal vs Student-t at the same example games
    print(f"\nExample curves (3 random 2024 games, Normal vs Student-t):")
    sigma_n24 = float(s24["sigma_normal"].iloc[0])
    for _, row in s24.sample(3, random_state=11).iterrows():
        cn = p_cover_normal(np.full_like(grid, row["mu_hat"]), sigma_n24, grid)
        ct = p_cover_t(np.full_like(grid, row["mu_hat"]), scale24, nu24, grid)
        print(f"\n{row['team']} wk{int(row['week'])} "
              f"(spread={row['spread']:+.1f}, mu_hat={row['mu_hat']:+.2f}, "
              f"edge_pts={row['mu_hat']+row['spread']:+.2f}):")
        print(f"  {'spread':>7s}  {'P(Normal)':>9s}  {'P(t)':>6s}  diff")
        for sp, pn, pt in zip(grid[::4], cn[::4], ct[::4]):
            mk = " <-- actual" if abs(sp - row["spread"]) < 0.25 else ""
            print(f"  {sp:+7.1f}    {pn:>6.3f}    {pt:>6.3f}   {pt-pn:+.3f}{mk}")


if __name__ == "__main__":
    main()
