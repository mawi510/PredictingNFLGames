"""
Production training job for the margin model (model v2).

Reads the curated features parquet, trains an XGBRegressor with target = margin,
calibrates a Student-t residual distribution on the most recent season (OOS),
then refits on all data and persists a deploy-ready bundle.

Output bundle (model_margin_v2.pkl):
    {
        "regressor":      fitted xgboost.XGBRegressor,
        "feature_names":  ordered list of 44 feature names,
        "scale_t":        Student-t scale (residual scale on OOS holdout),
        "nu_t":           Student-t degrees of freedom on OOS holdout,
        "sigma_normal":   Normal-fit residual std (kept for parity / display),
        "version":        semver string (margin-v2.YYYY.MM),
        "trained_at":     ISO timestamp,
        "training_seasons": list[int],
        "calib_season":   int (the season held out for residual calibration),
        "metadata":       free-form (dropped features, training row count, etc.),
    }

The bundle is intentionally a dict, NOT the raw sklearn model. The API loader
needs to: (a) verify the bundle structure (version + feature contract),
(b) call regressor.predict() on a row in feature_names order, (c) apply
scipy.stats.t.cdf((mu + spread) / scale_t, df=nu_t) at inference time.

Run:
    python -m pipeline.train_margin
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
FEATURES_PARQUET = REPO / "research" / "xgboost" / "features.parquet"
MODEL_OUT = REPO / "api" / "model_margin_v2.pkl"
MANIFEST_OUT = REPO / "api" / "feature_manifest_margin_v2.json"

ID_COLS = ["season", "week", "team", "is_cover"]
# Market-derived features dropped so the model is forced to learn team-strength signal.
# spread itself is also dropped from features and used only in the final
# Phi((mu + spread)/sigma) cover-probability calculation.
DROP_FEATS = [
    "spread", "spread_odds", "moneyline", "exp_total_points",
    "over_odds", "under_odds", "is_fav",
]

XGB_REG_KW = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="reg:squarederror", eval_metric="rmse",
    tree_method="hist", random_state=7, n_jobs=4,
)


def load_margins() -> pd.DataFrame:
    """Pull actual game margins from nfl_data_py schedules (training target)."""
    import nfl_data_py as nfl

    sched = nfl.import_schedules(list(range(2018, datetime.now(timezone.utc).year + 1)))
    sched = sched.dropna(subset=["home_score", "away_score"])
    home = sched.rename(columns={"home_team": "team"})
    home["margin"] = home["home_score"] - home["away_score"]
    away = sched.rename(columns={"away_team": "team"})
    away["margin"] = away["away_score"] - away["home_score"]
    out = pd.concat(
        [home[["season", "week", "team", "margin"]],
         away[["season", "week", "team", "margin"]]],
        ignore_index=True,
    )
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def main() -> None:
    feat_df = pd.read_parquet(FEATURES_PARQUET)
    df = feat_df.merge(load_margins(), on=["season", "week", "team"], how="inner")
    feat = [c for c in df.columns
            if c not in ID_COLS + ["margin"] and c not in DROP_FEATS]
    print(f"Training rows: {len(df):,}  | features: {len(feat)}")

    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        raise RuntimeError("Need at least 2 seasons of data to calibrate residuals.")
    calib_season = seasons[-1]
    training_seasons = seasons

    # 1) OOS residual calibration: fit on everything before the calib season,
    #    score on the calib season, fit Student-t to those residuals.
    cal_train = df[df["season"] < calib_season]
    cal_test = df[df["season"] == calib_season]
    m_cal = xgb.XGBRegressor(**XGB_REG_KW)
    m_cal.fit(cal_train[feat], cal_train["margin"])
    resid = cal_test["margin"].values - m_cal.predict(cal_test[feat])
    nu, _, scale = stats.t.fit(resid, floc=0)
    nu = float(np.clip(nu, 3.0, 50.0))
    sigma_normal = float(np.std(resid))
    print(f"OOS calibration on {calib_season}: nu={nu:.3f}, scale={scale:.3f}, "
          f"sigma_normal={sigma_normal:.3f}, n_resid={len(resid)}")

    # 2) Final fit on ALL data (the model the API will serve)
    final = xgb.XGBRegressor(**XGB_REG_KW)
    final.fit(df[feat], df["margin"])
    print(f"Final model trained on {len(df):,} rows across seasons {training_seasons}")

    # 3) Bundle + persist
    version = f"margin-v2.{datetime.now(timezone.utc).strftime('%Y.%m')}"
    bundle = {
        "regressor": final,
        "feature_names": feat,
        "scale_t": float(scale),
        "nu_t": float(nu),
        "sigma_normal": float(sigma_normal),
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_seasons": [int(s) for s in training_seasons],
        "calib_season": int(calib_season),
        "metadata": {
            "dropped_features": DROP_FEATS,
            "n_training_rows": int(len(df)),
            "n_calib_resid": int(len(resid)),
            "xgb_params": XGB_REG_KW,
        },
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUT)
    print(f"Wrote bundle -> {MODEL_OUT}  ({MODEL_OUT.stat().st_size/1024:.1f} KB)")

    # 4) Manifest: input contract for the API
    manifest = {
        "version": version,
        "model_type": "margin_regressor_xgb",
        "description": (
            "Margin model v2. Regressor predicts E[team_score - opp_score]; "
            "cover probability is computed at inference via "
            "Student-t CDF((mu_hat + spread) / scale_t, df=nu_t)."
        ),
        "n_features": len(feat),
        "features": feat,
        "dropped_features": DROP_FEATS,
        "spread_is_external": True,
        "training_seasons": [int(s) for s in training_seasons],
        "calib_season": int(calib_season),
        "scale_t": float(scale),
        "nu_t": float(nu),
        "sigma_normal": float(sigma_normal),
        "trained_at": bundle["trained_at"],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
