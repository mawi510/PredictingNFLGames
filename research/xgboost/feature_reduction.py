"""
Feature reduction analysis on the XGB-A feature set (51 features).

X. PCA cumulative explained variance -- how many components capture 70/80/90/95%?
Y. XGB feature importance -- which features carry the predictive signal?
Z. Backtest: XGB-A with top-K most important features at K in {10, 15, 20, 25, 30, 51}.
   Tests whether a reduced feature set holds ROI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features.parquet"

UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
ID_COLS = ["season", "week", "team", "is_cover"]

XGB_KW = dict(
    n_estimators=200, max_depth=5, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=7, n_jobs=4,
)


def fit_predict_xgb(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    parts = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        tr = df[df["season"] < s]; te = df[df["season"] == s].copy()
        m = xgb.XGBClassifier(**XGB_KW)
        m.fit(tr[feat], tr["is_cover"])
        te["_p"] = m.predict_proba(te[feat])[:, 1]
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def pair_and_score(predicted: pd.DataFrame) -> dict:
    bets = wins = 0; units = 0.0
    for (s, w), wk in predicted.groupby(["season", "week"], sort=False):
        wk = wk.reset_index(drop=True); used = set()
        for i in range(len(wk)):
            if i in used: continue
            ri = wk.iloc[i]
            cand = [j for j in range(len(wk)) if j != i and j not in used
                    and np.isclose(wk.iloc[j]["spread"], -ri["spread"])
                    and wk.iloc[j]["is_home"] == 1 - ri["is_home"]]
            if not cand: continue
            j = cand[0]; used.add(i); used.add(j); rj = wk.iloc[j]
            pa, pb, ya, yb = ri["_p"], rj["_p"], int(ri["is_cover"]), int(rj["is_cover"])
            if ya == 0 and yb == 0: continue
            pick_y = ya if pa >= pb else yb
            bets += 1
            if pick_y == 1: wins += 1; units += UNIT_PROFIT_WIN
            else: units -= UNIT_LOSS
    return dict(bets=bets, win_pct=round(wins/bets, 4) if bets else 0,
                units=round(units, 2), roi=round(units/bets, 4) if bets else 0)


def main():
    df = pd.read_parquet(FEATURES)
    feat = [c for c in df.columns if c not in ID_COLS]
    print(f"Loaded {len(df):,} rows x {len(feat)} features")

    # X. PCA
    print("\n=== X. PCA cumulative explained variance ===")
    X = df[feat].copy()
    X = X.fillna(X.median(numeric_only=True))
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=len(feat)).fit(X_std)
    cum = np.cumsum(pca.explained_variance_ratio_)
    table = pd.DataFrame({"n_components": np.arange(1, len(feat) + 1),
                          "cum_var": cum.round(4)})
    print(table.to_string(index=False))
    for thr in [0.70, 0.80, 0.90, 0.95, 0.99]:
        k = int(np.searchsorted(cum, thr) + 1)
        print(f"  {int(thr*100)}% variance reached at K = {k} components")

    # Y. XGB feature importance (averaged across the 6 walk-forward fits)
    print("\n=== Y. XGB feature importance (avg across 6 walk-forward fits) ===")
    importances = np.zeros(len(feat))
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        tr = df[df["season"] < s]
        m = xgb.XGBClassifier(**XGB_KW)
        m.fit(tr[feat], tr["is_cover"])
        importances += m.feature_importances_
    importances /= 6
    imp_df = pd.DataFrame({"feature": feat, "importance": importances.round(5)})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nTop 20:")
    print(imp_df.head(20).to_string(index=False))
    print("\nBottom 10:")
    print(imp_df.tail(10).to_string(index=False))

    # Z. Backtest XGB with top-K features
    print("\n=== Z. Backtest with top-K features (bet every game) ===")
    rows = []
    for K in [10, 15, 20, 25, 30, 40, 51]:
        top_feats = imp_df.head(K)["feature"].tolist()
        pred = fit_predict_xgb(df, top_feats)
        r = pair_and_score(pred)
        rows.append({"K": K, **r})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
