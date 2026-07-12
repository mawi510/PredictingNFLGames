"""
v3 backtest -- CORRECTED bet accounting per user 2026-05-30:

Bug in v1/v2: iterated over every team-week row and bet whenever p>threshold.
For a game between A and B, this could bet BOTH sides (hedged wash) and also
counted pushes (where is_cover=0 for both teams) as losses on whichever side
we picked.

Fixes:
  - Per-GAME decision: pair team-weeks by (season, week, opposite spread,
    opposite is_home). For each game, pick argmax(P_A, P_B). Bet only if
    max(P_A, P_B) > threshold.
  - Push exclusion: if both rows in the game have is_cover=0, drop from ROI
    entirely (real sportsbooks refund pushes; counting as loss artificially
    deflates ROI).
  - Diagnostic: also reports how many GAMES this season had both-sides-over-
    threshold (the old double-bet wash) and how many were pushes.

Also runs three checks the user asked for, with the corrected accounting:
  CHECK 1: SEED STABILITY (D config across 5 seeds)
  CHECK 2: TAU SWEEP (tau in {2,3,4,5})
  CHECK 3: ISOTONIC vs PLATT calibration

Plus baseline diagnostic on Config D so we know how much of the headline +2.80%
under the old logic was artifact vs. real signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent

BREAKEVEN = 110 / 210
UNIT_PROFIT_WIN = 100 / 110
UNIT_LOSS = 1.0
THRESHOLD = BREAKEVEN
ID_COLS = ["season", "week", "team", "is_cover"]
LEAGUE_FALLBACK = 0.5


def build_features(tau: float) -> tuple[pd.DataFrame, list[str]]:
    v1 = pd.read_parquet(ROOT / "features.parquet")
    labels = pd.concat([
        pd.read_csv(REPO_ROOT / "nfl_training_data.csv", usecols=["season", "team", "is_cover"]),
        pd.read_csv(REPO_ROOT / "nfl_current_data.csv", usecols=["season", "team", "is_cover"]),
    ]).dropna(subset=["is_cover"])
    labels["is_cover"] = labels["is_cover"].astype(int)
    ts = labels.groupby(["season", "team"])["is_cover"].mean().reset_index().rename(
        columns={"is_cover": "cover_rate"})
    ts["season"] = ts["season"] + 1
    ts = ts.rename(columns={"cover_rate": "prev_season_cover_rate"})
    df = v1.merge(ts, on=["season", "team"], how="left")
    df["prev_season_cover_rate"] = df["prev_season_cover_rate"].fillna(LEAGUE_FALLBACK)
    df["weight_on_prior_season"] = np.exp(-df["week"] / tau)
    cur = df["is_cover_record_shifted"].fillna(df["prev_season_cover_rate"])
    df["blended_cover_rate"] = (
        df["weight_on_prior_season"] * df["prev_season_cover_rate"]
        + (1 - df["weight_on_prior_season"]) * cur
    )
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    return df, feat_cols


def fit_predict(X_tr, y_tr, X_te, calibrate: str, seed: int) -> np.ndarray:
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed, n_jobs=4,
    )
    if calibrate == "none":
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]
    n = len(X_tr); cut = int(n * 0.8)
    model.fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
    p_cal = model.predict_proba(X_tr.iloc[cut:])[:, 1]
    y_cal = y_tr.iloc[cut:]
    p_te_raw = model.predict_proba(X_te)[:, 1]
    if calibrate == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal).transform(p_te_raw)
    if calibrate == "platt":
        return LogisticRegression(C=1.0).fit(p_cal.reshape(-1, 1), y_cal).predict_proba(
            p_te_raw.reshape(-1, 1))[:, 1]
    raise ValueError(calibrate)


def pair_and_score(test_df: pd.DataFrame, probs: np.ndarray) -> dict:
    """
    Pair team-week rows into games and apply the corrected bet accounting.

    Returns: bets, wins, losses, pushes_excluded, both_over_threshold_games,
             games_total, units, roi.
    """
    test = test_df.reset_index(drop=True).copy()
    test["_p"] = probs

    bets = wins = losses = 0
    pushes = both_over = no_pick = games = 0
    units = 0.0
    picks_log = []  # for the live-tool flavor — every game with picked side + p

    for (s, w), wk in test.groupby(["season", "week"], sort=False):
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
            pa, pb = ri["_p"], rj["_p"]
            ya, yb = int(ri["is_cover"]), int(rj["is_cover"])

            # diagnostic counters
            if pa > THRESHOLD and pb > THRESHOLD:
                both_over += 1
            if ya == 0 and yb == 0:
                pushes += 1
                continue  # push: excluded from ROI

            # per-game pick: argmax, only if > threshold
            if pa >= pb:
                pick_p, pick_y, pick_team = pa, ya, ri["team"]
                opp_team = rj["team"]
            else:
                pick_p, pick_y, pick_team = pb, yb, rj["team"]
                opp_team = ri["team"]

            picks_log.append({
                "season": int(s), "week": int(w),
                "picked": pick_team, "opponent": opp_team,
                "p": round(float(pick_p), 4), "covered": int(pick_y),
            })

            if pick_p <= THRESHOLD:
                no_pick += 1
                continue
            bets += 1
            if pick_y == 1:
                wins += 1
                units += UNIT_PROFIT_WIN
            else:
                losses += 1
                units -= UNIT_LOSS

    return {
        "games": games,
        "pushes_excluded": pushes,
        "both_over_threshold_games": both_over,
        "no_pick": no_pick,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_pct": round(wins / bets, 4) if bets else 0.0,
        "units": round(units, 2),
        "roi": round(units / bets, 4) if bets else 0.0,
    }, picks_log


def run_walkforward(df, feat, calibrate, seed):
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    per = []
    all_picks = []
    for s in seasons:
        tr, te = df[df["season"] < s], df[df["season"] == s]
        p = fit_predict(tr[feat], tr["is_cover"], te[feat], calibrate, seed)
        res, picks = pair_and_score(te, p)
        per.append({"season": s, **res})
        all_picks.extend(picks)
    per_df = pd.DataFrame(per)
    agg = dict(
        games=per_df["games"].sum(),
        pushes_excluded=per_df["pushes_excluded"].sum(),
        both_over_threshold_games=per_df["both_over_threshold_games"].sum(),
        no_pick=per_df["no_pick"].sum(),
        bets=per_df["bets"].sum(),
        wins=per_df["wins"].sum(),
        losses=per_df["losses"].sum(),
        units=round(per_df["units"].sum(), 2),
    )
    agg["win_pct"] = round(agg["wins"] / agg["bets"], 4) if agg["bets"] else 0.0
    agg["roi"] = round(agg["units"] / agg["bets"], 4) if agg["bets"] else 0.0
    return per_df, agg, pd.DataFrame(all_picks)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("DIAGNOSTIC: Config D (54 feat + isotonic, seed=7, tau=3) -- old vs new accounting")
    df, feat = build_features(tau=3.0)
    per, agg, picks = run_walkforward(df, feat, "isotonic", seed=7)
    print("\nPer-season under CORRECTED accounting:")
    print(per.to_string(index=False))
    print(f"\nAggregate: games={agg['games']}, pushes_excluded={agg['pushes_excluded']}, "
          f"games_with_both_sides_over_threshold (old-logic double-bet)={agg['both_over_threshold_games']}, "
          f"no_pick={agg['no_pick']}, bets={agg['bets']}, win%={agg['win_pct']*100:.2f}%, "
          f"units={agg['units']}, ROI={agg['roi']*100:+.2f}%")
    picks.to_csv(ROOT / "picks_log_D.csv", index=False)
    print(f"\nFull picks log -> picks_log_D.csv ({len(picks)} non-push games)")

    section("CHECK 1: SEED STABILITY (Config D, tau=3, isotonic)")
    rows = []
    for seed in [1, 3, 7, 11, 42]:
        _, agg_s, _ = run_walkforward(df, feat, "isotonic", seed=seed)
        rows.append({"seed": seed, "bets": agg_s["bets"], "win_pct": agg_s["win_pct"],
                     "units": agg_s["units"], "roi": agg_s["roi"]})
    a = pd.DataFrame(rows)
    print(a.to_string(index=False))
    print(f"\nROI mean: {a['roi'].mean()*100:+.2f}% +/- {a['roi'].std()*100:.2f}%")
    print(f"Bets mean: {a['bets'].mean():.0f} +/- {a['bets'].std():.0f}")

    section("CHECK 2: TAU SWEEP (Config D, isotonic, seed=7)")
    rows = []
    for tau in [2, 3, 4, 5]:
        df_t, feat_t = build_features(tau=float(tau))
        _, agg_t, _ = run_walkforward(df_t, feat_t, "isotonic", seed=7)
        rows.append({"tau": tau, "bets": agg_t["bets"], "win_pct": agg_t["win_pct"],
                     "units": agg_t["units"], "roi": agg_t["roi"]})
    print(pd.DataFrame(rows).to_string(index=False))

    section("CHECK 3: ISOTONIC vs PLATT (Config D, tau=3, seed=7)")
    df, feat = build_features(tau=3.0)
    rows = []
    per_each = {}
    for cal in ["isotonic", "platt"]:
        per_c, agg_c, _ = run_walkforward(df, feat, cal, seed=7)
        rows.append({"calibrator": cal, "bets": agg_c["bets"], "win_pct": agg_c["win_pct"],
                     "units": agg_c["units"], "roi": agg_c["roi"]})
        per_each[cal] = per_c
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPer-season bet counts:")
    for cal, per_c in per_each.items():
        print(f"  {cal}: " + ", ".join(f"{int(s)}={int(b)}"
              for s, b in zip(per_c["season"], per_c["bets"])))
    print("\nPer-season ROI:")
    for cal, per_c in per_each.items():
        print(f"  {cal}: " + ", ".join(f"{int(s)}={r*100:+.2f}%"
              for s, r in zip(per_c["season"], per_c["roi"])))


if __name__ == "__main__":
    main()
