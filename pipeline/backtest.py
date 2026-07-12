"""Season win/loss backtest for the margin model (v2).

Answers the site's headline feature: "if you'd followed the model's picks this
season, what would your record be?" For every game of the latest season, the
model predicts each side's cover probability at the market spread, picks the
higher-probability side, and is scored against the actual `is_cover` outcome.

Methodology (kept transparent and surfaced in the output JSON):
  * One pick per GAME (sides are paired; never both sides of the same game).
  * Pushes (neither side covers) are excluded from the record entirely —
    a real sportsbook refunds them.
  * Picks start at `MIN_WEEK` (matches the site's "come back for Week 4" rule).
  * Units assume flat -110 bets: a win pays +0.909u, a loss -1.0u.
  * Features are `_shifted` (built from prior weeks), so this is an honest
    walk-forward backtest with no leakage — same accounting the research
    lockbox validation used (research/xgboost/lockbox_2025.py).

Writes track_record.json, consumed by the API's /track-record endpoint.

Usage:
    python -m pipeline.backtest                       # local CSV + local model
    DATA_PATH=s3://nfl.data/nfl_current_data.csv python -m pipeline.backtest
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.margin_model import (
    DEFAULT_BUNDLE_PATH,
    attach_matchup_features,
    load_bundle,
    pair_games,
    predict_cover,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "api" / "feature_manifest_margin_v2.json"
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "nfl_current_data.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_BUNDLE_PATH))
OUTPUT_PATH = os.getenv("TRACK_RECORD_PATH", str(ROOT / "api" / "track_record.json"))
# When set, also upload the result here so the API can serve it without a redeploy.
OUTPUT_S3_URI = os.getenv("TRACK_RECORD_S3_URI", "")

MIN_WEEK = int(os.getenv("MIN_WEEK", "4"))
WIN_PAYOUT = 100 / 110  # flat -110 odds


def load_data() -> pd.DataFrame:
    storage_options = None
    if DATA_PATH.startswith("s3://"):
        storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
    df = pd.read_csv(DATA_PATH, storage_options=storage_options)
    df.columns = df.columns.str.lower()
    return df


def build_track_record() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    df = load_data()
    bundle = load_bundle(MODEL_PATH)

    latest_season = int(df["season"].max())
    latest_week = int(df[df["season"] == latest_season]["week"].max())
    season = df[(df["season"] == latest_season) & (df["week"] >= MIN_WEEK)].copy()
    season = season.sort_values(["week", "team"]).reset_index(drop=True)
    season = attach_matchup_features(season)

    if bundle is None:
        # Deterministic stub so the plumbing can run without the artifact.
        season["cover_proba"] = 1 / (1 + np.exp(-season["spread"].fillna(0.0) / 7.0))
    else:
        _, season["cover_proba"] = predict_cover(season, bundle)

    # One pick per game: pair sides, take the higher-probability one.
    pick_rows = []
    pushes = 0
    for week, wk in season.groupby("week", sort=True):
        wk = wk.reset_index(drop=True)
        for i, j in pair_games(wk):
            ri, rj = wk.iloc[i], wk.iloc[j]
            if int(ri["is_cover"]) == 0 and int(rj["is_cover"]) == 0:
                pushes += 1
                continue
            if float(ri["cover_proba"]) >= float(rj["cover_proba"]):
                side = ri
            else:
                side = rj
            won = int(side["is_cover"]) == 1
            pick_rows.append({
                "week": int(week),
                "team": str(side["team"]),
                "won": won,
                "units": WIN_PAYOUT if won else -1.0,
            })

    picks = pd.DataFrame(pick_rows, columns=["week", "team", "won", "units"])

    wins = int(picks["won"].sum()) if len(picks) else 0
    losses = int(len(picks) - wins)
    total = wins + losses
    win_pct = round(wins / total, 4) if total else 0.0
    total_units = round(float(picks["units"].sum()), 2) if total else 0.0
    roi = round(total_units / total, 4) if total else 0.0

    # Week-by-week cumulative curve for the equity chart on the site.
    weekly = []
    cum_units = 0.0
    cum_w = cum_l = 0
    for week, grp in picks.groupby("week"):
        w = int(grp["won"].sum())
        l = int(len(grp) - w)
        cum_w += w
        cum_l += l
        cum_units += float(grp["units"].sum())
        weekly.append(
            {
                "week": int(week),
                "wins": w,
                "losses": l,
                "cumulative_wins": cum_w,
                "cumulative_losses": cum_l,
                "cumulative_units": round(cum_units, 2),
            }
        )

    return {
        "season": latest_season,
        "latest_week": latest_week,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "is_stub": bundle is None,
        "model_version": manifest["version"],
        "methodology": {
            "pick_rule": "one pick per game: the side with the higher predicted "
                         "cover probability at the market spread",
            "min_week": MIN_WEEK,
            "odds_assumption": "flat -110 (win pays +0.909u, loss -1.0u)",
            "pick_unit": "one game (sides are paired; pushes excluded/refunded)",
        },
        "record": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total_picks": total,
            "win_pct": win_pct,
            "total_units": total_units,
            "roi": roi,
        },
        "weekly": weekly,
    }


def main() -> None:
    record = build_track_record()
    payload = json.dumps(record, indent=2)
    Path(OUTPUT_PATH).write_text(payload)
    if OUTPUT_S3_URI:
        import boto3

        bucket, _, key = OUTPUT_S3_URI.removeprefix("s3://").partition("/")
        boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        ).put_object(
            Bucket=bucket, Key=key, Body=payload.encode(), ContentType="application/json"
        )
        print(f"Uploaded track record to {OUTPUT_S3_URI}")
    r = record["record"]
    tag = " (STUB MODEL)" if record["is_stub"] else ""
    print(f"Wrote {OUTPUT_PATH}{tag}")
    print(
        f"  {record['season']} season: {r['wins']}-{r['losses']}-{r['pushes']}p "
        f"({r['win_pct']*100:.1f}%), {r['total_units']:+.2f}u, ROI {r['roi']*100:+.1f}%"
    )


if __name__ == "__main__":
    main()
