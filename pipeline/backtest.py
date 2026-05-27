"""Season win/loss backtest for the spread-cover model.

Answers the site's headline feature: "if you'd followed the model's picks this
season, what would your record be?" Runs the trained model over every team-week
of the latest season, treats P(cover) > threshold as a "bet this team to cover"
pick, and scores it against the actual `is_cover` outcome.

Methodology (kept transparent and surfaced in the output JSON):
  * Picks start at `MIN_WEEK` (the model needs ~3 weeks of history).
  * Each team-week is an independent pick, matching how users read the site
    (they look up a single team). Both sides of a game can therefore be picks.
  * Units assume flat -110 bets: a win pays +0.909u, a loss -1.0u.
  * Features are already `_shifted` (built from prior weeks), so this is an
    honest walk-forward backtest with no leakage.

Writes track_record.json, consumed by the API's /track-record endpoint.

Usage:
    python -m pipeline.backtest                       # local CSV + local model
    DATA_PATH=s3://nfl.data/nfl_current_data.csv \
    MODEL_PATH=api/model.pkl python -m pipeline.backtest
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "api" / "feature_manifest.json"
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "nfl_current_data.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "api" / "model.pkl"))
MODEL_S3_URI = os.getenv("MODEL_S3_URI", "")
OUTPUT_PATH = os.getenv("TRACK_RECORD_PATH", str(ROOT / "api" / "track_record.json"))
# When set, also upload the result here so the API can serve it without a redeploy.
OUTPUT_S3_URI = os.getenv("TRACK_RECORD_S3_URI", "")

THRESHOLD = float(os.getenv("PICK_THRESHOLD", "0.5"))
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


def load_model():
    path = Path(MODEL_PATH)
    if not path.exists() and MODEL_S3_URI:
        import boto3

        bucket, _, key = MODEL_S3_URI.removeprefix("s3://").partition("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        ).download_file(bucket, key, str(path))
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def predicted_cover_proba(df: pd.DataFrame, features: list[str], model) -> np.ndarray:
    """P(cover) per row. Falls back to a deterministic stub if no model yet."""
    X = df.reindex(columns=features).fillna(0.0).astype(float).to_numpy()
    if model is None:
        return 1 / (1 + np.exp(-X.sum(axis=1) / max(len(features), 1)))
    return model.predict_proba(X)[:, 1]


def build_track_record() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    features = manifest["features"]
    df = load_data()
    model = load_model()

    latest_season = int(df["season"].max())
    latest_week = int(df[df["season"] == latest_season]["week"].max())
    season = df[(df["season"] == latest_season) & (df["week"] >= MIN_WEEK)].copy()
    season = season.sort_values(["week", "team"]).reset_index(drop=True)

    season["cover_proba"] = predicted_cover_proba(season, features, model)
    picks = season[season["cover_proba"] > THRESHOLD].copy()
    picks["won"] = picks["is_cover"] == 1
    picks["units"] = np.where(picks["won"], WIN_PAYOUT, -1.0)

    wins = int(picks["won"].sum())
    losses = int((~picks["won"]).sum())
    total = wins + losses
    win_pct = round(wins / total, 4) if total else 0.0
    total_units = round(float(picks["units"].sum()), 2)
    roi = round(total_units / total, 4) if total else 0.0

    # Week-by-week cumulative curve for the equity chart on the site.
    weekly = []
    cum_units = 0.0
    cum_w = cum_l = 0
    for week, grp in picks.groupby("week"):
        w = int(grp["won"].sum())
        l = int((~grp["won"]).sum())
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
        "is_stub": model is None,
        "model_version": manifest["version"],
        "methodology": {
            "pick_rule": f"bet team to cover when P(cover) > {THRESHOLD}",
            "min_week": MIN_WEEK,
            "odds_assumption": "flat -110 (win pays +0.909u, loss -1.0u)",
            "pick_unit": "one team-week (both sides of a game can be picks)",
        },
        "record": {
            "wins": wins,
            "losses": losses,
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
        f"  {record['season']} season: {r['wins']}-{r['losses']} "
        f"({r['win_pct']*100:.1f}%), {r['total_units']:+.2f}u, ROI {r['roi']*100:+.1f}%"
    )


if __name__ == "__main__":
    main()
