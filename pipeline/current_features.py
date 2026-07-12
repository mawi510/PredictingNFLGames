"""Precompute per-team current-week features for the prediction API.

The margin model (v2) needs a team's 44-feature vector plus the market spread
to predict, but the slim API image has no pandas and shouldn't load the wide
CSV. So this job — run weekly alongside the backtest — distills the latest
week of data into a small JSON the API reads directly: for each team, its
named feature values (null where unavailable; the model handles missing
natively) plus a few display fields (spread, week, cover record).

Two model features are opponent-relative and not CSV columns; they're built
here via pipeline.margin_model.attach_matchup_features().

Writes current_features.json locally and, if CURRENT_FEATURES_S3_URI is set,
uploads it to S3.

Usage:
    python -m pipeline.current_features
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from pipeline.margin_model import attach_matchup_features

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "api" / "feature_manifest_margin_v2.json"
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "nfl_current_data.csv"))
OUTPUT_PATH = os.getenv("CURRENT_FEATURES_PATH", str(ROOT / "api" / "current_features.json"))
OUTPUT_S3_URI = os.getenv("CURRENT_FEATURES_S3_URI", "")

# Display field -> source column in the data.
COVER_RECORD_COL = "is_cover_record_shifted"


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


def build() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    features = manifest["features"]
    df = load_data()

    latest_season = int(df["season"].max())
    latest_week = int(df[df["season"] == latest_season]["week"].max())
    season = df[df["season"] == latest_season]
    season = attach_matchup_features(season)

    teams = {}
    for team, grp in season.groupby("team"):
        row = grp.sort_values("week").iloc[-1]  # latest week for this team
        teams[str(team)] = {
            "week": int(row["week"]),
            "spread": _num(row.get("spread")),
            "cover_record": _num(row.get(COVER_RECORD_COL)),
            # Ordered feature values; unavailable -> null (model is NaN-native).
            "features": {f: _num_or_none(row.get(f)) for f in features},
        }

    return {
        "season": latest_season,
        "latest_week": latest_week,
        "model_version": manifest["version"],
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "teams": teams,
    }


def _num(v) -> float:
    try:
        return float(v) if pd.notna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _num_or_none(v):
    try:
        return float(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    data = build()
    payload = json.dumps(data)
    Path(OUTPUT_PATH).write_text(payload)
    print(f"Wrote {OUTPUT_PATH}: {len(data['teams'])} teams, "
          f"{data['season']} week {data['latest_week']}")
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
        print(f"Uploaded to {OUTPUT_S3_URI}")


if __name__ == "__main__":
    main()
