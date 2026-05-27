"""Precompute per-team current-week features for the prediction API.

The model needs a team's full 986-feature vector to predict, but the frontend
(and the slim API image, which has no pandas) shouldn't load the wide CSV. So
this job — run weekly alongside the backtest — distills the latest week of data
into a small JSON the API can read directly: for each team, its ordered feature
values plus a few display fields (spread, week, cover record).

Writes current_features.json locally and, if TRACK_RECORD_S3_URI's bucket is
configured via CURRENT_FEATURES_S3_URI, uploads it to S3.

Usage:
    python -m pipeline.current_features
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "api" / "feature_manifest.json"
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

    teams = {}
    for team, grp in season.groupby("team"):
        row = grp.sort_values("week").iloc[-1]  # latest week for this team
        teams[str(team)] = {
            "week": int(row["week"]),
            "spread": _num(row.get("spread")),
            "cover_record": _num(row.get(COVER_RECORD_COL)),
            # Ordered feature values, missing -> 0.0 (matches the old fillna(0)).
            "features": {f: _num(row.get(f)) for f in features},
        }

    return {
        "season": latest_season,
        "latest_week": latest_week,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "teams": teams,
    }


def _num(v) -> float:
    try:
        return float(v) if pd.notna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


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
