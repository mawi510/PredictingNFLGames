"""Precompute per-team weekly stat series for the site's team-stats charts.

Combines the current season and past seasons into one compact JSON the API can
serve (no pandas in the API image), powering "how has my team been doing?" charts
across the current season and history — like the old Streamlit current/historical
sections.

Metric sourcing differs by file, so we normalize to shared logical keys:
  * historical data (nfl_historical_data.csv): raw columns, e.g. `points_scored`
  * current data (nfl_current_data.csv): only `_shifted` columns exist, so we read
    `points_scored_shifted` etc. (same convention the old app charted)

A season present in the current file wins over the historical file for that year,
so each season's series is internally consistent (one source).

Writes team_stats.json locally and, if TEAM_STATS_S3_URI is set, to S3.

Usage:
    python -m pipeline.team_stats
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_PATH = os.getenv("HISTORICAL_DATA_PATH", str(ROOT / "nfl_training_data.csv"))
CURRENT_PATH = os.getenv("DATA_PATH", str(ROOT / "nfl_current_data.csv"))
OUTPUT_PATH = os.getenv("TEAM_STATS_PATH", str(ROOT / "api" / "team_stats.json"))
OUTPUT_S3_URI = os.getenv("TEAM_STATS_S3_URI", "")

# Logical metric key -> display label, in display order.
METRICS = {
    "points_scored": "Points Scored",
    "points_allowed": "Points Allowed",
    "scoring_margin": "Scoring Margin",
    "passing_yards": "Passing Yards",
    "passing_tds": "Passing TDs",
    "rushing_yards": "Rushing Yards",
    "rushing_tds": "Rushing TDs",
    "interceptions": "Interceptions",
}


def _read(path: str) -> pd.DataFrame:
    storage_options = None
    if path.startswith("s3://"):
        storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
    df = pd.read_csv(path, storage_options=storage_options)
    df.columns = df.columns.str.lower()
    return df


def _series_for(df: pd.DataFrame, suffix: str) -> dict:
    """Build {team: {season: {weeks, metrics:{key:[values]}}}} from one source.

    `suffix` is "" for raw columns or "_shifted" for the current file.
    Only metrics whose source column exists are included.
    """
    present = {k: f"{k}{suffix}" for k in METRICS if f"{k}{suffix}" in df.columns}
    out: dict = {}
    if not present:
        return out
    for (team, season), grp in df.groupby(["team", "season"]):
        grp = grp.sort_values("week")
        weeks = [int(w) for w in grp["week"]]
        metrics = {
            key: [round(float(v), 2) if pd.notna(v) else None for v in grp[col]]
            for key, col in present.items()
        }
        out.setdefault(str(team), {})[str(int(season))] = {
            "weeks": weeks,
            "metrics": metrics,
        }
    return out


def build() -> dict:
    current = _read(CURRENT_PATH)
    current_series = _series_for(current, "_shifted")
    current_seasons = {s for t in current_series.values() for s in t}

    historical = _read(HISTORICAL_PATH)
    historical_series = _series_for(historical, "")

    # Merge: current wins for any season it covers.
    teams: dict = {}
    for source in (historical_series, current_series):
        for team, seasons in source.items():
            for season, payload in seasons.items():
                if source is historical_series and season in current_seasons:
                    continue  # current file owns this season
                teams.setdefault(team, {})[season] = payload

    # Which metrics actually have data somewhere.
    used = {
        m
        for t in teams.values()
        for s in t.values()
        for m in s["metrics"]
    }
    metrics = [{"key": k, "label": v} for k, v in METRICS.items() if k in used]

    # Per-team sorted season list for the UI dropdown.
    teams_out = {
        team: {
            "seasons": sorted((int(s) for s in seasons), reverse=True),
            "series": seasons,
        }
        for team, seasons in teams.items()
    }

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "metrics": metrics,
        "teams": teams_out,
    }


def main() -> None:
    data = build()
    payload = json.dumps(data)
    Path(OUTPUT_PATH).write_text(payload)
    print(
        f"Wrote {OUTPUT_PATH}: {len(data['teams'])} teams, "
        f"{len(data['metrics'])} metrics ({', '.join(m['key'] for m in data['metrics'])})"
    )
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
