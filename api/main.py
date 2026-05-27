"""FastAPI service for the NFL spread-cover model.

Serves live predictions from the trained RandomForest. Replaces the old Flask
app that took a fragile positional list; this version validates inputs against a
versioned feature manifest so a pipeline column change can't silently corrupt
inference.

Endpoints:
    GET  /health        -> liveness + whether a real model is loaded
    GET  /status        -> season awareness (in-season? week? predictions ready?)
    GET  /track-record  -> season win/loss backtest (precomputed weekly)
    POST /predict        -> cover probability for one team-week

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

API_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = API_DIR / "feature_manifest.json"
# Model is loaded from a local path if present, else pulled from S3 at boot.
MODEL_PATH = os.getenv("MODEL_PATH", str(API_DIR / "model.pkl"))
# If the artifact isn't on disk, pull it from S3 at boot (where it's versioned
# alongside the data). e.g. s3://nfl.data/models/cover_classifier.joblib
MODEL_S3_URI = os.getenv("MODEL_S3_URI", "")
TRACK_RECORD_PATH = os.getenv("TRACK_RECORD_PATH", str(API_DIR / "track_record.json"))
# The weekly job writes the backtest here; the API reads it live (cached) so a
# data refresh never requires an image rebuild/redeploy.
TRACK_RECORD_S3_URI = os.getenv(
    "TRACK_RECORD_S3_URI", "s3://nfl.data/track_record.json"
)
# Seconds to cache the S3 track record in memory (it only changes weekly).
TRACK_RECORD_TTL = int(os.getenv("TRACK_RECORD_TTL", "600"))

# Allow the Vercel site (and local dev) to call the API from the browser.
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://promatchpredict.com,https://www.promatchpredict.com,http://localhost:3000",
).split(",")


# --------------------------------------------------------------------------- #
# Feature manifest — the source of truth for the model's input contract.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model, or return None so the API still boots for dev.

    When None, /predict returns a clearly-labelled stub probability so the
    frontend and CI can be built before the real artifact is recovered from EC2.
    """
    path = Path(MODEL_PATH)
    if not path.exists() and MODEL_S3_URI:
        _download_from_s3(MODEL_S3_URI, path)
    if not path.exists():
        return None
    import joblib  # imported lazily so the stub path needs no sklearn

    return joblib.load(path)


def _download_from_s3(uri: str, dest: Path) -> None:
    """Download s3://bucket/key -> dest, using the same env creds as the pipeline."""
    import boto3

    bucket, _, key = uri.removeprefix("s3://").partition("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    ).download_file(bucket, key, str(dest))


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Named feature map for a single team-week.

    Keys must match the manifest's feature names. Missing features default to
    0.0 (the old app did `fillna(0)`); unknown keys are rejected so typos and
    stale pipelines surface loudly instead of silently shifting the vector.
    """

    features: dict[str, float] = Field(
        ..., description="Mapping of feature name -> value for one team-week."
    )
    team: Optional[str] = Field(None, description="Team abbreviation, for logging/UI.")


class PredictResponse(BaseModel):
    team: Optional[str]
    cover_probability: float
    model_version: str
    is_stub: bool = False


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="ProMatchPredict Model API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    manifest = load_manifest()
    return {
        "status": "ok",
        "model_loaded": load_model() is not None,
        "feature_manifest_version": manifest["version"],
        "n_features": manifest["n_features"],
    }


# Model needs ~3 weeks of history before it predicts (matches the old app's
# "come back for week 4" rule).
MIN_WEEK_FOR_PREDICTIONS = 3


def _in_season(now: Optional[datetime] = None) -> bool:
    """True during the NFL regular season window (Sept through early Jan)."""
    now = now or datetime.now(timezone.utc)
    return now.month in (9, 10, 11, 12, 1)


@app.get("/status")
def status() -> dict:
    """Season awareness so the site can adjust its copy.

    Tells the frontend whether it's NFL season, the latest season/week we have
    data for, and whether predictions are available yet (the model needs a few
    weeks of data). Single source of truth so the UI doesn't guess from the date.
    """
    in_season = _in_season()
    season = latest_week = None
    try:
        tr = track_record()  # cached; carries season + latest_week
        season = tr.get("season")
        latest_week = tr.get("latest_week")
    except HTTPException:
        pass

    has_data = (latest_week or 0) >= MIN_WEEK_FOR_PREDICTIONS
    predictions_available = bool(in_season and has_data)

    if not in_season:
        reason, message = (
            "off_season",
            "The NFL season isn't underway right now. Check back in September for "
            "weekly spread predictions — here's how the model did last season.",
        )
    elif not has_data:
        reason, message = (
            "insufficient_data",
            f"The model needs {MIN_WEEK_FOR_PREDICTIONS} weeks of data before it can "
            "predict. Come back right before Week 4.",
        )
    else:
        reason, message = "ok", f"Week {latest_week} predictions are live."

    return {
        "in_season": in_season,
        "season": season,
        "latest_week": latest_week,
        "predictions_available": predictions_available,
        "reason": reason,
        "message": message,
    }


_track_cache: dict = {"data": None, "ts": 0.0}


@app.get("/track-record")
def track_record() -> dict:
    """Return the precomputed season win/loss backtest.

    Written weekly by pipeline/backtest.py to S3 (or a local file in dev). Read
    live with a short in-memory cache so a data refresh needs no redeploy.
    """
    import time

    now = time.time()
    if _track_cache["data"] is not None and now - _track_cache["ts"] < TRACK_RECORD_TTL:
        return _track_cache["data"]

    data = None
    if TRACK_RECORD_S3_URI:
        try:
            data = _read_json_from_s3(TRACK_RECORD_S3_URI)
        except Exception:
            data = None  # fall back to a local file if S3 is unavailable
    if data is None and Path(TRACK_RECORD_PATH).exists():
        with open(TRACK_RECORD_PATH) as f:
            data = json.load(f)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Track record not yet computed. Run pipeline/backtest.py.",
        )

    _track_cache.update(data=data, ts=now)
    return data


def _read_json_from_s3(uri: str) -> dict:
    import boto3

    bucket, _, key = uri.removeprefix("s3://").partition("/")
    obj = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    ).get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    manifest = load_manifest()
    expected = manifest["features"]
    expected_set = set(expected)

    unknown = [k for k in req.features if k not in expected_set]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown feature(s) not in manifest v{manifest['version']}: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}",
        )

    # Build the vector in the manifest's exact order; default missing to 0.0.
    vector = np.array(
        [[float(req.features.get(name, 0.0)) for name in expected]], dtype=float
    )

    model = load_model()
    if model is None:
        # Deterministic stub so the frontend has something to render in dev.
        stub = float(1 / (1 + np.exp(-vector.sum() / (len(expected) or 1))))
        return PredictResponse(
            team=req.team,
            cover_probability=round(stub, 4),
            model_version=f"stub-{manifest['version']}",
            is_stub=True,
        )

    proba = float(model.predict_proba(vector)[0][1])
    return PredictResponse(
        team=req.team,
        cover_probability=round(proba, 4),
        model_version=manifest["version"],
        is_stub=False,
    )
