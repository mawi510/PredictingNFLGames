"""FastAPI service for the NFL spread-cover model.

Serves live predictions from the margin model (v2): an XGBoost regressor
predicts the expected scoring margin from 44 team-form features (no market
inputs), and the cover probability is computed at the market spread via a
Student-t CDF calibrated on out-of-sample residuals:

    P(cover) = t.cdf((predicted_margin + spread) / scale_t, df=nu_t)

Inputs are validated against a versioned feature manifest so a pipeline column
change can't silently corrupt inference. Missing features stay NaN — the
model handles them natively (do not fill with 0).

Endpoints:
    GET  /health        -> liveness + whether a real model is loaded
    GET  /status        -> season awareness (in-season? week? predictions ready?)
    GET  /track-record  -> season win/loss backtest (precomputed weekly)
    POST /predict        -> margin + cover probability for one team-week

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
MANIFEST_PATH = API_DIR / "feature_manifest_margin_v2.json"
# The model bundle is committed to the repo and baked into the image; no S3
# pull needed at boot (unlike the old RF, which was too big to commit).
MODEL_PATH = os.getenv("MODEL_PATH", str(API_DIR / "model_margin_v2.pkl"))
TRACK_RECORD_PATH = os.getenv("TRACK_RECORD_PATH", str(API_DIR / "track_record.json"))
# The weekly job writes the backtest here; the API reads it live (cached) so a
# data refresh never requires an image rebuild/redeploy.
TRACK_RECORD_S3_URI = os.getenv(
    "TRACK_RECORD_S3_URI", "s3://nfl.data/track_record.json"
)
# Seconds to cache the S3 track record in memory (it only changes weekly).
TRACK_RECORD_TTL = int(os.getenv("TRACK_RECORD_TTL", "600"))

# Per-team current-week features (precomputed weekly by pipeline/current_features.py).
CURRENT_FEATURES_PATH = os.getenv(
    "CURRENT_FEATURES_PATH", str(API_DIR / "current_features.json")
)
CURRENT_FEATURES_S3_URI = os.getenv(
    "CURRENT_FEATURES_S3_URI", "s3://nfl.data/current_features.json"
)
CURRENT_FEATURES_TTL = int(os.getenv("CURRENT_FEATURES_TTL", "600"))

# Per-team weekly stat series (precomputed weekly by pipeline/team_stats.py).
TEAM_STATS_PATH = os.getenv("TEAM_STATS_PATH", str(API_DIR / "team_stats.json"))
TEAM_STATS_S3_URI = os.getenv("TEAM_STATS_S3_URI", "s3://nfl.data/team_stats.json")
TEAM_STATS_TTL = int(os.getenv("TEAM_STATS_TTL", "3600"))

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
    """Load the model bundle, or return None so the API still boots for dev.

    The bundle is a dict: {regressor (XGBRegressor), feature_names, scale_t,
    nu_t, version, ...} written by pipeline/train_margin.py. When None,
    /predict returns a clearly-labelled stub so the frontend and CI still work.
    """
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    import joblib  # imported lazily so the stub path needs no ML deps

    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Named feature map + market spread for a single team-week.

    Keys must match the manifest's feature names. Missing or null features
    stay NaN (the model handles them natively); unknown keys are rejected so
    typos and stale pipelines surface loudly instead of silently shifting the
    vector. The spread is NOT a model feature — it enters only in the final
    cover-probability calculation.
    """

    features: dict[str, Optional[float]] = Field(
        ..., description="Mapping of feature name -> value for one team-week."
    )
    spread: float = Field(
        ..., description="Market spread for this team (negative = favored)."
    )
    team: Optional[str] = Field(None, description="Team abbreviation, for logging/UI.")


class PredictResponse(BaseModel):
    team: Optional[str]
    cover_probability: float
    predicted_margin: float
    edge_pts: float
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
    bundle = load_model()
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": bundle["version"] if bundle else None,
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

    # Build the vector in the manifest's exact order; missing/null -> NaN.
    def _val(name: str) -> float:
        v = req.features.get(name)
        return float(v) if v is not None else float("nan")

    vector = np.array([[_val(name) for name in expected]], dtype=float)

    bundle = load_model()
    if bundle is None:
        # Deterministic stub so the frontend has something to render in dev:
        # a market-implied-ish probability from the spread alone.
        stub_p = float(
            1 / (1 + np.exp(-req.spread / (manifest["scale_t"] or 13.0)))
        )
        return PredictResponse(
            team=req.team,
            cover_probability=round(stub_p, 4),
            predicted_margin=0.0,
            edge_pts=round(req.spread, 2),
            model_version=f"stub-{manifest['version']}",
            is_stub=True,
        )

    from scipy import stats  # lazy: keeps the stub path dependency-free

    mu = float(bundle["regressor"].predict(vector)[0])
    edge = mu + req.spread
    p_cover = float(stats.t.cdf(edge / bundle["scale_t"], df=bundle["nu_t"]))
    return PredictResponse(
        team=req.team,
        cover_probability=round(p_cover, 4),
        predicted_margin=round(mu, 2),
        edge_pts=round(edge, 2),
        model_version=bundle["version"],
        is_stub=False,
    )


# --------------------------------------------------------------------------- #
# Team-driven prediction: the frontend asks for a team, the API does the
# feature lookup (from the precomputed current_features.json) and predicts.
# --------------------------------------------------------------------------- #
_features_cache: dict = {"data": None, "ts": 0.0}


def load_current_features() -> Optional[dict]:
    import time

    now = time.time()
    if (
        _features_cache["data"] is not None
        and now - _features_cache["ts"] < CURRENT_FEATURES_TTL
    ):
        return _features_cache["data"]

    data = None
    if CURRENT_FEATURES_S3_URI:
        try:
            data = _read_json_from_s3(CURRENT_FEATURES_S3_URI)
        except Exception:
            data = None
    if data is None and Path(CURRENT_FEATURES_PATH).exists():
        with open(CURRENT_FEATURES_PATH) as f:
            data = json.load(f)
    if data is not None:
        _features_cache.update(data=data, ts=now)
    return data


@app.get("/teams")
def teams() -> dict:
    """List teams with display info for the picker (no prediction yet)."""
    data = load_current_features()
    if data is None:
        raise HTTPException(status_code=404, detail="Current features not available yet.")
    items = [
        {
            "team": t,
            "week": info["week"],
            "spread": info["spread"],
            "cover_record": info["cover_record"],
        }
        for t, info in sorted(data["teams"].items())
    ]
    return {"season": data["season"], "latest_week": data["latest_week"], "teams": items}


@app.get("/teams/{team}/prediction")
def team_prediction(team: str) -> dict:
    """Predict margin + cover probability for one team's current week."""
    data = load_current_features()
    if data is None:
        raise HTTPException(status_code=404, detail="Current features not available yet.")
    info = data["teams"].get(team.upper())
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown team '{team}'.")

    result = predict(
        PredictRequest(
            team=team.upper(), features=info["features"], spread=info["spread"]
        )
    )
    return {
        "team": team.upper(),
        "season": data["season"],
        "week": info["week"],
        "spread": info["spread"],
        "cover_record": info["cover_record"],
        "cover_probability": result.cover_probability,
        "predicted_margin": result.predicted_margin,
        "edge_pts": result.edge_pts,
        "model_version": result.model_version,
        "is_stub": result.is_stub,
    }


# --------------------------------------------------------------------------- #
# Team stat series for the current-season + historical charts.
# --------------------------------------------------------------------------- #
_stats_cache: dict = {"data": None, "ts": 0.0}


def load_team_stats() -> Optional[dict]:
    import time

    now = time.time()
    if _stats_cache["data"] is not None and now - _stats_cache["ts"] < TEAM_STATS_TTL:
        return _stats_cache["data"]

    data = None
    if TEAM_STATS_S3_URI:
        try:
            data = _read_json_from_s3(TEAM_STATS_S3_URI)
        except Exception:
            data = None
    if data is None and Path(TEAM_STATS_PATH).exists():
        with open(TEAM_STATS_PATH) as f:
            data = json.load(f)
    if data is not None:
        _stats_cache.update(data=data, ts=now)
    return data


@app.get("/teams/{team}/stats")
def team_stats(team: str) -> dict:
    """Weekly stat series for a team across the current and past seasons."""
    data = load_team_stats()
    if data is None:
        raise HTTPException(status_code=404, detail="Team stats not available yet.")
    entry = data["teams"].get(team.upper())
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No stats for team '{team}'.")
    return {
        "team": team.upper(),
        "metrics": data["metrics"],
        "seasons": entry["seasons"],
        "series": entry["series"],
    }
