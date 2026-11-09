"""FastAPI service for the NFL spread-cover model.

Serves live predictions from the margin model (v3): a ridge regression on the
differences between the two teams' 44 form features predicts the home margin
(this week's line is not an input). The cover probability is computed at the
market spread:

    P(cover) = t.cdf((predicted_margin + spread) / scale_t, df=nu_t)

with scale_t fitted on out-of-sample cover outcomes (pipeline/train_margin.py).

Inputs are validated against the bundle's feature list so a pipeline column
change can't silently corrupt inference. A missing feature contributes nothing
to the prediction.

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
MANIFEST_PATH = API_DIR / "model_margin_v3.json"  # the bundle doubles as the manifest
# The model bundle is committed to the repo and baked into the image; no S3
# pull needed at boot (unlike the old RF, which was too big to commit).
MODEL_PATH = os.getenv("MODEL_PATH", str(API_DIR / "model_margin_v3.json"))
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

    The bundle is plain JSON written by pipeline/train_margin.py: feature
    order, ridge coefficients, feature sds and means, scale_t, nu_t, version.
    When None, /predict returns a clearly-labelled stub so the frontend and CI
    still work.
    """
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _vector(features: dict, names: list) -> np.ndarray:
    """Feature values in bundle order; missing or null -> NaN."""
    return np.array(
        [float(features[n]) if features.get(n) is not None else float("nan") for n in names],
        dtype=float,
    )


def _margin(bundle: dict, team_features: dict, opp_features: dict) -> float:
    """Predicted margin for a team against an opponent: the bundle's ridge on
    standardized feature differences. A feature missing for either side adds 0."""
    names = bundle["features"]
    diff = _vector(team_features, names) - _vector(opp_features, names)
    z = np.nan_to_num(diff / np.asarray(bundle["feature_sd"], dtype=float))
    return float(z @ np.asarray(bundle["coef"], dtype=float))


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Named feature map + market spread for a single team-week.

    Keys must match the bundle's feature names. Missing or null features add
    nothing to the prediction; unknown keys are rejected so typos and stale
    pipelines surface loudly instead of silently shifting the vector. The
    spread is NOT a model feature: it enters only in the final
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

    # One team on its own: its margin against a league-average opponent
    # (the training mean of every feature). Game predictions use both teams.
    average = dict(zip(bundle["features"], bundle["feature_mean"]))
    mu = _margin(bundle, req.features, average)
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
    """Margin + cover probability for one team's current game (shared with its opponent)."""
    data = load_current_features()
    if data is None:
        raise HTTPException(status_code=404, detail="Current features not available yet.")
    info = data["teams"].get(team.upper())
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown team '{team}'.")

    game = _game_prediction(data, team.upper())
    if game is not None:
        return {"season": data["season"], **game}

    # Fallback for a features file without opponent data: the team's own
    # margin estimate (the two sides of a game then need not sum to 1).
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
# Game-level prediction: the model predicts one margin per game from the
# difference between the two teams' features, so the two sides' cover
# probabilities sum to 1.
# --------------------------------------------------------------------------- #
def _game_prediction(data: dict, team: str) -> Optional[dict]:
    """Prediction for `team` against its scheduled opponent, or None if the
    features file has no opponent for it (pre-refresh format or a bye)."""
    info = data["teams"].get(team)
    opp = info.get("opponent") if info else None
    opp_info = data["teams"].get(opp) if opp else None
    if not info or not opp_info or opp_info.get("opponent") != team:
        return None

    bundle = load_model()
    manifest = load_manifest()
    if bundle is None:
        margin, scale, nu = 0.0, (manifest["scale_t"] or 13.0), (manifest["nu_t"] or 30.0)
        version, is_stub = f"stub-{manifest['version']}", True
    else:
        margin = _margin(bundle, info["features"], opp_info["features"])
        scale, nu, version, is_stub = bundle["scale_t"], bundle["nu_t"], bundle["version"], False

    from scipy import stats

    edge = margin + info["spread"]
    p_cover = float(stats.t.cdf(edge / scale, df=nu))
    return {
        "team": team,
        "opponent": opp,
        "is_home": info.get("is_home"),
        "kickoff": info.get("kickoff"),
        "week": info["week"],
        "spread": info["spread"],
        "cover_record": info["cover_record"],
        "cover_probability": round(p_cover, 4),
        "predicted_margin": round(margin, 2),
        "edge_pts": round(edge, 2),
        "model_version": version,
        "is_stub": is_stub,
    }


def _final_scores(season: int, week: int) -> dict:
    """{(away, home): (away_score, home_score)} for graded games of one week.

    The daily results job rewrites the track record with every finished game,
    so this fills in as games end. Empty when there is no record yet.
    """
    try:
        tr = track_record()
    except HTTPException:
        return {}
    if tr.get("season") != season:
        return {}
    return {
        (g["away"], g["home"]): (g["away_score"], g["home_score"])
        for g in tr.get("games", [])
        if g["week"] == week
    }


def _logged_picks(season: int, week: int) -> dict:
    """{(away, home): log entry} for the week's published picks, if any. Each
    entry carries its model's scale_t / nu_t when the record lists them."""
    try:
        tr = track_record()
    except HTTPException:
        return {}
    if tr.get("season") != season:
        return {}
    params = {m["version"]: m for m in tr.get("models", [])}
    return {
        (p["away"], p["home"]): {**p, **{k: params.get(p["model_version"], {}).get(k) for k in ("scale_t", "nu_t")}}
        for p in tr.get("logged", [])
        if p["week"] == week
    }


def _apply_logged(game: dict, entry: dict) -> None:
    """Replace a live prediction with what was published for that game.

    A pick is fixed when it is logged before kickoff. Later line moves in the
    data and later model versions do not change it.
    """
    spread_home = entry["spread_home"]
    margin_home = entry.get("margin_home", game["predicted_margin_home"])
    home_is_pick = entry["pick"] == game["home"]
    p_home = entry["p_pick"] if home_is_pick else round(1 - entry["p_pick"], 4)
    game.update({
        "spread_home": spread_home,
        "favorite": game["home"] if spread_home < 0 else game["away"] if spread_home > 0 else None,
        "spread_fav": -abs(spread_home),
        "p_home": p_home,
        "p_away": round(1 - p_home, 4),
        "predicted_margin_home": margin_home,
        "pick": entry["pick"],
        "p_pick": entry["p_pick"],
        "edge_pts": round(margin_home + spread_home if home_is_pick else -margin_home - spread_home, 2),
        "pick_model_version": entry["model_version"],
    })
    # Line parameters of the model that made the pick, when it is not the
    # current one, so a client recomputes this row with the right model.
    if entry.get("scale_t") and entry.get("nu_t"):
        game["scale_t"], game["nu_t"] = entry["scale_t"], entry["nu_t"]


def _grade(game: dict, away_score: int, home_score: int) -> dict:
    """Grade a slate game against its own spread and pick."""
    diff = home_score - away_score + game["spread_home"]
    covered = None if diff == 0 else game["home"] if diff > 0 else game["away"]
    pick_result = "push" if covered is None else "win" if covered == game["pick"] else "loss"
    return {
        "away_score": away_score,
        "home_score": home_score,
        "covered": covered,
        "pick_result": pick_result,
    }


@app.get("/slate")
def slate() -> dict:
    """Every game of the current week: both sides, one shared cover probability.

    A game whose pick is already in the log shows the logged pick, line and
    probability (`pick_model_version` names the model that made it).
    Finished games carry a `result` (final score, who covered, pick result).
    `scale_t` and `nu_t` let a client recompute the cover probability at any
    other line: P(home covers) = t.cdf((predicted_margin_home + line) / scale_t, nu_t).
    """
    data = load_current_features()
    if data is None:
        raise HTTPException(status_code=404, detail="Current features not available yet.")
    week = data["latest_week"]
    games, seen, version, is_stub = [], set(), None, False
    for team, info in data["teams"].items():
        opp = info.get("opponent")
        if not opp or info["week"] != week or frozenset((team, opp)) in seen:
            continue
        g = _game_prediction(data, team)
        if g is None:
            continue
        seen.add(frozenset((team, opp)))
        version, is_stub = g["model_version"], g["is_stub"]
        home, away = (team, opp) if g["is_home"] else (opp, team)
        p_home = g["cover_probability"] if g["is_home"] else round(1 - g["cover_probability"], 4)
        margin_home = g["predicted_margin"] if g["is_home"] else -g["predicted_margin"]
        spread_home = data["teams"][home]["spread"]
        pick = home if p_home >= 0.5 else away
        games.append({
            "away": away,
            "home": home,
            "kickoff": g["kickoff"],
            "spread_home": spread_home,
            "favorite": home if spread_home < 0 else away if spread_home > 0 else None,
            "spread_fav": -abs(spread_home),
            "p_home": p_home,
            "p_away": round(1 - p_home, 4),
            "predicted_margin_home": round(margin_home, 2),
            "pick": pick,
            "p_pick": max(p_home, round(1 - p_home, 4)),
            "edge_pts": round(margin_home + spread_home if pick == home else -margin_home - spread_home, 2),
        })
    if not games:
        raise HTTPException(
            status_code=503,
            detail="Slate unavailable: current features carry no opponent data yet.",
        )
    games.sort(key=lambda g: (g["kickoff"] is None, g["kickoff"] or "", g["home"]))
    scores = _final_scores(data["season"], week)
    logged = _logged_picks(data["season"], week)
    for g in games:
        g["pick_model_version"] = version
        entry = logged.get((g["away"], g["home"]))
        if entry:
            _apply_logged(g, entry)
        final = scores.get((g["away"], g["home"]))
        g["result"] = _grade(g, *final) if final else None
    bundle = load_model()
    manifest = load_manifest()
    return {
        "season": data["season"],
        "week": week,
        "generated_at": data.get("generated_at"),
        "model_version": version,
        "is_stub": is_stub,
        "scale_t": bundle["scale_t"] if bundle else (manifest["scale_t"] or 13.0),
        "nu_t": bundle["nu_t"] if bundle else (manifest["nu_t"] or 30.0),
        "games": games,
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
