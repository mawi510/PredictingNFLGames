"""Append-only log of the picks the site publishes.

One entry per game, written by the Wednesday job before kickoff and never
changed afterwards. The track record is graded from this log, so a later data
refresh (closing lines replace Wednesday lines) or a model change cannot
rewrite what was published. Git history of the log file is the proof of when
each pick was made.

A game is logged only if it is in the data file's latest week, is not in the
log yet, and has not been played. A late run therefore cannot add a pick for
a game that already kicked off.

Usage:
    python -m pipeline.picks_log
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from pipeline.margin_model import (
    DEFAULT_BUNDLE_PATH,
    attach_matchup_features,
    load_bundle,
    load_schedule,
    pair_games,
    predict_cover,
)

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "pipeline" / "picks_log"
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "nfl_current_data.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_BUNDLE_PATH))
MIN_WEEK = int(os.getenv("MIN_WEEK", "4"))  # matches the site's "picks start Week 4"


def log_path(season: int) -> Path:
    return LOG_DIR / f"{season}.json"


def load_log(season: int) -> dict:
    path = log_path(season)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "season": season,
        "note": "One entry per game, written before kickoff and never changed.",
        "models": {},
        "picks": [],
    }


def week_picks(week_df: pd.DataFrame, bundle: dict) -> list[dict]:
    """The model's pick for every game in one week's team rows."""
    week_df = week_df.reset_index(drop=True).copy()
    week_df["margin"], week_df["p"] = predict_cover(week_df, bundle)
    picks = []
    for i, j in pair_games(week_df):
        ri, rj = week_df.iloc[i], week_df.iloc[j]
        home, away = (ri, rj) if int(ri["is_home"]) == 1 else (rj, ri)
        side = home if float(home["p"]) >= 0.5 else away
        picks.append({
            "week": int(home["week"]),
            "away": str(away["team"]),
            "home": str(home["team"]),
            "spread_home": float(home["spread"]),
            "margin_home": round(float(home["margin"]), 2),  # the model's line when logged
            "pick": str(side["team"]),
            "p_pick": round(float(side["p"]), 4),
            "model_version": bundle["version"],
        })
    return picks


def main() -> None:
    bundle = load_bundle(MODEL_PATH)
    if bundle is None:
        print("No model bundle; nothing logged.")
        return
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.lower()
    season = int(df["season"].max())
    week = int(df[df["season"] == season]["week"].max())
    if week < MIN_WEEK:
        print(f"Week {week} is before week {MIN_WEEK}; nothing logged.")
        return

    log = load_log(season)
    logged = {(p["week"], p["away"], p["home"]) for p in log["picks"]}
    sched = load_schedule([season])
    played = {
        (int(r.week), r.team) for r in sched.dropna(subset=["margin"]).itertuples() if r.is_home == 1
    }
    rows = attach_matchup_features(df[(df["season"] == season) & (df["week"] == week)])
    now = pd.Timestamp.now("UTC").isoformat()
    new = [
        {**p, "logged_at": now}
        for p in week_picks(rows, bundle)
        if (p["week"], p["away"], p["home"]) not in logged and (p["week"], p["home"]) not in played
    ]
    if not new:
        print(f"Week {week}: nothing new to log ({len(logged)} picks already in the log).")
        return
    log["picks"].extend(sorted(new, key=lambda p: p["home"]))
    # The probability parameters of the model that made these picks, so the
    # site can recompute a logged pick at another line with its own model.
    log.setdefault("models", {})[bundle["version"]] = {
        "scale_t": bundle["scale_t"], "nu_t": bundle["nu_t"],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path(season).write_text(json.dumps(log, indent=1))
    print(f"Week {week}: logged {len(new)} picks with {bundle['version']} -> {log_path(season)}")


if __name__ == "__main__":
    main()
