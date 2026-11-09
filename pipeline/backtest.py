"""Season track record for the site: "if you had followed the published picks".

Grades the append-only picks log (pipeline/picks_log) against final scores
from the nflverse schedule. No model runs here: a pick is whatever was logged
before kickoff, at the line logged with it, by the model version logged with
it. So neither a data refresh nor a model change can rewrite the record.

Methodology (also written into the output JSON):
  * One pick per game: the side with the higher cover probability at the
    line on the day the pick was logged.
  * Pushes are excluded from the record; a sportsbook refunds them.
  * Picks start at `MIN_WEEK` (the site's "picks start Week 4" rule).
  * Units assume flat -110 bets: a win pays +0.909u, a loss -1.0u.

Runs after the Wednesday data refresh and every morning in season (daily
results job), so finals reach the site the day after games.

Writes track_record.json, served by the API's /track-record endpoint.

Usage:
    python -m pipeline.backtest
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from pipeline.margin_model import DEFAULT_BUNDLE_PATH, load_bundle, load_schedule
from pipeline.picks_log import MIN_WEEK, load_log

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "nfl_current_data.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_BUNDLE_PATH))
OUTPUT_PATH = os.getenv("TRACK_RECORD_PATH", str(ROOT / "api" / "track_record.json"))
# When set, also upload the result here so the API can serve it without a redeploy.
OUTPUT_S3_URI = os.getenv("TRACK_RECORD_S3_URI", "")

WIN_PAYOUT = 100 / 110  # flat -110 odds


def load_data() -> pd.DataFrame:
    storage_options = None
    if DATA_PATH.startswith("s3://"):
        storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
    df = pd.read_csv(DATA_PATH, storage_options=storage_options, usecols=lambda c: c.lower() in ("season", "week"))
    df.columns = df.columns.str.lower()
    return df


def build_track_record() -> dict:
    df = load_data()
    bundle = load_bundle(MODEL_PATH)
    latest_season = int(df["season"].max())
    latest_week = int(df[df["season"] == latest_season]["week"].max())

    full_log = load_log(latest_season)
    log = full_log["picks"]
    sched = load_schedule([latest_season]).dropna(subset=["margin"])
    finals = {
        (int(r.week), r.team): (int(r.points_against), int(r.points_for))  # (away, home)
        for r in sched[sched["is_home"] == 1].itertuples()
    }

    games = []  # every graded game with its final score
    for p in sorted(log, key=lambda p: (p["week"], p["home"])):
        final = finals.get((p["week"], p["home"]))
        if final is None:
            continue  # not played yet
        away_score, home_score = final
        diff = home_score - away_score + p["spread_home"]
        if diff == 0:
            result, units = "push", 0.0
        else:
            covered = p["home"] if diff > 0 else p["away"]
            result = "win" if covered == p["pick"] else "loss"
            units = WIN_PAYOUT if result == "win" else -1.0
        games.append({
            "week": p["week"], "away": p["away"], "home": p["home"],
            "away_score": away_score, "home_score": home_score,
            "spread_home": p["spread_home"], "pick": p["pick"], "p_pick": p["p_pick"],
            "model_version": p["model_version"],
            "result": result, "units": round(units, 3),
        })

    decided = [g for g in games if g["result"] != "push"]
    wins = sum(g["result"] == "win" for g in decided)
    losses = len(decided) - wins
    total_units = round(sum(g["units"] for g in decided), 2)

    # Week-by-week cumulative curve for the equity chart on the site.
    weekly = []
    cum_w = cum_l = 0
    cum_units = 0.0
    for week in sorted({g["week"] for g in decided}):
        wk = [g for g in decided if g["week"] == week]
        w = sum(g["result"] == "win" for g in wk)
        cum_w += w
        cum_l += len(wk) - w
        cum_units += sum(g["units"] for g in wk)
        weekly.append({
            "week": week, "wins": w, "losses": len(wk) - w,
            "cumulative_wins": cum_w, "cumulative_losses": cum_l,
            "cumulative_units": round(cum_units, 2),
        })

    # Which model made the picks, and from which week.
    models = []
    for p in sorted(log, key=lambda p: p["week"]):
        if not models or models[-1]["version"] != p["model_version"]:
            models.append({
                "version": p["model_version"], "from_week": p["week"],
                **full_log.get("models", {}).get(p["model_version"], {}),
            })

    return {
        "season": latest_season,
        "latest_week": latest_week,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "is_stub": bundle is None,
        "model_version": bundle["version"] if bundle else (models[-1]["version"] if models else None),
        "models": models,
        "methodology": {
            "pick_rule": "one pick per game: the side with the higher cover probability, "
                         "logged before kickoff at that day's line and never changed",
            "min_week": MIN_WEEK,
            "odds_assumption": "flat -110 (win pays +0.909u, loss -1.0u)",
            "pick_unit": "one game (pushes excluded/refunded)",
        },
        "record": {
            "wins": wins,
            "losses": losses,
            "pushes": len(games) - len(decided),
            "total_picks": len(decided),
            "win_pct": round(wins / len(decided), 4) if decided else 0.0,
            "total_units": total_units,
            "roi": round(total_units / len(decided), 4) if decided else 0.0,
        },
        "weekly": weekly,
        "games": games,
        # The current week's logged picks, played or not, so the slate shows
        # exactly what was published even after a data refresh or model change.
        "logged": [p for p in log if p["week"] == latest_week],
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
    print(f"Wrote {OUTPUT_PATH}")
    print(
        f"  {record['season']} season: {r['wins']}-{r['losses']}-{r['pushes']}p "
        f"({r['win_pct']*100:.1f}%), {r['total_units']:+.2f}u, ROI {r['roi']*100:+.1f}%"
    )


if __name__ == "__main__":
    main()
