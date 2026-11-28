"""Analyst picks: Claude researches each game and picks a side, before kickoff.

A second track beside the model. The model sees only team form; the analyst
also reads injury reports, weather, rest and line moves on the web. Each run
logs the games that kick off within WINDOW_HOURS and are not logged yet, so a
daily run reads every game 4 to 30 hours before kickoff (Sunday games after
the Friday injury report). Entries are append-only and graded like the
model's picks, at the line logged with them.

The analyst cannot be backtested: the model may know past results. It can
only be judged on games logged before they are played.

Usage:
    python -m pipeline.analyst             # log games kicking off soon
    python -m pipeline.analyst --dry-run --limit 1   # print one read, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from pipeline.margin_model import TEAM_ALIASES, _kickoff_iso
from pipeline.picks_log import LOG_DIR, MIN_WEEK, load_log

MODEL = os.getenv("ANALYST_MODEL", "claude-opus-5-5")
WINDOW_HOURS = float(os.getenv("WINDOW_HOURS", "30"))
MAX_SEARCHES = 8  # per game; caps the web search fee

FLAGS = ["qb_out", "key_injuries", "weather", "rest_or_travel", "key_number", "line_moved", "public_side"]

SYSTEM = """You are an NFL handicapper picking one game against the spread for a public picks site.
Every pick is published before kickoff and graded honestly, so a careful read matters more than a bold one.

Research the game with web search, then call record_pick exactly once. Check:
- Injuries: the latest injury report and practice status, expected inactives, and who replaces them.
  Quarterback and offensive line news matters most.
- Weather for outdoor stadiums: wind above about 15 mph and heavy rain or snow change games.
- Schedule: rest days, short weeks, travel, international games, letdown and lookahead spots.
- Matchup edges: trenches, secondary against the passing game, pace, turnovers.
- Line movement since the open, and the public betting split. A lopsided public side is a caution
  flag, not a reason to fade.
- Key numbers: 3 and 7. A spread exactly on 3 or 7 makes a push likely; a hook (2.5, 3.5, 6.5, 7.5)
  matters.

Rules:
- Handicap the game on its merits first. Use line value only to break a tie between sides you rate
  close. Half a point is worth about two percentage points of cover probability, not more.
- Pick a side at the market spread given in the message, even when the game is close.
- Use only news about this season and this game. Ignore stale or unrelated pages.
- Our statistical model's view is one input. Disagree with it when the research says so.
- Confidence: 1 is a coin flip, 3 is a real lean with a concrete reason, 5 is rare and needs a
  clear edge such as a starting quarterback out that the line has not caught up to."""

RECORD_PICK = {
    "name": "record_pick",
    "description": "Record your final pick for this game. Call it once, after your research.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "pick": {"type": "string", "enum": ["away", "home"],
                     "description": "The side you take against the spread."},
            "confidence": {"type": "integer", "enum": [1, 2, 3, 4, 5],
                           "description": "1 = coin flip, 5 = strongest edge."},
            "reason": {"type": "string",
                       "description": "One plain sentence, at most 30 words: the main reason for the pick."},
            "flags": {"type": "array", "items": {"type": "string", "enum": FLAGS},
                      "description": "Factors that shaped this game. Empty if none."},
        },
        "required": ["pick", "confidence", "reason", "flags"],
        "additionalProperties": False,
    },
}

WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES}


def log_path(season: int) -> Path:
    return LOG_DIR / f"{season}_analyst.json"


def load_analyst_log(season: int) -> dict:
    path = log_path(season)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "season": season,
        "note": "Claude's research picks. One entry per game, written before kickoff and never changed.",
        "picks": [],
    }


def upcoming_games(season: int, now: pd.Timestamp) -> pd.DataFrame:
    """Regular-season games with a spread that kick off within the window."""
    from nflverse_compat import import_schedules

    s = import_schedules([season])
    s = s[(s["game_type"] == "REG") & s["home_score"].isna() & s["spread_line"].notna()].copy()
    for col in ("home_team", "away_team"):
        s[col] = s[col].replace(TEAM_ALIASES)
    s["kickoff"] = _kickoff_iso(s["gameday"], s["gametime"])
    ko = pd.to_datetime(s["kickoff"], utc=True)
    return s[(ko > now) & (ko <= now + pd.Timedelta(hours=WINDOW_HOURS)) & (s["week"] >= MIN_WEEK)]


def game_prompt(g, model_entry: dict | None, now: pd.Timestamp) -> str:
    # nflverse spread_line is the home team's expected margin; the site's spread is its negative.
    spread_home = -float(g.spread_line)
    kickoff = pd.Timestamp(g.kickoff).strftime("%a %b %-d, %-I:%M%p ET")
    rest = "unknown" if pd.isna(g.away_rest) or pd.isna(g.home_rest) else \
        f"{g.away_team} {g.away_rest:.0f}, {g.home_team} {g.home_rest:.0f}"
    lines = [
        f"{g.season} week {g.week}: {g.away_team} at {g.home_team}.",
        f"Kickoff: {kickoff}. Stadium: {g.stadium} "
        f"(roof: {g.roof if pd.notna(g.roof) else 'unknown'}, surface: {g.surface}).",
        f"Divisional game: {'yes' if g.div_game == 1 else 'no'}. Rest days: {rest}.",
        f"Market spread now: {g.home_team} {spread_home:+g}, so {g.away_team} {-spread_home:+g}.",
    ]
    if model_entry:
        m = model_entry["margin_home"]
        fav = g.home_team if m >= 0 else g.away_team
        lines.append(
            f"Our statistical model expects {fav} by {abs(m):.1f}. It took {model_entry['pick']} at "
            f"Wednesday's line ({g.home_team} {model_entry['spread_home']:+g}) with a "
            f"{model_entry['p_pick']:.1%} cover probability. It uses team form only: no injuries, "
            "weather or line moves."
        )
    lines.append(f"It is now {now.tz_convert('America/New_York'):%a %b %-d, %-I:%M%p} ET. "
                 "Research the game, then call record_pick.")
    return "\n".join(lines)


def read_game(client, prompt: str) -> tuple[dict, object, list[str], dict]:
    """Run the research turn until Claude calls record_pick. Returns
    (pick input, final response, source urls, usage totals)."""
    messages = [{"role": "user", "content": prompt}]
    sources: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "web_searches": 0}
    for _ in range(6):
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=[WEB_SEARCH, RECORD_PICK],
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        )
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        stu = getattr(resp.usage, "server_tool_use", None)
        usage["web_searches"] += getattr(stu, "web_search_requests", 0) or 0
        for block in resp.content:
            # A failed search returns an error object instead of a list.
            if block.type == "web_search_tool_result" and isinstance(block.content, list):
                sources += [r.url for r in block.content if r.url not in sources]
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused: {resp.stop_details}")
        call = next((b for b in resp.content if b.type == "tool_use" and b.name == "record_pick"), None)
        if call is not None:
            return call.input, resp, sources, usage
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "pause_turn":
            # The server resumes a paused turn by itself; otherwise ask for the pick.
            messages.append({"role": "user", "content": "Call record_pick with your pick now."})
    raise RuntimeError("no record_pick call after 6 requests")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the reads, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="read at most this many games")
    args = ap.parse_args()

    import anthropic

    now = pd.Timestamp.now("UTC")
    season = now.year if now.month >= 3 else now.year - 1
    log = load_analyst_log(season)
    logged = {(p["week"], p["away"], p["home"]) for p in log["picks"]}
    model_log = {(p["week"], p["away"], p["home"]): p for p in load_log(season)["picks"]}

    games = [g for g in upcoming_games(season, now).itertuples()
             if (int(g.week), g.away_team, g.home_team) not in logged]
    if args.limit:
        games = games[: args.limit]
    if not games:
        print(f"No unlogged games kick off in the next {WINDOW_HOURS:g} hours.")
        return

    client = anthropic.Anthropic()
    total = {"input_tokens": 0, "output_tokens": 0, "web_searches": 0}
    failed = []
    for g in games:
        key = (int(g.week), g.away_team, g.home_team)
        prompt = game_prompt(g, model_log.get(key), now)
        try:
            pick, resp, sources, usage = read_game(client, prompt)
        except Exception as exc:  # one bad game must not lose the others' picks
            print(f"{g.away_team} at {g.home_team}: no pick ({exc})")
            failed.append(f"{g.away_team} at {g.home_team}")
            continue
        for k in total:
            total[k] += usage[k]
        entry = {
            "week": key[0], "away": key[1], "home": key[2],
            "kickoff": g.kickoff,
            "spread_home": -float(g.spread_line),
            "pick": key[1] if pick["pick"] == "away" else key[2],
            "confidence": pick["confidence"],
            "reason": pick["reason"],
            "flags": pick["flags"],
            "sources": sources[:5],
            "model": resp.model,
            "logged_at": pd.Timestamp.now("UTC").isoformat(),
        }
        print(json.dumps(entry, indent=1) if args.dry_run else
              f"{g.away_team} at {g.home_team}: {entry['pick']} ({entry['confidence']}/5) {entry['reason']}")
        if not args.dry_run:
            log["picks"].append(entry)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path(season).write_text(json.dumps(log, indent=1))  # after each game, so a crash keeps the rest
    print(f"Usage: {total['input_tokens']:,} input tokens, {total['output_tokens']:,} output tokens, "
          f"{total['web_searches']} web searches.")
    if failed:
        # Fail the job so a missing pick shows in GitHub; the workflow still commits the rest.
        raise SystemExit(f"No pick for {len(failed)} of {len(games)} games: {', '.join(failed)}")


if __name__ == "__main__":
    main()
