"use client";

import Link from "next/link";
import { useState } from "react";
import type { SlateGame } from "@/lib/api";
import { formatSpread } from "@/lib/format";
import { homeCoverProbability } from "@/lib/stats";
import CoverProbability from "./CoverProbability";
import PickBadge from "./PickBadge";

const RANGE = 10; // points either side of the market line
const STEP = 0.5;

// A home handicap written the way a sportsbook shows it: on the favourite.
function lineToken(game: SlateGame, line: number): string {
  if (line === 0) return "PK";
  return line < 0 ? `${game.home} ${formatSpread(line)}` : `${game.away} ${formatSpread(-line)}`;
}

// Move the line for one game and see what the model says at that number.
// The model's margin does not change; only the line it is compared with does.
export default function LinePanel({
  game,
  scale,
  nu,
}: {
  game: SlateGame;
  scale: number;
  nu: number;
}) {
  const market = game.spread_home;
  const [line, setLine] = useState(market);
  const pHome = homeCoverProbability(game.predicted_margin_home, line, scale, nu);
  const pick = pHome >= 0.5 ? game.home : game.away;
  const pickSpread = pick === game.home ? line : -line;
  const pPick = Math.max(pHome, 1 - pHome);
  const moved = line !== market;
  const evenLine = -game.predicted_margin_home;

  const r = game.result;
  let hindsight: string | null = null;
  if (r) {
    const diff = r.home_score - r.away_score + line;
    const covered = diff === 0 ? null : diff > 0 ? game.home : game.away;
    hindsight =
      covered === null
        ? `Final ${game.away} ${r.away_score}, ${game.home} ${r.home_score}: a push at this line.`
        : `Final ${game.away} ${r.away_score}, ${game.home} ${r.home_score}: ${covered} covers at this line.`;
  }

  const step = (delta: number) =>
    setLine((cur) => Math.min(market + RANGE, Math.max(market - RANGE, cur + delta)));

  return (
    <div className="border-b border-line bg-surface-2 px-3 py-3">
      <div className="grid gap-x-8 gap-y-3 lg:grid-cols-[minmax(260px,1fr)_auto_auto] lg:items-center">
        <div>
          <div className="flex items-baseline justify-between gap-3">
            <label htmlFor={`line-${game.away}-${game.home}`} className="label text-fg-2">
              Your line
            </label>
            <span className="num font-mono text-[14px] text-fg">{lineToken(game, line)}</span>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <StepButton label={`Favour ${game.away} by half a point more`} onClick={() => step(STEP)}>
              {game.away}
            </StepButton>
            {/* The slider runs on "points the home team is favoured by", so
                left favours the away team, matching the away-then-home row. */}
            <input
              id={`line-${game.away}-${game.home}`}
              type="range"
              min={-market - RANGE}
              max={-market + RANGE}
              step={STEP}
              value={-line}
              onChange={(e) => setLine(-Number(e.target.value))}
              className="h-1.5 w-full cursor-pointer accent-[var(--accent)]"
              aria-valuetext={lineToken(game, line)}
            />
            <StepButton label={`Favour ${game.home} by half a point more`} onClick={() => step(-STEP)}>
              {game.home}
            </StepButton>
          </div>
          <div className="label mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-fg-3">
            <span>Market {lineToken(game, market)}</span>
            <span>Model {lineToken(game, evenLine)}</span>
            {moved && (
              <button type="button" onClick={() => setLine(market)} className="label text-accent">
                Reset
              </button>
            )}
          </div>
        </div>

        <div>
          <div className="label mb-1 text-fg-2">Cover prob at this line</div>
          <CoverProbability away={game.away} home={game.home} pAway={1 - pHome} pick={pick} />
        </div>

        <div>
          <div className="label mb-1 text-fg-2">Pick at this line</div>
          <PickBadge team={pick} spread={pickSpread} probability={pPick} />
        </div>
      </div>

      <p className="mt-3 text-[13px] leading-5 text-fg-2">
        The model&apos;s margin stays fixed; only the line changes. At {lineToken(game, evenLine)} the
        two sides are even, so a line better than that for a team makes it the pick.
        {hindsight && ` ${hindsight}`}
      </p>
      <p className="label mt-2 text-fg-3">
        Team stats:{" "}
        <Link href={`/teams/?team=${game.away}`} className="text-accent">
          {game.away}
        </Link>{" "}
        &middot;{" "}
        <Link href={`/teams/?team=${game.home}`} className="text-accent">
          {game.home}
        </Link>
      </p>
    </div>
  );
}

function StepButton({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={onClick}
      className="label h-7 shrink-0 rounded-[4px] border border-line px-2 text-fg-2 transition-colors duration-100 hover:bg-surface"
    >
      {children}
    </button>
  );
}
