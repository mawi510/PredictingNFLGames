"use client";

import Link from "next/link";
import type { SlateGame } from "@/lib/api";
import { formatKickoffTime, formatSpread } from "@/lib/format";
import { teamLogo, teamName } from "@/lib/teams";
import CoverProbability from "./CoverProbability";
import PickBadge from "./PickBadge";

// One-line layout from 1024px up. Team columns flex; the numeric columns are
// fixed so every number lines up down the page. Fixed widths plus gaps must
// stay under the narrowest lg viewport or the last column spills out of the
// table: 72 + 176 + 112 + 88 + 5 gaps of 16 + 2 x 140 minimum = 808px.
export const ROW_COLS =
  "lg:grid-cols-[minmax(140px,1fr)_minmax(140px,1fr)_72px_176px_112px_88px]";

export default function MatchupRow({
  game,
  showPicks,
  dimmed = false,
}: {
  game: SlateGame;
  showPicks: boolean;
  dimmed?: boolean;
}) {
  const pickIsHome = game.pick === game.home;
  const pickSpread = pickIsHome ? game.spread_home : -game.spread_home;
  // The model's expected final margin, written as a line on the team it
  // favours so it reads directly against the market spread beside it.
  const modelFav = game.predicted_margin_home >= 0 ? game.home : game.away;
  const modelLine = -Math.abs(game.predicted_margin_home);

  return (
    <Link
      href={`/teams/?team=${encodeURIComponent(showPicks ? game.pick : game.home)}`}
      className={`grid min-h-16 grid-cols-2 items-center gap-x-4 gap-y-1 border-b border-line px-3 py-2 transition-colors duration-100 hover:bg-surface-2 lg:min-h-11 lg:py-0 ${ROW_COLS} ${
        dimmed ? "opacity-60" : ""
      }`}
    >
      <Team abbr={game.away} />
      <Team abbr={game.home} home />

      <div className="label col-span-2 flex items-center gap-3 text-fg-2 lg:hidden">
        <span>{game.kickoff ? formatKickoffTime(game.kickoff) : "TBD"}</span>
        <SpreadToken game={game} />
      </div>

      <div className="hidden whitespace-nowrap text-right font-mono text-[13px] text-fg lg:block">
        <SpreadToken game={game} />
      </div>

      <div>
        {showPicks ? (
          <CoverProbability away={game.away} home={game.home} pAway={game.p_away} pick={game.pick} />
        ) : (
          <span className="text-fg-3">-</span>
        )}
      </div>

      <div className="text-right lg:text-left">
        {showPicks && (
          <PickBadge team={game.pick} spread={pickSpread} probability={game.p_pick} />
        )}
      </div>

      <div className="num hidden whitespace-nowrap text-right font-mono text-[13px] text-fg-2 lg:block">
        {showPicks ? (modelLine === 0 ? "PK" : `${modelFav} ${formatSpread(modelLine)}`) : "-"}
      </div>
    </Link>
  );
}

function SpreadToken({ game }: { game: SlateGame }) {
  if (!game.favorite) return <span className="font-mono text-[13px] text-fg">PK</span>;
  return (
    <span className="font-mono text-[13px] text-fg">
      {game.favorite} {formatSpread(game.spread_fav)}
    </span>
  );
}

function Team({ abbr, home = false }: { abbr: string; home?: boolean }) {
  const logo = teamLogo(abbr);
  return (
    <div className="flex min-w-0 items-center gap-2 text-[14px]">
      {logo && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logo} alt="" width={20} height={20} className="h-5 w-5 shrink-0 object-contain" />
      )}
      <span className="truncate">
        <span className="hidden lg:inline">{teamName(abbr)}</span>
        <span className="lg:hidden">{abbr}</span>
      </span>
      {home && <span className="label hidden text-fg-3 lg:inline">home</span>}
    </div>
  );
}
