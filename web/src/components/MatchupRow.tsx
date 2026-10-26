"use client";

import Link from "next/link";
import type { SlateGame } from "@/lib/api";
import { formatKickoffTime, formatMargin, formatSpread } from "@/lib/format";
import { teamLogo, teamName } from "@/lib/teams";
import PickBadge from "./PickBadge";
import ProbabilityBar from "./ProbabilityBar";

// Desktop column widths. Team columns flex; the numeric columns are fixed so
// every number lines up down the page.
export const ROW_COLS =
  "md:grid-cols-[minmax(160px,1fr)_minmax(160px,1fr)_80px_200px_120px_80px]";

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
  const marginTeam = game.predicted_margin_home >= 0 ? game.home : game.away;
  const marginAbs = Math.abs(game.predicted_margin_home);

  return (
    <Link
      href={`/teams/?team=${encodeURIComponent(showPicks ? game.pick : game.home)}`}
      className={`grid min-h-16 grid-cols-2 items-center gap-x-6 gap-y-1 border-b border-line px-3 py-2 transition-colors duration-100 hover:bg-surface-2 md:min-h-11 md:py-0 ${ROW_COLS} ${
        dimmed ? "opacity-60" : ""
      }`}
    >
      <Team abbr={game.away} />
      <Team abbr={game.home} home />

      <div className="label col-span-2 flex items-center gap-3 text-fg-2 md:hidden">
        <span>{game.kickoff ? formatKickoffTime(game.kickoff) : "TBD"}</span>
        <SpreadToken game={game} />
      </div>

      <div className="hidden text-right font-mono text-[13px] text-fg md:block">
        <SpreadToken game={game} />
      </div>

      <div className="col-span-1 md:col-span-1">
        {showPicks ? (
          <ProbabilityBar pAway={game.p_away} pickSide={pickIsHome ? "home" : "away"} />
        ) : (
          <span className="text-fg-3">-</span>
        )}
      </div>

      <div className="text-right md:text-left">
        {showPicks && (
          <PickBadge team={game.pick} spread={pickSpread} probability={game.p_pick} />
        )}
      </div>

      <div className="num hidden text-right font-mono text-[13px] text-fg-2 md:block">
        {showPicks ? `${marginTeam} ${formatMargin(marginAbs)}` : "-"}
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
      {home && <span className="label hidden text-fg-3 md:inline">home</span>}
    </div>
  );
}
