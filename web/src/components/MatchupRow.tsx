"use client";

import { useState } from "react";
import type { SlateGame } from "@/lib/api";
import { formatKickoffTime, formatSpread } from "@/lib/format";
import { teamLogo, teamName } from "@/lib/teams";
import CoverProbability from "./CoverProbability";
import LinePanel from "./LinePanel";
import PickBadge from "./PickBadge";

// One-line layout from 1024px up. Team columns flex; the numeric columns are
// fixed so every number lines up down the page. Fixed widths plus gaps must
// stay under the narrowest lg viewport or the last column spills out of the
// table: 72 + 196 + 128 + 88 + 5 gaps of 16 + 2 x 140 minimum = 844px.
export const ROW_COLS =
  "lg:grid-cols-[minmax(140px,1fr)_minmax(140px,1fr)_72px_196px_128px_88px]";

export interface LineModel {
  scale: number;
  nu: number;
}

export default function MatchupRow({
  game,
  showPicks,
  dimmed = false,
  lineModel,
}: {
  game: SlateGame;
  showPicks: boolean;
  dimmed?: boolean;
  lineModel: LineModel | null;
}) {
  const [open, setOpen] = useState(false);
  const pickIsHome = game.pick === game.home;
  const pickSpread = pickIsHome ? game.spread_home : -game.spread_home;
  // The model's expected final margin, written as a line on the team it
  // favours so it reads directly against the market spread beside it.
  const modelFav = game.predicted_margin_home >= 0 ? game.home : game.away;
  const modelLine = -Math.abs(game.predicted_margin_home);
  const r = game.result ?? null;
  const canOpen = showPicks && lineModel !== null;
  const toggle = () => setOpen((o) => !o);

  return (
    <div className={dimmed ? "opacity-60" : ""}>
      <div
        {...(canOpen
          ? {
              role: "button",
              tabIndex: 0,
              "aria-expanded": open,
              onClick: toggle,
              onKeyDown: (e: React.KeyboardEvent) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  toggle();
                }
              },
            }
          : {})}
        className={`grid min-h-16 grid-cols-2 items-center gap-x-4 gap-y-1 border-b border-line px-3 py-2 transition-colors duration-100 lg:min-h-11 lg:py-0 ${ROW_COLS} ${
          canOpen ? "cursor-pointer hover:bg-surface-2" : ""
        } ${open ? "bg-surface-2" : ""}`}
      >
        <Team abbr={game.away} score={r?.away_score} won={r ? r.away_score > r.home_score : undefined} />
        <Team abbr={game.home} home score={r?.home_score} won={r ? r.home_score > r.away_score : undefined} />

        <div className="label col-span-2 flex items-center gap-3 text-fg-2 lg:hidden">
          <span>{r ? "Final" : game.kickoff ? formatKickoffTime(game.kickoff) : "TBD"}</span>
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
            <PickBadge
              team={game.pick}
              spread={pickSpread}
              probability={game.p_pick}
              result={r?.pick_result}
            />
          )}
        </div>

        <div className="num hidden whitespace-nowrap text-right font-mono text-[13px] text-fg-2 lg:block">
          {showPicks ? (modelLine === 0 ? "PK" : `${modelFav} ${formatSpread(modelLine)}`) : "-"}
        </div>
      </div>

      {open && lineModel && <LinePanel game={game} scale={lineModel.scale} nu={lineModel.nu} />}
    </div>
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

function Team({
  abbr,
  home = false,
  score,
  won,
}: {
  abbr: string;
  home?: boolean;
  score?: number;
  won?: boolean;
}) {
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
      {score !== undefined && (
        <span className={`num ml-auto font-mono text-[14px] ${won ? "font-medium text-fg" : "text-fg-2"}`}>
          {score}
        </span>
      )}
    </div>
  );
}
