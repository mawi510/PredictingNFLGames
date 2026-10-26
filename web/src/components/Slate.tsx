"use client";

import type { Slate as SlateData, SlateGame } from "@/lib/api";
import { formatKickoffDay } from "@/lib/format";
import MatchupRow, { ROW_COLS } from "./MatchupRow";

const HEADERS: { label: string; align?: "right" }[] = [
  { label: "Away" },
  { label: "Home" },
  { label: "Spread", align: "right" },
  { label: "Cover prob" },
  { label: "Pick" },
  { label: "Model line", align: "right" },
];

function groupByDay(games: SlateGame[]): { day: string; games: SlateGame[] }[] {
  const groups: { day: string; games: SlateGame[] }[] = [];
  for (const g of games) {
    const day = g.kickoff ? formatKickoffDay(g.kickoff) : "TBD";
    const last = groups[groups.length - 1];
    if (last && last.day === day) last.games.push(g);
    else groups.push({ day, games: [g] });
  }
  return groups;
}

export default function Slate({
  slate,
  showPicks,
  dimmed = false,
}: {
  slate: SlateData;
  showPicks: boolean;
  dimmed?: boolean;
}) {
  return (
    <div className="rounded-[4px] border border-line bg-surface">
      <div className={`label hidden h-8 items-center gap-x-4 border-b border-line px-3 text-fg-2 lg:grid ${ROW_COLS}`}>
        {HEADERS.map((h) => (
          <span key={h.label} className={h.align === "right" ? "text-right" : ""}>
            {h.label}
          </span>
        ))}
      </div>
      {groupByDay(slate.games).map((group) => (
        <section key={group.day}>
          <h2 className="label flex h-8 items-center border-b border-line bg-surface-2 px-3 text-fg-2">
            {group.day}
          </h2>
          {group.games.map((g) => (
            <MatchupRow
              key={`${g.away}-${g.home}`}
              game={g}
              showPicks={showPicks}
              dimmed={dimmed}
            />
          ))}
        </section>
      ))}
    </div>
  );
}

// Static blocks in place of rows while the slate loads. No shimmer.
export function SlateLoading({ rows = 14 }: { rows?: number }) {
  return (
    <div className="rounded-[4px] border border-line bg-surface" aria-busy="true">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-11 border-b border-line px-3 py-3 last:border-b-0">
          <div className="h-5 w-full bg-surface-2" />
        </div>
      ))}
    </div>
  );
}
