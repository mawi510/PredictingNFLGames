"use client";

import type { Slate as SlateData, SlateGame } from "@/lib/api";
import { formatKickoffDay, formatKickoffTime, kickoffHourET } from "@/lib/format";
import MatchupRow, { ROW_COLS, type LineModel } from "./MatchupRow";

const HEADERS: { label: string; align?: "right" }[] = [
  { label: "Away" },
  { label: "Home" },
  { label: "Spread", align: "right" },
  { label: "Cover prob" },
  { label: "Pick" },
  { label: "Model line", align: "right" },
];

interface Window {
  key: string;
  label: string;
  games: SlateGame[];
}

// Games in the order they air, one block per broadcast window: Thursday night,
// Sunday early, Sunday late afternoon, Sunday night, Monday night. Kickoffs in
// the same Eastern hour (4:05 and 4:25) share a window. The API already sorts
// by kickoff; this only draws the boundaries.
function groupByWindow(games: SlateGame[]): Window[] {
  const windows: Window[] = [];
  for (const g of games) {
    const key = g.kickoff ? `${formatKickoffDay(g.kickoff)}-${kickoffHourET(g.kickoff)}` : "TBD";
    const last = windows[windows.length - 1];
    if (last && last.key === key) {
      last.games.push(g);
    } else {
      const label = g.kickoff
        ? `${formatKickoffDay(g.kickoff)} · ${formatKickoffTime(g.kickoff)}`
        : "TBD";
      windows.push({ key, label, games: [g] });
    }
  }
  return windows;
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
  const lineModel: LineModel | null =
    slate.scale_t && slate.nu_t ? { scale: slate.scale_t, nu: slate.nu_t } : null;
  return (
    <div className="rounded-[4px] border border-line bg-surface">
      <div className={`label hidden h-8 items-center gap-x-4 border-b border-line px-3 text-fg-2 lg:grid ${ROW_COLS}`}>
        {HEADERS.map((h) => (
          <span key={h.label} className={h.align === "right" ? "text-right" : ""}>
            {h.label}
          </span>
        ))}
      </div>
      {groupByWindow(slate.games).map((w) => (
        <section key={w.key}>
          <h2 className="label flex h-8 items-center border-b border-line bg-surface-2 px-3 text-fg-2">
            {w.label}
          </h2>
          {w.games.map((g) => (
            <MatchupRow
              key={`${g.away}-${g.home}`}
              game={g}
              showPicks={showPicks}
              dimmed={dimmed}
              lineModel={lineModel}
            />
          ))}
        </section>
      ))}
      {showPicks && lineModel && (
        <p className="label px-3 py-2 text-fg-3">Select a game to test a different line.</p>
      )}
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
