"use client";

import HowToRead from "@/components/HowToRead";
import RecordStrip from "@/components/RecordStrip";
import Slate, { SlateLoading } from "@/components/Slate";
import StatTile from "@/components/StatTile";
import type { SlateGame } from "@/lib/api";
import { useAppData } from "@/lib/data";
import { formatPct, formatUnits, formatUpdatedDay } from "@/lib/format";

export default function Home() {
  const { status, slate, record } = useAppData();
  const s = status.data;
  const minWeek = record.data?.methodology.min_week ?? 4;

  if (status.loading || slate.loading) {
    return (
      <>
        <PageHeading title="" />
        <SlateLoading />
      </>
    );
  }

  // API down and nothing cached: one line, no advice.
  if (!s && !slate.data) {
    const lastGood = record.data?.generated_at;
    return (
      <p className="text-[13px] text-fg-2">
        The model API is not responding.
        {lastGood && ` Last good update: ${titleCase(formatUpdatedDay(lastGood))}.`} Try
        again in a minute.
      </p>
    );
  }

  const offSeason = s ? !s.in_season : false;
  const showPicks = s ? s.predictions_available : true;
  const week = slate.data?.week ?? s?.latest_week;

  if (offSeason) {
    const r = record.data?.record;
    const season = record.data?.season ?? s?.season;
    return (
      <>
        <PageHeading title={`${season} final`} />
        {r && (
          <div className="mb-6 grid grid-cols-2 gap-6 sm:grid-cols-3">
            <StatTile label="Record" value={`${r.wins}-${r.losses}`} meta={`${r.total_picks} picks`} />
            <StatTile label="Units" value={formatUnits(r.total_units)} tone={r.total_units} />
            <StatTile label="Win %" value={formatPct(r.win_pct)} meta={`since Wk ${minWeek}`} />
          </div>
        )}
        <p className="mb-6 text-[13px] text-fg-2">
          Off season.
          {r && ` ${season} final: ${r.wins}-${r.losses}, ${formatUnits(r.total_units)} over ${r.total_picks} picks.`}{" "}
          Next slate: Week {minWeek}, {(season ?? 0) + 1}.
        </p>
        {slate.data && <Slate slate={slate.data} showPicks dimmed />}
      </>
    );
  }

  return (
    <>
      <PageHeading title={`Week ${week} slate`}>
        {record.data && <RecordStrip record={record.data} />}
      </PageHeading>

      {!showPicks && (
        <p className="mb-4 text-[13px] text-fg-2">
          Picks start Week {minWeek}. The model needs three weeks of results.
          Spreads below are the current market.
        </p>
      )}

      {slate.data && slate.data.games.length > 0 ? (
        <Slate slate={slate.data} showPicks={showPicks} dimmed={slate.stale} />
      ) : (
        <p className="text-[13px] text-fg-2">No slate yet. Picks start Week {minWeek}.</p>
      )}

      {slate.data && showPicks && slate.data.games.length > 0 && (
        <HowToRead game={largestGap(slate.data.games)} modelVersion={slate.data.model_version} />
      )}
    </>
  );
}

function PageHeading({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-4 flex min-h-7 flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
      <h1 className="text-[20px] font-semibold leading-tight">{title}</h1>
      {children}
    </div>
  );
}

// The clearest example is the game where the model and the market disagree most.
function largestGap(games: SlateGame[]): SlateGame {
  return games.reduce((best, g) => (Math.abs(g.edge_pts) > Math.abs(best.edge_pts) ? g : best), games[0]);
}

function titleCase(upper: string): string {
  return upper
    .toLowerCase()
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
