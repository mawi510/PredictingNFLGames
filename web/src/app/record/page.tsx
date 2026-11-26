"use client";

import CalibrationView from "@/components/CalibrationView";
import EquityChart from "@/components/EquityChart";
import Methodology from "@/components/Methodology";
import StatTile from "@/components/StatTile";
import WeeklyResultsTable from "@/components/WeeklyResultsTable";
import { useAppData } from "@/lib/data";
import { formatPct, formatRoi, formatUnits, formatUpdated } from "@/lib/format";

export default function RecordPage() {
  const { record } = useAppData();

  if (record.loading) {
    return <h1 className="text-[20px] font-semibold leading-tight">Track record</h1>;
  }
  const tr = record.data;
  if (!tr) {
    return (
      <>
        <h1 className="mb-4 text-[20px] font-semibold leading-tight">Track record</h1>
        <p className="text-[13px] text-fg-2">No graded picks yet this season.</p>
      </>
    );
  }
  const r = tr.record;
  const since = `since Wk ${tr.methodology.min_week}`;

  return (
    <>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h1 className="text-[20px] font-semibold leading-tight">{tr.season} track record</h1>
        <span className="label text-fg-3">
          through week {tr.latest_week} &middot; updated {formatUpdated(tr.generated_at)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <StatTile label="Record" value={`${r.wins}-${r.losses}`} meta={since} />
        <StatTile label="Win %" value={formatPct(r.win_pct, 1)} meta="breakeven 52.4% at -110" />
        <StatTile label="Units" value={formatUnits(r.total_units)} tone={r.total_units} meta="flat 1u per pick" />
        <StatTile label="ROI" value={formatRoi(r.roi)} tone={r.roi} meta="units per pick" />
        <StatTile label="Picks" value={String(r.total_picks)} meta="one per game" />
        <StatTile label="Pushes" value={String(r.pushes)} meta="refunded, not counted" />
      </div>

      {tr.analyst && (
        <Section title="Analyst picks">
          <p className="mb-3 max-w-[760px] text-[13px] leading-5 text-fg-2">
            Since week {tr.analyst.from_week}, Claude also researches each game on the web (injuries,
            weather, rest, line moves) and logs its own pick before kickoff, graded the same way at
            its own line. It cannot be backtested, so this record is the only evidence. A few dozen
            picks cannot separate skill from luck.
          </p>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatTile
              label="Analyst"
              value={`${tr.analyst.record.wins}-${tr.analyst.record.losses}`}
              meta={`${tr.analyst.record.pushes} pushes`}
            />
            <StatTile
              label="Analyst units"
              value={formatUnits(tr.analyst.record.total_units)}
              tone={tr.analyst.record.total_units}
              meta="flat 1u per pick"
            />
            <StatTile
              label="Model, same games"
              value={`${tr.analyst.model_same_games.wins}-${tr.analyst.model_same_games.losses}`}
              meta={`${tr.analyst.model_same_games.pushes} pushes`}
            />
            <StatTile
              label="Model units"
              value={formatUnits(tr.analyst.model_same_games.total_units)}
              tone={tr.analyst.model_same_games.total_units}
              meta="same games"
            />
          </div>
        </Section>
      )}

      <Section title="Units by week">
        <EquityChart weekly={tr.weekly} />
      </Section>

      <Section title="Weekly results">
        <div className="rounded-[4px] border border-line bg-surface">
          <WeeklyResultsTable weekly={tr.weekly} />
        </div>
      </Section>

      <Section title="Calibration">
        <CalibrationView bins={tr.calibration} picks={r.total_picks} brier={tr.brier} />
      </Section>

      <Section title="Methodology">
        <Methodology methodology={tr.methodology} models={tr.models} />
      </Section>
    </>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-8">
      <h2 className="label mb-3 text-fg-2">{title}</h2>
      {children}
    </section>
  );
}
