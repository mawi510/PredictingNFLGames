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
        <StatTile label="Win %" value={formatPct(r.win_pct)} meta="breakeven 52.4% at -110" />
        <StatTile label="Units" value={formatUnits(r.total_units)} tone={r.total_units} meta="flat 1u per pick" />
        <StatTile label="ROI" value={formatRoi(r.roi)} tone={r.roi} meta="units per pick" />
        <StatTile label="Picks" value={String(r.total_picks)} meta="one per game" />
        <StatTile label="Pushes" value={String(r.pushes)} meta="refunded, not counted" />
      </div>

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
        <Methodology methodology={tr.methodology} />
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
