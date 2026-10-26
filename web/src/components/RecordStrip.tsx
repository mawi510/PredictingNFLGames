"use client";

import Link from "next/link";
import type { TrackRecord } from "@/lib/api";
import { formatUnits } from "@/lib/format";

// Season W-L, units and sample size, in the same type as the picks.
export default function RecordStrip({ record }: { record: TrackRecord }) {
  const r = record.record;
  const unitsColor = r.total_units > 0 ? "text-win" : r.total_units < 0 ? "text-loss" : "text-push";
  return (
    <Link
      href="/record/"
      className="num flex items-baseline gap-2 text-[14px] text-fg-2 hover:text-fg"
    >
      <span className="font-medium text-fg">
        {r.wins}-{r.losses}
      </span>
      <span>&middot;</span>
      <span className={`font-medium ${unitsColor}`}>{formatUnits(r.total_units)}</span>
      <span>&middot;</span>
      <span>
        {r.total_picks} picks since Wk {record.methodology.min_week}
      </span>
    </Link>
  );
}
