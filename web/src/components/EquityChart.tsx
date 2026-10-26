"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WeeklyResult } from "@/lib/api";
import { formatUnits } from "@/lib/format";

const TICK = { fontSize: 11, fontFamily: "var(--font-mono)", fill: "var(--fg-3)" };

// Units change once per week, so the line steps.
export default function EquityChart({ weekly }: { weekly: WeeklyResult[] }) {
  const data = weekly.map((w) => ({ week: w.week, units: w.cumulative_units }));
  return (
    <div className="h-60 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <ReferenceLine y={0} stroke="var(--line-strong)" />
          <XAxis dataKey="week" tick={TICK} tickLine={false} axisLine={{ stroke: "var(--line)" }} />
          <YAxis tick={TICK} tickLine={false} axisLine={false} width={48} />
          <Tooltip
            cursor={{ stroke: "var(--line-strong)" }}
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--line)",
              borderRadius: 4,
              color: "var(--fg)",
              fontSize: 13,
            }}
            labelFormatter={(w) => `Week ${w}`}
            formatter={(v) => [formatUnits(v as number), "units"]}
          />
          <Line
            type="stepAfter"
            dataKey="units"
            stroke="var(--accent)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
