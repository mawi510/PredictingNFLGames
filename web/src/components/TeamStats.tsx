"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getTeamStats, type TeamStats as Stats } from "@/lib/api";
import SegmentedControl from "./SegmentedControl";

const TICK = { fontSize: 11, fontFamily: "var(--font-mono)", fill: "var(--fg-3)" };

interface Loaded {
  forTeam: string;
  stats: Stats | null;
  failed: boolean;
}

export default function TeamStats({ team }: { team: string }) {
  // Keyed by team so a team change shows the loading block without a reset
  // effect; the old team's stats are simply not for this team.
  const [loaded, setLoaded] = useState<Loaded>({ forTeam: "", stats: null, failed: false });
  const [season, setSeason] = useState("");
  const [metric, setMetric] = useState("");
  const stats = loaded.forTeam === team ? loaded.stats : null;
  const failed = loaded.forTeam === team && loaded.failed;

  useEffect(() => {
    if (!team) return;
    let active = true;
    getTeamStats(team)
      .then((s) => {
        if (!active) return;
        setLoaded({ forTeam: team, stats: s, failed: false });
        setSeason(String(s.seasons[0] ?? ""));
        setMetric(s.metrics[0]?.key ?? "");
      })
      .catch(() => {
        if (active) setLoaded({ forTeam: team, stats: null, failed: true });
      });
    return () => {
      active = false;
    };
  }, [team]);

  // Metrics that actually have data for the chosen season; if the chosen
  // metric is not among them, fall back to the first one that is.
  const availableMetrics = useMemo(() => {
    if (!stats || !season) return stats?.metrics ?? [];
    const present = stats.series[season]?.metrics ?? {};
    return stats.metrics.filter((m) => m.key in present);
  }, [stats, season]);
  const activeMetric = availableMetrics.some((m) => m.key === metric)
    ? metric
    : (availableMetrics[0]?.key ?? "");

  const chartData = useMemo(() => {
    if (!stats || !season || !activeMetric) return [];
    const s = stats.series[season];
    if (!s || !(activeMetric in s.metrics)) return [];
    return s.weeks.map((week, i) => ({ week, value: s.metrics[activeMetric][i] }));
  }, [stats, season, activeMetric]);

  const metricLabel = stats?.metrics.find((m) => m.key === activeMetric)?.label ?? "";

  if (failed) {
    return <p className="text-[13px] text-fg-2">Stats for {team} are not available right now.</p>;
  }
  if (!stats) {
    return <div className="h-60 w-full bg-surface-2" aria-busy="true" />;
  }

  return (
    <section>
      <div className="mb-4 flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="label w-14 text-fg-3">Season</span>
          <SegmentedControl
            label="Season"
            options={stats.seasons.map((s) => ({ value: String(s), label: String(s) }))}
            value={season}
            onChange={setSeason}
          />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="label w-14 text-fg-3">Metric</span>
          <SegmentedControl
            label="Metric"
            options={availableMetrics.map((m) => ({ value: m.key, label: m.label }))}
            value={activeMetric}
            onChange={setMetric}
          />
        </div>
      </div>

      <div className="rounded-[4px] border border-line bg-surface p-4">
        <h2 className="label mb-3 text-fg-2">
          {metricLabel} by week, {season}
        </h2>
        {chartData.length > 0 ? (
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
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
                  formatter={(v) => [v as number, metricLabel]}
                />
                <Line
                  type="linear"
                  dataKey="value"
                  stroke="var(--accent)"
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-[13px] text-fg-2">No data for this season and metric.</p>
        )}
      </div>
    </section>
  );
}
