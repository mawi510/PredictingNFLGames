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

export default function TeamStats({ team }: { team: string }) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [season, setSeason] = useState("");
  const [metric, setMetric] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setStats(null);
    setError(null);
    if (!team) return;
    getTeamStats(team)
      .then((s) => {
        setStats(s);
        setSeason(String(s.seasons[0] ?? ""));
        setMetric(s.metrics[0]?.key ?? "");
      })
      .catch(() => setError(`Couldn't load stats for ${team}.`));
  }, [team]);

  // Metrics that actually have data for the chosen season.
  const availableMetrics = useMemo(() => {
    if (!stats || !season) return stats?.metrics ?? [];
    const present = stats.series[season]?.metrics ?? {};
    return stats.metrics.filter((m) => m.key in present);
  }, [stats, season]);

  const chartData = useMemo(() => {
    if (!stats || !season || !metric) return [];
    const s = stats.series[season];
    if (!s || !(metric in s.metrics)) return [];
    return s.weeks.map((week, i) => ({ week, value: s.metrics[metric][i] }));
  }, [stats, season, metric]);

  // Keep the metric valid when the season changes.
  useEffect(() => {
    if (availableMetrics.length && !availableMetrics.some((m) => m.key === metric)) {
      setMetric(availableMetrics[0].key);
    }
  }, [availableMetrics, metric]);

  const metricLabel =
    stats?.metrics.find((m) => m.key === metric)?.label ?? "";

  return (
    <section className="mt-8">
      <h2 className="mb-1 text-lg font-semibold">Team performance</h2>
      <p className="mb-4 text-sm text-muted">
        How a team&apos;s weekly numbers have trended, this season and in past
        seasons.
      </p>

      <div className="rounded-xl border border-border bg-surface p-6">
        <div className="grid gap-4 sm:grid-cols-2">
          <Select
            label="Season"
            value={season}
            onChange={setSeason}
            disabled={!stats}
          >
            {stats?.seasons.map((s) => (
              <option key={s} value={String(s)}>
                {s}
              </option>
            ))}
          </Select>

          <Select
            label="Metric"
            value={metric}
            onChange={setMetric}
            disabled={!stats}
          >
            {availableMetrics.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </Select>
        </div>

        {error && <p className="mt-4 text-sm text-danger">{error}</p>}

        {stats && chartData.length > 0 && (
          <div className="mt-6 h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis
                  dataKey="week"
                  stroke="var(--muted)"
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  label={{ value: "Week", position: "insideBottom", offset: -2, fill: "var(--muted)", fontSize: 12 }}
                />
                <YAxis stroke="var(--muted)" tick={{ fontSize: 12 }} tickLine={false} width={44} />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface-2)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    color: "var(--foreground)",
                  }}
                  labelFormatter={(w) => `Week ${w}`}
                  formatter={(v) => [v as number, metricLabel]}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {!stats && !error && (
          <p className="mt-6 text-sm text-muted">
            {team
              ? "Loading trends…"
              : "Select a team above to see their weekly trends."}
          </p>
        )}
      </div>
    </section>
  );
}

function Select({
  label,
  value,
  onChange,
  disabled,
  children,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-sm text-muted">{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-border bg-surface-2 px-4 py-3 text-foreground outline-none focus:border-accent disabled:opacity-50"
      >
        {children}
      </select>
    </label>
  );
}
