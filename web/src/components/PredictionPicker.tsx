"use client";

import { useEffect, useState } from "react";
import {
  getTeams,
  getTeamPrediction,
  type Prediction,
  type TeamInfo,
} from "@/lib/api";

function formatSpread(spread: number): string {
  return spread > 0 ? `+${spread}` : `${spread}`;
}

export default function PredictionPicker() {
  const [teams, setTeams] = useState<TeamInfo[] | null>(null);
  const [team, setTeam] = useState<string>("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTeams()
      .then((res) => setTeams(res.teams))
      .catch(() => setError("Couldn't load teams from the model API."));
  }, []);

  async function onSelect(value: string) {
    setTeam(value);
    setPrediction(null);
    setError(null);
    if (!value) return;
    setLoading(true);
    try {
      setPrediction(await getTeamPrediction(value));
    } catch {
      setError(`Couldn't get a prediction for ${value}.`);
    } finally {
      setLoading(false);
    }
  }

  if (error && !teams) {
    return <p className="text-sm text-danger">{error}</p>;
  }

  return (
    <div className="flex flex-col gap-6">
      <label className="flex flex-col gap-2">
        <span className="text-sm text-muted">Select a team</span>
        <select
          value={team}
          onChange={(e) => onSelect(e.target.value)}
          className="w-full rounded-lg border border-border bg-surface-2 px-4 py-3 text-foreground outline-none focus:border-accent"
        >
          <option value="">Choose a team…</option>
          {teams?.map((t) => (
            <option key={t.team} value={t.team}>
              {t.team} (spread {formatSpread(t.spread)})
            </option>
          ))}
        </select>
      </label>

      {loading && <p className="text-sm text-muted">Running the model…</p>}
      {error && teams && <p className="text-sm text-danger">{error}</p>}

      {prediction && !loading && (
        <PredictionCard prediction={prediction} />
      )}
    </div>
  );
}

function PredictionCard({ prediction }: { prediction: Prediction }) {
  const pct = Math.round(prediction.cover_probability * 100);
  const leans = pct >= 50;
  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">
          {prediction.team}{" "}
          <span className="text-sm font-normal text-muted">
            · Week {prediction.week}, {prediction.season}
          </span>
        </h2>
        <span className="text-sm text-muted">
          spread {formatSpread(prediction.spread)}
        </span>
      </div>

      <div className="mt-6 flex items-end gap-3">
        <span
          className="text-5xl font-bold tabular-nums"
          style={{ color: leans ? "var(--accent)" : "var(--danger)" }}
        >
          {pct}%
        </span>
        <span className="pb-1 text-sm text-muted">chance to cover the spread</span>
      </div>

      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-surface-2">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: leans ? "var(--accent)" : "var(--danger)",
          }}
        />
      </div>

      <p className="mt-5 text-sm text-muted">
        The model {leans ? "leans toward" : "leans against"}{" "}
        <span className="text-foreground">{prediction.team}</span> covering at a
        spread of {formatSpread(prediction.spread)}. This season they&apos;ve
        covered {Math.round(prediction.cover_record * 100)}% of the time.
      </p>

      {prediction.is_stub && (
        <p className="mt-3 text-xs text-danger">
          Showing placeholder output (no model loaded).
        </p>
      )}
    </div>
  );
}
