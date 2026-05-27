"use client";

import { useEffect, useState } from "react";
import { getTeamPrediction, type Prediction } from "@/lib/api";
import { teamLogo, teamName } from "@/lib/teams";

function formatSpread(spread: number): string {
  return spread > 0 ? `+${spread}` : `${spread}`;
}

export default function PredictionPicker({ team }: { team: string }) {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPrediction(null);
    setError(null);
    if (!team) return;
    setLoading(true);
    getTeamPrediction(team)
      .then(setPrediction)
      .catch(() => setError(`Couldn't get a prediction for ${team}.`))
      .finally(() => setLoading(false));
  }, [team]);

  if (!team) {
    return <p className="text-sm text-muted">Select a team to see this week&apos;s prediction.</p>;
  }
  if (loading) return <p className="text-sm text-muted">Running the model…</p>;
  if (error) return <p className="text-sm text-danger">{error}</p>;
  if (!prediction) return null;
  return <PredictionCard prediction={prediction} />;
}

function PredictionCard({ prediction }: { prediction: Prediction }) {
  const pct = Math.round(prediction.cover_probability * 100);
  const leans = pct >= 50;
  const logo = teamLogo(prediction.team);
  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {logo && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={logo}
              alt={`${teamName(prediction.team)} logo`}
              width={44}
              height={44}
              className="h-11 w-11 object-contain"
            />
          )}
          <h2 className="text-lg font-semibold">
            {teamName(prediction.team)}{" "}
            <span className="block text-sm font-normal text-muted">
              Week {prediction.week}, {prediction.season}
            </span>
          </h2>
        </div>
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

      <p className="mt-5 text-sm text-muted">
        The model {leans ? "leans toward" : "leans against"}{" "}
        <span className="text-foreground">{teamName(prediction.team)}</span>{" "}
        covering at a spread of {formatSpread(prediction.spread)}. This season
        they&apos;ve covered {Math.round(prediction.cover_record * 100)}% of the
        time.
      </p>

      {prediction.is_stub && (
        <p className="mt-3 text-xs text-danger">
          Showing placeholder output (no model loaded).
        </p>
      )}
    </div>
  );
}
