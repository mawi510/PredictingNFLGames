"use client";

import { useEffect, useState } from "react";
import PredictionPicker from "@/components/PredictionPicker";
import { getStatus, type Status } from "@/lib/api";

const FORCE_IN_SEASON =
  process.env.NEXT_PUBLIC_FORCE_IN_SEASON === "true";

export default function Home() {
  const [status, setStatus] = useState<Status | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setFailed(true));
  }, []);

  const showPicker = FORCE_IN_SEASON || status?.predictions_available;

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <section className="mb-10">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Will they cover the spread?
        </h1>
      </section>

      {failed && (
        <Card>
          <p className="text-danger">
            Couldn&apos;t reach the model API. Check that it&apos;s running and
            that <code className="text-foreground">NEXT_PUBLIC_API_BASE_URL</code>{" "}
            is set.
          </p>
        </Card>
      )}

      {!failed && !status && (
        <Card>
          <p className="text-muted">Loading…</p>
        </Card>
      )}

      {!failed && status && (
        <>
          {FORCE_IN_SEASON && !status.predictions_available && (
            <p className="mb-4 text-xs text-accent">
              Dev preview: showing the picker out of season
              (NEXT_PUBLIC_FORCE_IN_SEASON).
            </p>
          )}

          {showPicker ? (
            <Card>
              <PredictionPicker />
            </Card>
          ) : (
            <Card>
              <h2 className="text-lg font-semibold">
                {status.reason === "off_season"
                  ? "We're between seasons"
                  : "Predictions aren't live yet"}
              </h2>
              <p className="mt-2 text-muted">{status.message}</p>
              {status.season && (
                <p className="mt-4 text-sm text-muted">
                  Most recent season on record:{" "}
                  <span className="text-foreground">{status.season}</span>
                  {status.latest_week
                    ? ` (through Week ${status.latest_week})`
                    : ""}
                  .
                </p>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      {children}
    </div>
  );
}
