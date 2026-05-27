"use client";

import { useEffect, useState } from "react";
import PredictionPicker from "@/components/PredictionPicker";
import TeamStats from "@/components/TeamStats";
import { getStatus, getTeams, type Status } from "@/lib/api";
import { teamName } from "@/lib/teams";

const FORCE_IN_SEASON = process.env.NEXT_PUBLIC_FORCE_IN_SEASON === "true";

export default function Home() {
  const [status, setStatus] = useState<Status | null>(null);
  const [teams, setTeams] = useState<string[]>([]);
  const [team, setTeam] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getStatus().then(setStatus).catch(() => setFailed(true));
    getTeams()
      .then((res) => setTeams(res.teams.map((t) => t.team)))
      .catch(() => setFailed(true));
  }, []);

  const showPicker = FORCE_IN_SEASON || status?.predictions_available;

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <section className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Will they cover the spread?
        </h1>
      </section>

      {failed ? (
        <Card>
          <p className="text-danger">
            Couldn&apos;t reach the model API. Check that it&apos;s running and
            that <code className="text-foreground">NEXT_PUBLIC_API_BASE_URL</code>{" "}
            is set.
          </p>
        </Card>
      ) : (
        <>
          {/* One team selector drives both the prediction and the stats below. */}
          <label className="mb-8 flex max-w-sm flex-col gap-2">
            <span className="text-sm text-muted">Select a team</span>
            <select
              value={team}
              onChange={(e) => setTeam(e.target.value)}
              className="w-full rounded-lg border border-border bg-surface-2 px-4 py-3 text-foreground outline-none focus:border-accent"
            >
              <option value="">Choose a team…</option>
              {teams.map((t) => (
                <option key={t} value={t}>
                  {teamName(t)}
                </option>
              ))}
            </select>
          </label>

          {/* This week's prediction (season-gated). */}
          {status && (
            <>
              {FORCE_IN_SEASON && !status.predictions_available && (
                <p className="mb-3 text-xs text-accent">
                  Dev preview: showing the picker out of season
                  (NEXT_PUBLIC_FORCE_IN_SEASON).
                </p>
              )}
              <Card>
                {showPicker ? (
                  <>
                    <h2 className="mb-4 text-lg font-semibold">
                      This week&apos;s prediction
                    </h2>
                    <PredictionPicker team={team} />
                  </>
                ) : (
                  <>
                    <h2 className="text-lg font-semibold">
                      {status.reason === "off_season"
                        ? "We're between seasons"
                        : "Predictions aren't live yet"}
                    </h2>
                    <p className="mt-2 text-muted">{status.message}</p>
                  </>
                )}
              </Card>
            </>
          )}

          {/* Team stats are useful year-round, in-season and off. */}
          <TeamStats team={team} />
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
