"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import SegmentedControl from "@/components/SegmentedControl";
import TeamStats from "@/components/TeamStats";
import { getTeams } from "@/lib/api";
import { teamLogo, teamName } from "@/lib/teams";

export default function TeamsPage() {
  return (
    <Suspense fallback={<h1 className="text-[20px] font-semibold leading-tight">Teams</h1>}>
      <Teams />
    </Suspense>
  );
}

function Teams() {
  const params = useSearchParams();
  const [teams, setTeams] = useState<string[]>([]);
  const [team, setTeam] = useState(params.get("team")?.toUpperCase() ?? "");

  useEffect(() => {
    getTeams()
      .then((res) => {
        const list = res.teams.map((t) => t.team);
        setTeams(list);
        setTeam((cur) => cur || list[0] || "");
      })
      .catch(() => setTeams([]));
  }, []);

  const logo = team ? teamLogo(team) : null;

  return (
    <>
      <div className="mb-4 flex items-center gap-3">
        {logo && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={logo} alt="" width={28} height={28} className="h-7 w-7 object-contain" />
        )}
        <h1 className="text-[20px] font-semibold leading-tight">
          {team ? teamName(team) : "Teams"}
        </h1>
      </div>

      {teams.length > 0 && (
        <div className="mb-6">
          <SegmentedControl
            label="Team"
            options={teams.map((t) => ({ value: t, label: t }))}
            value={team}
            onChange={setTeam}
          />
        </div>
      )}

      {team && <TeamStats team={team} />}
    </>
  );
}
