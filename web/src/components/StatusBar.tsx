"use client";

import { useAppData } from "@/lib/data";
import { formatUpdated, formatUpdatedDay, shortModelVersion } from "@/lib/format";

const SEP = " · ";

export default function StatusBar() {
  const { status, slate, record } = useAppData();
  const s = status.data;
  const generatedAt = record.data?.generated_at ?? slate.data?.generated_at ?? null;

  let left: string;
  let error = false;

  if (status.loading) {
    left = "";
  } else if (status.failed && !status.stale) {
    error = true;
    left = generatedAt
      ? `API UNREACHABLE${SEP}LAST GOOD UPDATE ${formatUpdatedDay(generatedAt)}`
      : "API UNREACHABLE";
  } else if (s && !s.in_season) {
    const next = s.season != null ? s.season + 1 : "";
    left = `OFF SEASON${SEP}NEXT SLATE WK 4 ${next}`.trim();
  } else if (s) {
    const parts = [String(s.season), `WEEK ${s.latest_week}`];
    if (slate.data) parts.push(`${slate.data.games.length} GAMES`);
    left = parts.join(SEP);
  } else {
    left = "";
  }

  const rightParts: string[] = [];
  if (generatedAt) rightParts.push(`UPDATED ${formatUpdated(generatedAt)}`);
  const version = slate.data?.model_version ?? record.data?.model_version;
  if (version) rightParts.push(shortModelVersion(version));

  const stale = status.stale || slate.stale || record.stale;
  const placeholder = slate.data?.is_stub || record.data?.is_stub;

  return (
    <div className="border-b border-line bg-surface">
      <div
        className={`label mx-auto flex h-9 max-w-[1200px] items-center justify-between gap-4 px-4 md:px-6 ${
          error ? "text-loss" : "text-fg-2"
        }`}
      >
        <span className="truncate">{left}</span>
        <span className="flex shrink-0 items-center gap-3">
          {rightParts.length > 0 && (
            <span className="hidden sm:inline">{rightParts.join(SEP)}</span>
          )}
          {stale && <Chip tone="loss">Stale</Chip>}
          {placeholder && <Chip tone="flag">Placeholder model</Chip>}
        </span>
      </div>
    </div>
  );
}

function Chip({ tone, children }: { tone: "loss" | "flag"; children: string }) {
  const color = tone === "loss" ? "text-loss border-loss" : "text-flag border-flag";
  return (
    <span className={`label rounded-[4px] border px-1.5 py-0.5 ${color}`}>{children}</span>
  );
}
