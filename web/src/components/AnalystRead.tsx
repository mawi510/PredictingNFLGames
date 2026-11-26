import type { SlateGame } from "@/lib/api";
import { formatSpread, formatUpdated } from "@/lib/format";

const RESULT_CLS = { win: "text-win", loss: "text-loss", push: "text-push" } as const;

const FLAG_LABEL: Record<string, string> = {
  qb_out: "QB out",
  key_injuries: "Injuries",
  weather: "Weather",
  rest_or_travel: "Rest/travel",
  key_number: "Key number",
  line_moved: "Line moved",
  public_side: "Public side",
};

function pickSpread(game: SlateGame): number {
  const a = game.analyst!;
  return a.pick === game.home ? a.spread_home : -a.spread_home;
}

// One line under the model's pick: the analyst's side, its line and confidence.
export function AnalystTag({ game }: { game: SlateGame }) {
  const a = game.analyst;
  if (!a) return null;
  const cls = a.result ? RESULT_CLS[a.result] : "text-fg-2";
  return (
    <div className={`num mt-0.5 whitespace-nowrap text-[11px] ${cls}`}>
      <span className="text-fg-3">Analyst </span>
      {a.pick} {formatSpread(pickSpread(game))} &middot; {a.confidence}/5
    </div>
  );
}

// The analyst's reasoning, shown when a row is opened.
export function AnalystRead({ game }: { game: SlateGame }) {
  const a = game.analyst;
  if (!a) return null;
  return (
    <div className="border-b border-line bg-surface-2 px-3 pb-3 text-[13px] leading-5">
      <div className="label mb-1 text-fg-2">
        Analyst read: {a.pick} {formatSpread(pickSpread(game))}, confidence {a.confidence} of 5
      </div>
      <p className="text-fg">{a.reason}</p>
      {a.flags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {a.flags.map((f) => (
            <span key={f} className="label rounded-[4px] border border-line px-1.5 py-0.5 text-fg-2">
              {FLAG_LABEL[f] ?? f}
            </span>
          ))}
        </div>
      )}
      <p className="mt-2 text-[12px] text-fg-3">
        Claude researched injuries, weather, rest and line moves on the web. Logged{" "}
        {formatUpdated(a.logged_at)}, before kickoff.
        {a.sources.length > 0 && (
          <>
            {" "}Sources:{" "}
            {a.sources.map((url, i) => (
              <span key={url}>
                {i > 0 && ", "}
                <a href={url} target="_blank" rel="noopener noreferrer" className="underline hover:text-fg-2">
                  {new URL(url).hostname.replace(/^www\./, "")}
                </a>
              </span>
            ))}
          </>
        )}
      </p>
    </div>
  );
}
