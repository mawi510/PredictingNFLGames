import type { SlateGame } from "@/lib/api";
import { formatPct, formatSpread } from "@/lib/format";
import { teamName } from "@/lib/teams";

// A worked example, one row per step from the market line to the pick, using
// a real game from this week's slate.
export default function HowToRead({ game, modelVersion }: { game: SlateGame; modelVersion: string }) {
  const marketLine = game.favorite ? `${game.favorite} ${formatSpread(game.spread_fav)}` : "PK";
  const modelFav = game.predicted_margin_home >= 0 ? game.home : game.away;
  const modelAbs = Math.abs(game.predicted_margin_home);
  const modelLine = modelAbs === 0 ? "PK" : `${modelFav} ${formatSpread(-modelAbs)}`;
  const pickIsHome = game.pick === game.home;
  const other = pickIsHome ? game.away : game.home;
  const pickSpread = pickIsHome ? game.spread_home : -game.spread_home;
  const pPick = formatPct(game.p_pick, 1);
  const pOther = formatPct(1 - game.p_pick, 1);
  const lean = game.p_pick < 0.52;

  const rows: [string, string, string][] = [
    [
      "Market spread",
      marketLine,
      game.favorite
        ? `The sportsbook line. ${teamName(game.favorite)} must win by more than ${Math.abs(game.spread_fav)} to cover.`
        : "The sportsbook line. A pick-em: whoever wins covers.",
    ],
    [
      "Model line",
      modelLine,
      `The model's expected final margin, written the same way. It comes from the gap between the two teams' form; this week's line is not an input.`,
    ],
    [
      "Gap",
      `${game.pick} ${formatSpread(game.edge_pts)}`,
      `Model line minus market spread, from the side it favours. The model rates ${teamName(game.pick)} ${Math.abs(game.edge_pts).toFixed(1)} points better than the market does.`,
    ],
    [
      "Cover prob",
      `${game.pick} ${pPick} · ${other} ${pOther}`,
      "How often a gap this size has covered, measured on past seasons of out-of-sample predictions. The two sides sum to 100.",
    ],
    [
      "Pick",
      `${game.pick} ${formatSpread(pickSpread)}${lean ? " · lean" : ""}`,
      lean
        ? "The side with the higher cover probability. Under 52% it is a lean: the gap is inside the model's noise."
        : "The side with the higher cover probability. Picks are graded after kickoff on the Record page.",
    ],
  ];

  return (
    <section className="mt-8 max-w-[760px]">
      <h2 className="label mb-3 text-fg-2">
        How to read this: {game.away} at {game.home}
      </h2>
      {/* Three columns from 640px; below that the explanation drops under its label. */}
      <dl className="rounded-[4px] border border-line bg-surface text-[13px]">
        {rows.map(([label, value, meaning]) => (
          <div
            key={label}
            className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 border-b border-line px-3 py-2.5 last:border-b-0 sm:grid-cols-[8rem_11rem_1fr]"
          >
            <dt className="label pt-0.5 text-fg-3">{label}</dt>
            <dd className="num whitespace-nowrap font-mono text-fg">{value}</dd>
            <dd className="col-span-2 leading-5 text-fg-2 sm:col-span-1">{meaning}</dd>
          </div>
        ))}
      </dl>
      <p className="label mt-3 text-fg-3">{modelVersion}</p>
    </section>
  );
}
