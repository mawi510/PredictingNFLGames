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
  const pPick = formatPct(game.p_pick);
  const pOther = formatPct(1 - game.p_pick);
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
      `The model's expected final margin, written the same way. It comes from each team's form; the market line is not an input.`,
    ],
    [
      "Gap",
      `${game.pick} ${formatSpread(game.edge_pts)}`,
      `Model line minus market spread, from the side it favours. The model rates ${teamName(game.pick)} ${Math.abs(game.edge_pts).toFixed(1)} points better than the market does.`,
    ],
    [
      "Cover prob",
      `${game.pick} ${pPick} · ${other} ${pOther}`,
      "How often a gap this size covers, given how far the model's past predictions missed. The two sides sum to 100.",
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
      <div className="rounded-[4px] border border-line bg-surface">
        <table className="w-full text-[13px]">
          <tbody>
            {rows.map(([label, value, meaning]) => (
              <tr key={label} className="border-b border-line align-top last:border-b-0">
                <th scope="row" className="label w-28 px-3 py-2.5 text-left font-normal text-fg-3 sm:w-32">
                  {label}
                </th>
                <td className="num w-36 whitespace-nowrap px-3 py-2.5 font-mono text-fg sm:w-44">{value}</td>
                <td className="px-3 py-2.5 leading-5 text-fg-2">{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="label mt-3 text-fg-3">{modelVersion}</p>
    </section>
  );
}
