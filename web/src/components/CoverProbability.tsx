// Both sides' cover probabilities, each labelled with its team, in a fixed
// away-then-home order. The pick is the brighter pair. No bar: a fill that
// swaps sides from row to row reads as if the prediction itself flipped.
export default function CoverProbability({
  away,
  home,
  pAway,
  pick,
}: {
  away: string;
  home: string;
  pAway: number;
  pick: string;
}) {
  // One decimal, and the two sides always sum to exactly 100.
  const a = Math.round(pAway * 1000) / 10;
  const h = Math.round((100 - a) * 10) / 10;
  return (
    <span className="num whitespace-nowrap text-[14px]">
      <Side team={away} pct={a} strong={pick === away} />
      <span className="px-1.5 text-fg-3">&middot;</span>
      <Side team={home} pct={h} strong={pick === home} />
    </span>
  );
}

function Side({ team, pct, strong }: { team: string; pct: number; strong: boolean }) {
  return (
    <span className={strong ? "font-medium text-fg" : "text-fg-2"}>
      <span className="font-mono text-[12px]">{team}</span> {pct.toFixed(1)}%
    </span>
  );
}
