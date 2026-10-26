// A 6px bar split at the probability. The pick side is accent, the other
// side is a neutral line. The two numbers sum to 100 and are never coloured
// by threshold.
export default function ProbabilityBar({
  pAway,
  pickSide,
}: {
  pAway: number;
  pickSide: "away" | "home";
}) {
  const away = Math.round(pAway * 100);
  const home = 100 - away;
  const awayStrong = away >= home;
  return (
    <div className="num flex w-full items-center gap-2 text-[14px]">
      <span className={`w-6 text-right ${awayStrong ? "text-fg" : "text-fg-2"}`}>{away}</span>
      <div className="flex h-1.5 flex-1 overflow-hidden rounded-[1px]">
        <div
          style={{ width: `${away}%` }}
          className={pickSide === "away" ? "bg-accent" : "bg-line-strong"}
        />
        <div
          style={{ width: `${home}%` }}
          className={pickSide === "home" ? "bg-accent" : "bg-line-strong"}
        />
      </div>
      <span className={`w-6 ${awayStrong ? "text-fg-2" : "text-fg"}`}>{home}</span>
    </div>
  );
}
