import { formatSpread } from "@/lib/format";

// Below 52% the pick is a lean, and the badge says so.
const LEAN_BELOW = 0.52;

export default function PickBadge({
  team,
  spread,
  probability,
}: {
  team: string;
  spread: number;
  probability: number;
}) {
  const lean = probability < LEAN_BELOW;
  return (
    <span
      className={`num inline-block whitespace-nowrap rounded-[4px] border px-2 py-0.5 text-[12px] font-medium ${
        lean ? "border-line-strong text-fg-2" : "border-accent text-fg"
      }`}
    >
      {team} {formatSpread(spread)}
      {lean && <span className="text-fg-3"> &middot; lean</span>}
    </span>
  );
}
