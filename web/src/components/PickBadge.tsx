import { formatSpread } from "@/lib/format";

// Below 52% the pick is a lean, and the badge says so.
const LEAN_BELOW = 0.52;

const RESULT_STYLE = {
  win: { cls: "border-win text-win", mark: "W" },
  loss: { cls: "border-loss text-loss", mark: "L" },
  push: { cls: "border-push text-push", mark: "P" },
} as const;

// Once the game is final the badge takes the result's colour and letter.
export default function PickBadge({
  team,
  spread,
  probability,
  result,
}: {
  team: string;
  spread: number;
  probability: number;
  result?: "win" | "loss" | "push";
}) {
  const lean = probability < LEAN_BELOW;
  const graded = result ? RESULT_STYLE[result] : null;
  const cls = graded
    ? graded.cls
    : lean
      ? "border-line-strong text-fg-2"
      : "border-accent text-fg";
  return (
    <span
      className={`num inline-block whitespace-nowrap rounded-[4px] border px-2 py-0.5 text-[12px] font-medium ${cls}`}
    >
      {team} {formatSpread(spread)}
      {graded ? (
        <span> &middot; {graded.mark}</span>
      ) : (
        lean && <span className="text-fg-3"> &middot; lean</span>
      )}
    </span>
  );
}
