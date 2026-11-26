import { formatSpread } from "@/lib/format";

const RESULT_STYLE = {
  win: { cls: "border-win text-win", mark: "W" },
  loss: { cls: "border-loss text-loss", mark: "L" },
  push: { cls: "border-push text-push", mark: "P" },
} as const;

// Once the game is final the badge takes the result's colour and letter.
export default function PickBadge({
  team,
  spread,
  result,
}: {
  team: string;
  spread: number;
  result?: "win" | "loss" | "push";
}) {
  const graded = result ? RESULT_STYLE[result] : null;
  const cls = graded ? graded.cls : "border-accent text-fg";
  return (
    <span
      className={`num inline-block whitespace-nowrap rounded-[4px] border px-2 py-0.5 text-[12px] font-medium ${cls}`}
    >
      {team} {formatSpread(spread)}
      {graded && <span> &middot; {graded.mark}</span>}
    </span>
  );
}
