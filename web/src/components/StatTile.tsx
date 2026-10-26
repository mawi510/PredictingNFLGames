// Label on top, one big tabular number, one line of meta. Colour only when
// the value is a graded result (pass `tone` with the signed number).
export default function StatTile({
  label,
  value,
  meta,
  tone,
}: {
  label: string;
  value: string;
  meta?: string;
  tone?: number;
}) {
  const color =
    tone === undefined ? "text-fg" : tone > 0 ? "text-win" : tone < 0 ? "text-loss" : "text-push";
  return (
    <div className="rounded-[4px] border border-line bg-surface px-4 py-3">
      <div className="label text-fg-2">{label}</div>
      <div className={`num mt-1 text-[28px] font-semibold ${color}`}>{value}</div>
      {meta && <div className="mt-1 text-[13px] text-fg-2">{meta}</div>}
    </div>
  );
}
