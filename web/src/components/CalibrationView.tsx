import type { CalibrationBin } from "@/lib/api";
import { formatPct } from "@/lib/format";

const MIN_PICKS = 100;

// Reliability by probability bin. Hidden behind a sample-size guard: a
// calibration read on a few dozen picks is noise.
export default function CalibrationView({
  bins,
  picks,
  brier,
}: {
  bins?: CalibrationBin[];
  picks: number;
  brier?: number;
}) {
  if (!bins || bins.length === 0 || picks < MIN_PICKS) {
    return (
      <p className="text-[13px] text-fg-2">
        Too few picks to read calibration yet. {picks} so far; {MIN_PICKS} needed.
      </p>
    );
  }
  return (
    <div>
      <table className="num w-full text-[14px]">
        <thead>
          <tr className="label h-8 border-b border-line text-fg-2">
            <th className="px-3 text-left font-normal">Predicted</th>
            <th className="px-3 text-right font-normal">Observed</th>
            <th className="px-3 text-right font-normal">Picks</th>
          </tr>
        </thead>
        <tbody>
          {bins.map((b) => (
            <tr key={b.bin} className="h-11 border-b border-line last:border-b-0">
              <td className="px-3">{b.bin}</td>
              <td className="px-3 text-right">{formatPct(b.observed)}</td>
              <td className="px-3 text-right">{b.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {brier !== undefined && (
        <p className="label mt-2 text-fg-3">Brier {brier.toFixed(3)} &middot; n={picks}</p>
      )}
    </div>
  );
}
