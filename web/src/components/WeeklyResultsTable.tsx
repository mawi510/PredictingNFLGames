import type { WeeklyResult } from "@/lib/api";
import { formatUnits } from "@/lib/format";

// The API gives cumulative units per week; the weekly change is the difference.
export default function WeeklyResultsTable({ weekly }: { weekly: WeeklyResult[] }) {
  return (
    <table className="num w-full text-[14px]">
      <thead>
        <tr className="label h-8 border-b border-line text-fg-2">
          <th className="px-3 text-left font-normal">Week</th>
          <th className="px-3 text-right font-normal">W</th>
          <th className="px-3 text-right font-normal">L</th>
          <th className="px-3 text-right font-normal">Units</th>
          <th className="px-3 text-right font-normal">Cum. units</th>
        </tr>
      </thead>
      <tbody>
        {weekly.map((w, i) => {
          const prev = i > 0 ? weekly[i - 1].cumulative_units : 0;
          const units = w.cumulative_units - prev;
          return (
            <tr key={w.week} className="h-11 border-b border-line last:border-b-0">
              <td className="px-3">{w.week}</td>
              <td className="px-3 text-right">{w.wins}</td>
              <td className="px-3 text-right">{w.losses}</td>
              <td className={`px-3 text-right ${units > 0 ? "text-win" : units < 0 ? "text-loss" : "text-push"}`}>
                {formatUnits(units)}
              </td>
              <td className="px-3 text-right">{formatUnits(w.cumulative_units)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
