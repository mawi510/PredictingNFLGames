import type { Methodology as MethodologyData } from "@/lib/api";

export default function Methodology({
  methodology,
  models,
}: {
  methodology: MethodologyData;
  models?: { version: string; from_week: number }[];
}) {
  const rows = [
    ["Pick rule", methodology.pick_rule],
    ["Odds", methodology.odds_assumption],
    ["Unit", methodology.pick_unit],
    ["First graded week", `Week ${methodology.min_week}`],
  ];
  if (models && models.length > 0) {
    rows.push(["Model", models.map((m) => `${m.version} from week ${m.from_week}`).join("; ")]);
  }
  return (
    <div className="max-w-[680px] text-[13px] leading-6 text-fg-2">
      <dl>
        {rows.map(([k, v]) => (
          <div key={k} className="flex gap-4 border-b border-line py-2 last:border-b-0">
            <dt className="label w-36 shrink-0 pt-1 text-fg-3">{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4">
        In walk-forward tests on 2019 to 2025 the current model picked about 51% against the
        closing line, below the 52.4% needed to break even at -110. The record above is the
        current season graded week by week; a season is one sample.
      </p>
    </div>
  );
}
