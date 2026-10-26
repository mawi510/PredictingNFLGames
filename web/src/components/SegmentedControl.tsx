"use client";

// Replaces native selects. 32px tall, mono labels, wraps when it must.
export default function SegmentedControl<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-1">
      {options.map((o) => {
        const selected = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(o.value)}
            className={`label h-8 rounded-[4px] border px-2.5 transition-colors duration-100 ${
              selected
                ? "border-line-strong bg-surface-2 text-fg"
                : "border-line text-fg-2 hover:bg-surface-2"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
