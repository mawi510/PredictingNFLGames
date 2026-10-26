// Number and date formatting. Every number on the site goes through here so
// signs, decimals and time zones stay consistent.

function signed(n: number, digits: number): string {
  const fixed = Math.abs(n).toFixed(digits);
  if (Number(fixed) === 0) return (0).toFixed(digits);
  return n < 0 ? `-${fixed}` : `+${fixed}`;
}

// Spreads carry a sign and one decimal; a zero spread is a pick-em.
export function formatSpread(spread: number): string {
  if (spread === 0) return "PK";
  return signed(spread, 1);
}

// Probabilities are whole percent, never coloured.
export function formatPct(p: number): string {
  return `${Math.round(p * 100)}%`;
}

export function formatUnits(units: number): string {
  return `${signed(units, 1)}u`;
}

export function formatMargin(margin: number): string {
  return signed(margin, 1);
}

export function formatRoi(roi: number): string {
  return `${signed(roi * 100, 1)}%`;
}

const ET = "America/New_York";

// "SUN OCT 25" in Eastern time, the time zone NFL schedules are published in.
export function formatKickoffDay(iso: string): string {
  return new Date(iso)
    .toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      timeZone: ET,
    })
    .replace(",", "")
    .toUpperCase();
}

// "1:00 PM ET"
export function formatKickoffTime(iso: string): string {
  return `${new Date(iso).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: ET,
  })} ET`;
}

// "WED OCT 21 17:56 UTC" for the status bar.
export function formatUpdated(iso: string): string {
  const d = new Date(iso);
  const day = d
    .toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    })
    .replace(",", "");
  const time = d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
  return `${day} ${time} UTC`.toUpperCase();
}

// "WED OCT 21" for the error state.
export function formatUpdatedDay(iso: string): string {
  return new Date(iso)
    .toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    })
    .replace(",", "")
    .toUpperCase();
}

// "margin-v2.2026.10" shows as "MARGIN-V2" in the status bar.
export function shortModelVersion(version: string): string {
  return version.split(".")[0].toUpperCase();
}
