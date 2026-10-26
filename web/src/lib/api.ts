// Typed client for the ProMatchPredict model API.
// The base URL is baked in at build time so one build can point at a local
// API container or the live service.

import slateFixture from "./fixtures/slate.json";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const USE_FIXTURE = process.env.NEXT_PUBLIC_USE_FIXTURE === "true";

export interface Status {
  in_season: boolean;
  season: number | null;
  latest_week: number | null;
  predictions_available: boolean;
  reason: "off_season" | "insufficient_data" | "ok" | string;
  message: string;
}

export interface TeamInfo {
  team: string;
  week: number;
  spread: number;
  cover_record: number;
}

export interface TeamsResponse {
  season: number;
  latest_week: number;
  teams: TeamInfo[];
}

export interface SlateGame {
  away: string;
  home: string;
  kickoff: string | null;
  // Handicap applied to the home score; negative means home is favoured.
  spread_home: number;
  favorite: string | null;
  spread_fav: number;
  p_home: number;
  p_away: number;
  predicted_margin_home: number;
  pick: string;
  p_pick: number;
  edge_pts: number;
}

export interface Slate {
  season: number;
  week: number;
  generated_at: string;
  model_version: string;
  is_stub: boolean;
  games: SlateGame[];
}

export interface Methodology {
  pick_rule: string;
  min_week: number;
  odds_assumption: string;
  pick_unit: string;
}

export interface RecordSummary {
  wins: number;
  losses: number;
  pushes: number;
  total_picks: number;
  win_pct: number;
  total_units: number;
  roi: number;
}

export interface WeeklyResult {
  week: number;
  wins: number;
  losses: number;
  cumulative_wins: number;
  cumulative_losses: number;
  cumulative_units: number;
}

export interface CalibrationBin {
  bin: string;
  predicted: number;
  observed: number;
  count: number;
}

export interface TrackRecord {
  season: number;
  latest_week: number;
  generated_at: string;
  is_stub: boolean;
  model_version: string;
  methodology: Methodology;
  record: RecordSummary;
  weekly: WeeklyResult[];
  calibration?: CalibrationBin[];
  brier?: number;
}

export interface MetricMeta {
  key: string;
  label: string;
}

export interface SeasonSeries {
  weeks: number[];
  metrics: Record<string, (number | null)[]>;
}

export interface TeamStats {
  team: string;
  metrics: MetricMeta[];
  seasons: number[];
  series: Record<string, SeasonSeries>;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const getStatus = () => getJSON<Status>("/status");
export const getTeams = () => getJSON<TeamsResponse>("/teams");
export const getTrackRecord = () => getJSON<TrackRecord>("/track-record");
export const getTeamStats = (team: string) =>
  getJSON<TeamStats>(`/teams/${encodeURIComponent(team)}/stats`);

// The fixture only ever stands in for a failed request, and only when the
// build opted in. A production build never shows it.
export async function getSlate(): Promise<Slate> {
  try {
    return await getJSON<Slate>("/slate");
  } catch (err) {
    if (USE_FIXTURE) return slateFixture as Slate;
    throw err;
  }
}
