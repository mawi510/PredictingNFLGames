// Typed client for the ProMatchPredict model API.
// Base URL comes from NEXT_PUBLIC_API_BASE_URL so the same build can point at a
// local API container or the live service.

const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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

export interface Prediction {
  team: string;
  season: number;
  week: number;
  spread: number;
  cover_record: number;
  cover_probability: number;
  model_version: string;
  is_stub: boolean;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
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

export const getStatus = () => getJSON<Status>("/status");
export const getTeams = () => getJSON<TeamsResponse>("/teams");
export const getTeamPrediction = (team: string) =>
  getJSON<Prediction>(`/teams/${encodeURIComponent(team)}/prediction`);
export const getTeamStats = (team: string) =>
  getJSON<TeamStats>(`/teams/${encodeURIComponent(team)}/stats`);
