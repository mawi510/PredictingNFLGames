// Display metadata for the 32 NFL teams, keyed by the abbreviations used in the
// model data. ESPN logo codes differ for a few teams (Rams, Washington), hence
// the explicit `espn` field rather than just lowercasing the abbr.

interface TeamMeta {
  name: string;
  espn: string; // ESPN CDN logo code
}

const TEAMS: Record<string, TeamMeta> = {
  ARI: { name: "Arizona Cardinals", espn: "ari" },
  ATL: { name: "Atlanta Falcons", espn: "atl" },
  BAL: { name: "Baltimore Ravens", espn: "bal" },
  BUF: { name: "Buffalo Bills", espn: "buf" },
  CAR: { name: "Carolina Panthers", espn: "car" },
  CHI: { name: "Chicago Bears", espn: "chi" },
  CIN: { name: "Cincinnati Bengals", espn: "cin" },
  CLE: { name: "Cleveland Browns", espn: "cle" },
  DAL: { name: "Dallas Cowboys", espn: "dal" },
  DEN: { name: "Denver Broncos", espn: "den" },
  DET: { name: "Detroit Lions", espn: "det" },
  GB: { name: "Green Bay Packers", espn: "gb" },
  HOU: { name: "Houston Texans", espn: "hou" },
  IND: { name: "Indianapolis Colts", espn: "ind" },
  JAX: { name: "Jacksonville Jaguars", espn: "jax" },
  KC: { name: "Kansas City Chiefs", espn: "kc" },
  LA: { name: "Los Angeles Rams", espn: "lar" },
  LAC: { name: "Los Angeles Chargers", espn: "lac" },
  LV: { name: "Las Vegas Raiders", espn: "lv" },
  MIA: { name: "Miami Dolphins", espn: "mia" },
  MIN: { name: "Minnesota Vikings", espn: "min" },
  NE: { name: "New England Patriots", espn: "ne" },
  NO: { name: "New Orleans Saints", espn: "no" },
  NYG: { name: "New York Giants", espn: "nyg" },
  NYJ: { name: "New York Jets", espn: "nyj" },
  PHI: { name: "Philadelphia Eagles", espn: "phi" },
  PIT: { name: "Pittsburgh Steelers", espn: "pit" },
  SEA: { name: "Seattle Seahawks", espn: "sea" },
  SF: { name: "San Francisco 49ers", espn: "sf" },
  TB: { name: "Tampa Bay Buccaneers", espn: "tb" },
  TEN: { name: "Tennessee Titans", espn: "ten" },
  WAS: { name: "Washington Commanders", espn: "wsh" },
};

export function teamName(abbr: string): string {
  return TEAMS[abbr]?.name ?? abbr;
}

export function teamLogo(abbr: string): string | null {
  const meta = TEAMS[abbr];
  return meta
    ? `https://a.espncdn.com/i/teamlogos/nfl/500/${meta.espn}.png`
    : null;
}
