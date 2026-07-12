# Proposed feature cut: 986 → ~45

Draft for you to redline. Goal: drop the rank_ / expand_ / unshifted duplicate explosion,
keep one good representation of each underlying signal. All features below are the `_shifted`
version (prior-week info, no leakage), except situational/market columns that describe the
upcoming game itself.

## Keep verbatim (situational + market — describe the upcoming game)
- `spread`, `spread_odds`, `moneyline`, `exp_total_points`, `over_odds`, `under_odds`
- `is_home`, `is_fav`, `rest`, `div_game`
- Roof: `dome`, `outdoors`, `closed` (drop `open` — always 0 in 2024)
- Surface: `grass`, `fieldturf`, `astroturf` (drop the rare ones unless you want them)
- `players_injured_adj`

## Team form — offense (use `_rolling_shifted` only; drop raw & expanding & ranks)
- `points_scored_rolling_shifted`, `scoring_margin_rolling_shifted`
- `total_passing_epa_rolling_shifted`, `avg_passing_epa_rolling_shifted`
- `total_rushing_epa_rolling_shifted`, `avg_rushing_epa_rolling_shifted`
- `passing_yards_rolling_shifted`, `rushing_yards_rolling_shifted`
- `passing_tds_rolling_shifted`, `rushing_tds_rolling_shifted`
- `interceptions_rolling_shifted`, `sacks_rolling_shifted`
- `dakota_rolling_shifted` (composite passing efficiency)

## Team form — defense (the `_allowed_rolling_shifted` mirror)
- `points_allowed_rolling_shifted`, `scoring_margin_allowed_rolling_shifted`
- `total_passing_epa_allowed_rolling_shifted`
- `total_rushing_epa_allowed_rolling_shifted`
- `passing_yards_allowed_rolling_shifted`, `rushing_yards_allowed_rolling_shifted`
- `interceptions_allowed_rolling_shifted` (turnovers forced)

## Pressure / line play (PFR)
- `times_pressured_pct_rolling_shifted` (offensive line)
- `times_pressured_pct_allowed_rolling_shifted` (pass rush)
- `passing_bad_throw_pct_rolling_shifted`
- `rushing_yards_before_contact_avg_rolling_shifted` (run-blocking)
- `rushing_yards_after_contact_avg_rolling_shifted` (back quality)

## QB (ESPN QBR)
- `qbr_total_adj_rolling_shifted` (team-weighted QB play)
- `epa_total_adj_rolling_shifted`
- `qbr_total_adj_allowed_rolling_shifted` (defense vs QBs)

## ATS / W-L records (season-to-date, already cumulative)
- `is_cover_record_shifted`, `is_winner_record_shifted`
- `is_over_record_shifted`, `is_fav_record_shifted`

---

**What I'm dropping (and why):**
- All `rank_*_weekly` columns — pure rank-transform duplicates of the underlying metric;
  trees can learn the ordering themselves.
- All `expand_*` columns — season-to-date sums/avgs are largely redundant with `_record_shifted`
  + the 3-game rolling means; keeping both adds collinearity, not signal.
- Raw (unshifted, single-week) metrics — leakage risk + noise vs the rolling form.
- The receiving group as a whole — receiving_yards/tds/epa are mostly captured by passing
  metrics from the QB side (highly correlated). Add back if you disagree.
- `wopr` family — player-level metric; aggregated to team it's noisy.
- The two-point-conversion / fumble counters — rare events, low base rate.

**Decisions (2026-05-30, from user):**
1. Receiving family: keep dropped (passing is the proxy).
2. ADD matchup-relative: `passing_epa_minus_opp_passing_epa_allowed` (and the rushing twin
   `rushing_epa_minus_opp_rushing_epa_allowed`). Needs an opponent join when building features.
3. Skip `implied_cover_prob` from `spread_odds` — collinear with the spread itself; no new info.
4. Weather: skip the raw game-day weather (both teams experience it; signal cancels). Worth
   exploring a "weather mismatch" feature instead: delta between visiting team's home-city
   climate norm and the game-day conditions (warm-weather team in cold venue, dome team
   outdoors, etc.). Park as a follow-up.
