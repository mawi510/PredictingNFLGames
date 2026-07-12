# Paper-trade plan — 2025 NFL season (margin model v2)

The lockbox 2024 result (+10.10% ROI, 57.67% win, 189 bets) and the 6/6 walk-forward seasons positive aren't enough to risk real money. The 2025 paper-trade is the genuine forward test: predictions are logged in real time, before games kick off, and compared to actuals as weeks resolve. No retraining mid-season, no model swaps.

## Objective

Confirm that the model's backtested edge persists in live conditions over 12+ weeks of 2025 regular-season play. If yes, consider live deployment for 2026. If no, treat the model as research-grade only and investigate why the backtest didn't transfer.

## What gets logged each week

Every Wednesday morning (after `nfl_data_py` refreshes the spread feed — see [[weekly-cron-wednesday]]), a job writes one row per team-week to `s3://nfl.data/paper_trade_2025/wk{NN}.json`:

```jsonc
{
  "season": 2025,
  "week": 4,
  "logged_at": "2025-09-24T11:02:00Z",
  "model_version": "margin-v2.2025.09",
  "picks": [
    {
      "team_a": "BUF", "team_b": "NO",
      "spread_a": -7.5,                  // market closing line as of log time
      "is_home_a": 1,
      "mu_hat_a": -4.91, "mu_hat_b": +4.91,
      "scale_t": 14.4, "nu_t": 11.8,    // from prod manifest
      "p_cover_a": 0.367, "p_cover_b": 0.633,
      "edge_pts_a": -12.41,              // mu_a + spread_a
      "pick": "NO",
      "p_pick": 0.633
    }
  ]
}
```

The actual outcome (margin, push y/n) is filled in by a Tuesday morning resolver job after MNF using the same `nfl_data_py.import_schedules` source the lockbox used. Resolved rows append into `s3://nfl.data/paper_trade_2025/resolved.json`.

## Snapshot timing

Log at **Wednesday 11:00 ET**. Reasons:
1. `nfl_data_py` updates spreads from Lee Sharpe's habitatring feed on Wednesdays ([[weekly-cron-wednesday]]).
2. User actually places picks Wed evening (TNF) and Sat afternoon (Sun + MNF) per stated workflow.
3. Wednesday morning is far enough from kickoff that we're predicting against early-week lines, not chasing late line moves. This intentionally hurts ROI slightly (we don't capture line shopping) but gives a conservative read.

For TNF specifically: the Wednesday log captures the line as of that morning. For Sun/Mon games: same Wednesday log governs (we don't re-log Saturday for the late games). Keep it boring and deterministic — late-week line chasing is a separate research question.

## Success criteria

After the season ends (or sooner if hit), evaluate:

| Cohort | Bets expected | Threshold |
|---|---|---|
| All picks, bet-every-game | ~280-310 | win% ≥ 53.5% (1pp above breakeven; not heroic) |
| Filtered P(pick) ≥ 0.60 | ~110-140 | win% ≥ 56.0% with non-negative ROI |
| Calibration | n/a | monotone: higher P-bucket → higher hit rate |

**If all three hit:** strong evidence the +6-10% backtested ROI is real, not look-ahead artifact. Proceed to bet-sizing / live-money discussion for 2026.

**If only the calibration metric hits (monotone) but ROI is flat or negative:** the model knows what it knows but the edge has been arbitraged out. Stop here, treat as research artifact, do not bet live.

**If aggregate is positive but calibration breaks (high-conf bets lose):** suspicious. Investigate before celebrating — likely a few lucky weeks masking a structural issue.

## Hard guardrails (the rules I'd hold myself to)

1. **No model changes during the season.** Even if week 6 looks ugly. The whole point of paper-trade is to test the frozen model. Changing it mid-stream invalidates the test.
2. **No filtering after the fact.** Pre-declare the filtered cohort (P ≥ 0.60) now. Don't go fishing for a subset that's profitable in retrospect.
3. **Track pushes honestly.** Pushes excluded from win% denominator (same as the backtest accounting).
4. **No closing-line tracking as the primary signal.** Closing lines move toward the truth ([[feedback-closing-line-leakage]]). We log the Wednesday line and stick with that result. Closing line is informational only.
5. **One season is one data point.** Even +10% over 280 bets has wide Wilson CI. Two seasons of confirmation is the minimum for live money.

## Reporting cadence

Mid-week tweet/dashboard cadence is fine, but the official scorecard updates **only** after each week's MNF resolves (Tuesday morning). Publish: cumulative bets, win%, ROI, calibration buckets. Same metrics the lockbox 2024 summary uses, so they're directly comparable.

## Implementation TODO (post-EC2 cutover)

- [ ] `pipeline/log_paper_trade.py` — Wed 11:00 ET cron, pulls current week predictions from the live API, writes per-week JSON to S3
- [ ] `pipeline/resolve_paper_trade.py` — Tue 09:00 ET cron, pulls actual scores from `nfl_data_py`, joins to picks, appends to resolved.json
- [ ] `api/main.py` `/paper-trade/scorecard` endpoint that reads resolved.json and exposes cumulative metrics for the web UI
- [ ] `web/src/app/track-record/` route that displays the live 2025 scorecard alongside the lockbox 2024 baseline

## What this is NOT

This is not a betting strategy. It's a model-validation exercise. Bet sizing, bankroll, and live execution are separate problems and require σ-calibrated edge + Kelly fraction discipline — both of which exist in the current model but neither of which is being tested by flat-unit paper-trade.
