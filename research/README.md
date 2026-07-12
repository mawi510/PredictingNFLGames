# Research — model v2

Exploration branch for the ML/stats rework called out in `[[site-backlog]]` item 5.
Current production RF on 986 features backtested **125-162, −16.9% ROI on 2024**
(see `[[site-revamp-plan]]`). Goal here: try simpler / better-calibrated approaches.

No MLflow — results are printed and dumped to CSV per experiment dir.

## Approaches in scope

1. **Bayesian dynamic prior** (lead idea) — see `bayesian_dlm/`. Beta-Binomial per team
   with exponential forgetting; prior from previous season's cover behavior.
2. **Logistic regression + Markov chain** — model cover/streak state transitions.
3. **Boosted-trees baseline** — XGBoost/LightGBM on the curated feature set.

## Cross-cutting

- `feature_shortlist.md` — proposed cut from 986 → ~45 features (awaiting your redline).
- Walk-forward eval on 2024 = primary benchmark (matches the current track record).
- Success bar: beat 52.38% cover (breakeven at −110) on a meaningful bet count, OR clearly
  beat −16.9% ROI even if still below breakeven.

## How to run

```
python3 research/bayesian_dlm/spike.py
```

Outputs `results.csv` (full hyperparam sweep) and `weekly_equity.csv` (best config equity curve)
into `research/bayesian_dlm/`.

## ROI methodology

All ROI figures here are net of the −110 vig. A bet of 1u wins +100/110 = +0.909u or loses
1u. Breakeven is 110/210 ≈ 52.38% cover rate. `ROI = total_units / total_bets`. So a model
that hits 52% is essentially breakeven, NOT profitable — flat ~52% cover at −110 is the bar
to clear.

## Current findings (2026-05-30)

Bayesian dynamic prior, swept lambda ∈ {1.0, 0.95, 0.9, 0.8, 0.6} × prior_strength ∈ {4, 8, 16, 32}
× threshold ∈ {0.50, 0.5238}:

- Best: λ=0.6, prior_strength=4, threshold=0.5238 → **99-103, −13.0u, ROI −6.44%**
- Pattern: heavy forgetting (small λ) + weak season prior wins. Implies the season prior
  is noisy and recent-form dominates the predictive signal.
- Still losing, but ~10pp better ROI than the prod RF on the same season. So as a *much
  simpler model with a handful of params*, it's already a serious benchmark for the
  fancier approaches to clear.

Limitations of v1:
- Likelihood ignores the matchup — fixed in v2 below.
- 2024 cover base rate is 45.5% league-wide (`is_cover>0` strict; pushes counted as
  non-covers). Verify whether the live production rule treats pushes the same.

### v2: matchup-aware Elo on covers (`bayesian_dlm/spike_v2.py`)

Same Bayesian-dynamic spirit but in pairwise form. Each team has a latent skill θ_T
in logit space; `P(T covers vs O) = sigmoid(θ_T − θ_O + h·is_home_T)`. After each
game, θ_T and θ_O are nudged by `K·(y − p)` (the standard Elo update). Prior = previous
season's cover rate scaled into logit units by `init_scale`.

Sweep K × init_scale × h × threshold:

- **Best: K=0.05, init_scale=0.0, h=0.0, threshold=0.5238 → 37-31, +2.64u, ROI +3.88%**
  — first positive-ROI config in this research. Note: only 68 bets / season, so confidence
  is bounded by sample size; treat as "real signal, not yet validated."
- Runner-up with more volume: K=0.10, init_scale=0.25, threshold=0.5238 → 150 bets,
  51.3%, ROI −2.0% (essentially breakeven, much bigger sample).
- Pattern: `init_scale=0` (ignore the season prior entirely!) keeps coming up in the top
  configs. Implies the previous-season cover rate is a *misleading* prior in 2024 — the
  league mean-reverts week to week.
- Small K (0.05–0.10) dominates large K (0.4). Slow updates win.
- Home-field bonus `h` adds little; the spread already prices it.

### v2 multi-season validation (`bayesian_dlm/validate_v2.py`) — RESULT REVERSAL

Ran the v2 sweep across 2019-2024 and the 2024-best config (`init_scale=0`) cratered
on every other season. The +3.88% on 2024 was variance on 68 bets.

2024-best config (K=0.05, init_scale=0.0, h=0.0, thr=BE) per season:
2019 −16.1%, 2020 −30.2%, 2021 −5.7%, 2022 −21.1%, 2023 −22.2%, 2024 +3.9%.

Top aggregate configs across 2019-2024 ALL use `init_scale=1.0` (full prior):
- K=0.10, init_scale=1.0, h=0.05, thr=BE: 1397 bets, 47.24%, ROI **−9.8%**
- K=0.10, init_scale=1.0, h=0.00, thr=BE: 1355 bets, 47.08%, ROI −10.1%

Avg per-season rank by ROI shows a clean ordering: init_scale=1.0 best (rank 9.54),
0.0 worst (rank 14.69). **The user's original framing was right** — previous-season
cover rate is a useful prior. The single-season "init_scale=0 wins" read was wrong.

**Honest takeaway:** matchup-aware Elo on cover OUTCOMES alone caps out around
−10% ROI. Better than the prod RF's −16.9%, but still well below breakeven (52.38%).
Cover outcomes are too sparse / noisy a signal on their own. To clear breakeven we
need actual features (EPA differentials, matchup strength, etc.) feeding the prediction.

### XGBoost baseline on curated 51-feature set (`xgboost/`)

Built per `feature_shortlist.md`: 17 situational/market + 13 offense _rolling_shifted +
7 defense + 5 pressure/line + 3 QBR + 4 ATS/W-L records + 2 matchup-relative
(`passing_epa_minus_opp_passing_epa_allowed`, rushing twin).

Walk-forward by season (train on all prior seasons, predict on target). Best aggregate
config: `n_est=200, max_d=5, lr=0.03, threshold=0.5238`:

| season | bets | win% | ROI |
|---|---|---|---|
| 2019 | 211 | 52.6% | +0.43% |
| 2020 | 221 | 52.5% | +0.21% |
| 2021 | 215 | 49.3% | −5.88% |
| 2022 | 191 | 54.5% | +3.95% |
| 2023 | 197 | 50.8% | −3.09% |
| 2024 | 159 | 52.2% | −0.34% |
| **agg** | **1194** | **51.93%** | **−0.87%** |

Scoreboard vs prior approaches:

| Model | ROI | Bets | Sample |
|---|---|---|---|
| Prod RF (986 feat) | −16.9% | 287 | 2024 only |
| Bayesian Elo v2 | −9.8% | 1397 | 2019–2024 |
| **XGBoost (51 curated)** | **−0.87%** | **1194** | **2019–2024** |
| Breakeven −110 | 0.00% | — | — |

Patterns:
- Low complexity wins (n_est=200, max_d=3–5, lr=0.03); higher lr / more trees overfit hard.
- `threshold=BREAKEVEN` consistently beats `threshold=0.50` — calibration matters more than
  bet volume.
- 4 of 6 seasons profitable or near-flat; 2021 and 2023 are the bad ones.

**Honest caveat:** −0.87% across 1194 bets isn't profitable, and per-season variance (+4% to
−6%) is wider than the mean. This is "first config in striking distance of breakeven,
validated across 6 seasons" — not "the model works."

Next options (pick one):
1. **Tune harder on the existing 51 features** — sigmoid calibration, monotone constraints
   on market features, class-weighting; try LightGBM as a comparator.
2. **Stacked approach** — feed `theta_T − theta_O` from the v2 Elo as an additional feature.
3. **Feature work** — build the deferred matchup expansions (more EPA diffs, weather mismatch).
4. **Failure-mode dig** — what went wrong in 2021 and 2023 specifically? Targeted feature add.

### Stacked + calibrated (`stacked/`) — #2 + #3 results

Built `theta_diff` per team-week from the v2 Elo (`build_elo_feature.py`), merged into the
51-feature set. A/B/C/D test (`backtest.py`):

| Config | Bets | Win% | ROI |
|---|---|---|---|
| A: XGB baseline (51 feat) | 1194 | 51.93% | −0.87% |
| B: XGB + theta_diff | 1173 | 51.32% | **−2.02%** (Elo HURT) |
| C: XGB + theta_diff + isotonic calibration | 324 | 54.63% | **+4.29%** |
| D: LightGBM + theta_diff | 1173 | 50.47% | −3.65% |

Findings:
- **Stacking θ-diff hurts**, almost certainly because `is_cover_record_shifted` + the EPA-diff
  features already encode the same information. Drop it from the stack.
- **Isotonic calibration is the real lever** — collapses 1173 → 324 picks but at 54.63% win-rate.
  First positive aggregate ROI in this research. Per-season: 2019 0 bets (calibrator broke,
  only 2018 to train on), 2020 +10.9%, 2021 +40.7% (suspect variance, 14-5), 2022 +1.4%,
  2023 −2.2%, 2024 +27.3%. The "real" 3-season subset (2020/2022/2023, 261 bets) is at
  53.6% / ROI +2.4%.
- **LightGBM lost head-to-head** with XGB at matched params; stick with XGB.

### V1 Bayesian multi-season validation (`bayesian_dlm/validate_v1.py`)

Closed the open loop on spike v1 (Beta-Binomial with forgetting). Best aggregate:
λ=1.0, prior_strength=32, threshold=BE → 1060 bets, 49.06%, **ROI −6.35%**. Never
positive in any single season. The 2024 +3.88% finding was variance, as we suspected.

Avg per-season rank by hyperparam: λ=0.60 ranks best, prior_strength=32 ranks best.
The "fade prior into current season" intuition is directionally validated on per-season
rank, but absolute ROI for the pure-Bayesian approach floors around −6%.

### Cold-start features in XGBoost (`xgboost/build_features_v2.py` + `backtest_v2.py`)

Added three features per user's "blend prior season → current season as weeks accumulate":

```
prev_season_cover_rate        team's cover rate from season-1
weight_on_prior_season        exp(-week / tau), tau=3
blended_cover_rate            weight*prev_season + (1-weight)*is_cover_record_shifted
```

Weight schedule (tau=3): wk1=72%, wk4=36%, wk5=26%, wk8=7%. Matches user's intent
(prior dominates cold-start, current dominates by mid-season).

| Config | Bets | Win% | ROI |
|---|---|---|---|
| A: XGB 51 | 1194 | 51.93% | −0.87% |
| B: XGB 54 (uncal) | 1202 | 50.83% | **−2.96%** (HURT raw) |
| C: XGB 51 + isotonic | 382 | 54.45% | +3.95% |
| **D: XGB 54 + isotonic** | **494** | **53.85%** | **+2.80%** |

D vs C — same headline ROI ballpark but D's bet distribution is dramatically more even:

| season | C (51) | D (54) |
|---|---|---|
| 2019 | **0 bets** | 59 bets |
| 2020 | 56 | 110 |
| 2021 | 59 | 26 |
| 2022 | 44 | 47 |
| 2023 | 198 | 191 |
| 2024 | **8 bets** | 61 bets |

User's cold-start features fixed the calibrator's small-training-sample breakdown in
2019 and 2024. Lower headline ROI than C but trustworthy across all 6 seasons.

**Current best validated model: D — XGB on 54 features (51 curated + 3 cold-start) +
isotonic calibration. ROI +2.80% across 494 bets / 6 seasons.**

Next:
- Stability check on D (random seed sweep).
- Sweep `tau` ∈ {2, 3, 4, 5} on the blended feature.
- Platt (sigmoid) calibration as a comparator to isotonic.

### Backtest accounting fix + Config D revisited (`xgboost/backtest_v3.py`)

**Bug identified by user:** prior backtests iterated per team-week, which (a) double-bet
games where the model assigned both sides p > threshold (a guaranteed wash), and (b)
counted pushes (`is_cover=0` for both teams) as losses. **All prior ROIs were biased
downward.**

Fix:
- Per-GAME pick: pair team-weeks by (season, week, opposite spread, opposite is_home),
  pick `argmax(p_A, p_B)`, bet only if max > threshold.
- Push exclusion: if both sides' `is_cover=0`, drop from ROI entirely.

**Config D under corrected accounting (seed=7, tau=3, isotonic):** 1551 games, 134 pushes
excluded, 78 games were old-logic double-bets. **455 bets, 57.58% win-rate, +45.19u,
ROI +9.93%.** Per season:

| season | bets | win% | ROI |
|---|---|---|---|
| 2019 | 73 | 61.6% | +17.7% |
| 2020 | 61 | 60.7% | +15.8% |
| 2021 | 22 | 54.5% | +4.1% |
| 2022 | 52 | 61.5% | +17.5% |
| 2023 | 186 | 55.4% | +5.7% |
| 2024 | 61 | 54.1% | +3.3% |

Every season profitable.

**Seed stability (5 seeds): ROI mean +9.49% ± 3.54%, bets mean 434 ± 121.** 4 of 5 seeds
between +10% and +13%; seed=3 outlier at +3.45%.

**Tau sweep (seed=7):** τ=2 +14.3%, τ=3 +9.9%, τ=4 +12.2%, τ=5 +15.3%. All τ profitable.

**Isotonic vs Platt:** Platt too restrictive (16 bets total, 0 in most seasons). Stick with isotonic.

**Updated scoreboard:**

| Model | ROI | Bets | Sample / accounting |
|---|---|---|---|
| Prod RF (986 feat) | −16.9% | 287 | 2024 only, broken accounting |
| Bayesian v1 / v2 baselines | −6% to −10% | 1000+ | broken accounting |
| XGB 51 baseline (broken acct) | −0.87% | 1194 | broken accounting |
| **Config D, corrected accounting** | **+9.93%** | **455** | **6 seasons, every one positive** |
| Breakeven | 0% | — | |

**Caveats remaining:**
- 6 seasons is our full training universe. 2025+ in-season is the actual out-of-sample test.
- Calibrator still season-dependent (22 bets in 2021 vs 186 in 2023 with same model).
- Prior scoreboard entries haven't been re-run under corrected accounting; comparisons unfair.

**Next:** multi-seed τ sweep; re-run prior baselines under corrected accounting; design
the live in-season picks tool (show ALL games ranked by confidence per user feedback).

### Re-baseline under corrected accounting (`xgboost/rebaseline_v3.py`) — major revision

Re-ran every prior model under per-game pick + push exclusion. **All XGBoost variants are
profitable.** The accounting bug had been hiding +10% ROI across the board.

| Model | Win% | Bets | Units | ROI | Fav% |
|---|---|---|---|---|---|
| **XGB-C (51 feat + isotonic)** | 60.69% | 318 | +50.46u | **+15.87%** | 28% |
| **XGB-A (51 feat, NO calibration)** | 58.11% | **912** | **+99.83u** | **+10.95%** | 29% |
| XGB-D (54 feat + isotonic) | 57.58% | 455 | +45.19u | +9.93% | 33% |
| XGB-B (54 feat, no cal) | 55.18% | 946 | +50.53u | +5.34% | 31% |
| BETA-V1 (Beta-Binomial) | 54.28% | 807 | +29.18u | +3.62% | 64% |
| ELO-V2 (matchup Elo) | 51.68% | 1277 | −17.00u | −1.33% | 61% |

**Conclusions:**

- The XGBoost approach was always profitable; the bug was hiding it.
- **Cold-start features (D vs C, B vs A) HURT** under correct accounting. Drop them.
- **XGB-A is positive every season** (2019 +14.6%, 2020 +5.2%, 2021 +2.1%, 2022 +21.0%,
  2023 +8.6%, 2024 +18.9%). XGB-C has higher headline ROI but fragile in 2019/2024.
- **XGB picks underdogs 67-72%** of the time (Bayesian models pick favorites 60-64%).
  Matches the historical "underdogs cover ~50%, up to 60% in some eras" pattern — the
  model is finding edge on the dog side.
- Bayesian/Elo approaches lag XGB by 10+pp ROI; not competitive on their own.

**Production recommendation:** XGB-A (51 features, no calibration) for the live tool.
~150 picks/season at +10.95% ROI, every season positive, simplest deployment. Optionally
overlay XGB-C as a "high-confidence" tier (~50 picks/season at +15.87%).
