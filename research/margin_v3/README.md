# Margin model v3: fixing the narrow line (issue #25)

## Problem

The v2 margin model (one XGBoost row per team, game line = average of the two
teams' views) produced a line about half as wide as the market's. In 85% of
games it sat inside the spread, so the pick was the underdog: 89% of 2026
picks. Its record tracked "always bet the underdog" within a few games in
every holdout season, and the site's record fell when favorites started
covering in weeks 6 to 9 of 2026.

## Method

`walkforward.py` trains each candidate on seasons before `s` and predicts
season `s`, for 2019 through 2026. No candidate sees this week's line.

- `team`: the v2 form.
- `team+opp`: v2 with the opponent's features appended to each row.
- `diff_xgb`: one XGBoost row per game on home-minus-away feature differences.
- `diff_ridge`: ridge regression on the same standardized differences.
- `*_cal`: the same line times a slope fitted on earlier out-of-sample seasons.

Selection rule, fixed before the run: lowest pooled out-of-sample RMSE of the
final margin on 2019-2025, slope nearest 1 as tie-break. The record against
the spread is reported but was not used to choose. 2026 was not used to
choose.

## Results

Pooled 2019-2025, 1,823 games. Market: RMSE 12.72, line sd 6.22.

| Model | RMSE | Slope | Line sd | Against the spread | Underdog picks |
|---|---|---|---|---|---|
| team (v2 form) | 13.72 | 1.08 | 3.60 | 929-851 (52.2%) | 85% |
| team+opp | 13.61 | 0.75 | 6.04 | 923-857 (51.9%) | 67% |
| diff_xgb | 13.64 | 0.71 | 6.37 | 896-884 (50.3%) | 64% |
| **diff_ridge** | **13.30** | 0.93 | 5.54 | 907-873 (51.0%) | 68% |
| team_cal | 13.76 | 1.25 | 3.05 | 939-841 (52.8%) | 88% |
| team+opp_cal | 13.59 | 1.00 | 4.35 | 924-856 (51.9%) | 80% |
| diff_xgb_cal | 13.59 | 0.96 | 4.54 | 921-859 (51.7%) | 79% |
| diff_ridge_cal | 13.33 | 0.95 | 5.33 | 897-883 (50.4%) | 71% |

RMSE by season (lower is better):

| Season | Market | team (v2 form) | diff_ridge |
|---|---|---|---|
| 2019 | 13.13 | 14.85 | 13.82 |
| 2020 | 12.77 | 13.81 | 13.29 |
| 2021 | 13.67 | 14.60 | 14.36 |
| 2022 | 11.43 | 12.28 | 12.01 |
| 2023 | 13.07 | 13.85 | 13.74 |
| 2024 | 12.58 | 13.19 | 12.47 |
| 2025 | 12.27 | 13.27 | 13.16 |
| 2026 to week 9 | 12.00 | 12.52 | 12.48 |

`diff_ridge` wins by the rule, beats the v2 form in every season, and needs no
slope correction. It closes about 40% of the v2 form's gap to the market.

## What this does not show

No candidate beats the spread. Every record is within noise of 50%, and all
are below the 52.4% needed to break even at -110. On 2026 to date `diff_ridge`
went 61-68 in this harness against 65-64 for the v2 form; a four-game
difference on 129 picks is noise in either direction. The v2 form's 52.2%
pooled record is the underdog effect: favorites covered under 47% in every
season from 2018 to 2022.

The change makes the line accurate in scale and removes the mechanical
underdog bias. It is not an edge.

## Cover probabilities were overconfident

v2 turned a margin into a cover probability with the margin residual sd (about
13 points). That treats the model's line as the truth and ignores what the
market line already holds. On out-of-sample cover outcomes (weeks 4+, pushes
excluded):

| Scale | Log loss |
|---|---|
| Coin flip | 0.6931 |
| Residual sd, 13.3 (v2 method) | 0.7125 |
| Fitted on cover outcomes, 67.9 | 0.6919 |

The v2 method scored worse than a coin flip. With the fitted scale, nested by
season (scale for season `s` from seasons before `s`, 2021-2026):

| Shown probability | Picks | Hit rate |
|---|---|---|
| 50-52% | 743 | 49.7% |
| 52-54% | 294 | 52.7% |
| 54-57% | 87 | 58.6% |
| 57%+ | 4 | 25.0% |

The probabilities are honest now, and they say most picks are near coin flips.
The 54-57% row is 87 picks over six seasons and is not evidence of an edge.

## One feature is past market information

The largest coefficient is `is_fav_record_shifted`: how often the team was the
favorite in earlier weeks of the season. It is known before kickoff, so it is
not leakage, but "no market input" would overstate the model's independence.
The accurate claim: this week's line is not an input. v2 used the same
feature.

## Shipped

- `pipeline/train_margin.py` fits the ridge, refits the cover scale from
  walk-forward lines, and writes `api/model_margin_v3.json` (plain JSON).
- Serving needs numpy and scipy only; XGBoost is gone from the API image.
- Published picks are locked in `pipeline/picks_log/`, so the model change
  does not rewrite weeks 4 to 9, which v2 picked.

## Files

- `walkforward.py`: the comparison. Output in `walkforward_summary.csv` and
  `walkforward_games.csv` (one row per game, every candidate's line).
