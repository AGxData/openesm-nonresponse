# Feature Engineering Documentation

## Overview

The `engineer_features()` function generates temporal and behavioral features for predicting participant nonresponse in Experience Sampling Method (ESM) and Ecological Momentary Assessment (EMA) studies.

The function is designed to work with reconstructed prompt-level datasets where each row represents one scheduled prompt, including prompts that participants did not answer.

The feature engineering process relies primarily on:

- Participant ID
- Study day
- Beep number
- Response indicator

No timestamps, demographic information, or additional participant metadata are required.

The objective is to capture:

1. Individual response tendencies.
2. Recent response behavior.
3. Within-day response patterns.
4. Participant-specific behavior at particular beep positions.
5. Changes in participation over the course of a study.
6. Temporal patterns in response behavior.

---

# 1. Input Requirements

## Required columns

| Column | Description |
|---|---|
| `id` | Participant identifier |
| `day` | Study day |
| `beep` | Prompt/beep number within the study day |

## Response information

The function supports either:

- `responded`: binary indicator, where 1 represents a response and 0 represents nonresponse.
- `response_status`: categorical response status.

The response status values `answered` and `partial` are treated as responses.

Reconstructed prompts and explicitly identified nonresponses are forced to zero.

## Important assumptions

- Each row represents one scheduled prompt.
- The dataset includes reconstructed nonresponses where available.
- Study day and beep number must be numeric or numerically convertible.
- Duplicate participant/day/beep combinations are not permitted for same-beep matching.
- Rows are sorted by participant, study day, and beep number before features are calculated.

---

# 2. Feature Configuration

The function supports configurable feature windows and smoothing parameters.

| Parameter | Default | Description |
|---|---|---|
| `rolling_windows` | `(3, 6, 12)` | Number of previous prompts used for rolling response rates |
| `daily_windows` | `(3, 7)` | Number of previous observed days used for daily rolling rates |
| `ewm_spans` | `(3, 6)` | Exponential weighting spans |
| `beta_prior_alpha` | `1.0` | Alpha parameter for Beta smoothing |
| `beta_prior_beta` | `1.0` | Beta parameter for Beta smoothing |

All historical behavioral features exclude the current prompt.

---

# 3. Basic Temporal Features

These features describe the participant's position within the study and the structure of the prompt schedule.

| Feature | Description | Definition |
|---|---|---|
| `prompt_index` | Sequential prompt number | Ordinal position of the prompt within the participant's observed dataset |
| `day_index` | Numeric study day | Numeric representation of the study day |
| `beep_index` | Numeric beep position | Numeric representation of the beep number |
| `day_squared` | Nonlinear study-day effect | `day_index²` |
| `beep_squared` | Nonlinear beep-position effect | `beep_index²` |
| `day_beep_interaction` | Day and beep interaction | `day_index × beep_index` |
| `days_since_first_observed` | Elapsed study days | Current study day minus the participant's first observed study day |
| `first_observed_day` | First observed study day indicator | 1 if the current day is the participant's first observed day |
| `last_observed_day` | Last observed study day indicator | 1 if the current day is the participant's last observed day |

### Interpretation

These features allow models to identify broad temporal patterns.

For example:

- Response behavior may deteriorate as the study progresses.
- Participants may respond differently to early and late prompts.
- The relationship between study day and beep position may be nonlinear.

### Predictor suitability

| Feature | Suitable for prediction? |
|---|---|
| `prompt_index` | Yes |
| `day_index` | Yes |
| `beep_index` | Yes |
| `day_squared` | Yes |
| `beep_squared` | Yes |
| `day_beep_interaction` | Yes |
| `days_since_first_observed` | Yes |
| `first_observed_day` | Yes |
| `last_observed_day` | Depends on whether the study endpoint is known in advance |

`last_observed_day` is retrospective if the participant's final observed day is determined from the complete dataset. Do not use it as a predictor unless the scheduled study endpoint is independently known before prediction.

---

# 4. Schedule Structure Features

These features describe the position of a prompt within its study day.

| Feature | Description | Definition |
|---|---|---|
| `prompts_in_day` | Number of observed scheduled prompts that day | Count of participant/day rows |
| `first_beep_of_day` | First beep indicator | 1 if the current beep is the earliest beep that day |
| `last_beep_of_day` | Last beep indicator | 1 if the current beep is the latest beep that day |
| `beep_position` | Relative position within the day | Beep rank divided by number of prompts that day |
| `days_since_previous_prompt` | Gap since previous observed prompt | Current study day minus previous prompt's study day |
| `beep_difference_from_previous` | Difference in beep number | Current beep number minus previous prompt's beep number |
| `beep_gap_from_previous_same_day` | Gap between same-day beeps | Difference between current and previous beep number within the day |
| `scheduled_beeps_in_day` | Number of prompts in the day | Currently identical to `prompts_in_day` |
| `scheduled_beep_position` | Sequential position within the day's rows | Row position within participant/day |
| `scheduled_beep_position_rate` | Relative sequential position | Scheduled beep position divided by scheduled beeps in the day |

### Interpretation

These features allow the model to learn whether response probability depends on prompt position.

For example:

- Participants may be less likely to answer the final prompt of the day.
- Response probability may differ between early and late daily prompts.
- Larger gaps between prompts may influence subsequent response behavior.

### Important limitation

`prompts_in_day`, `first_beep_of_day`, `last_beep_of_day`, and relative position features are safe predictors only when the schedule is fully reconstructed and known before the target prompt.

If the number of prompts is determined retrospectively from observed responses, these features may introduce leakage or schedule-related bias.

---

# 5. Response Indicator

The response indicator is the target variable used throughout the behavioral feature calculations.

| Value | Meaning |
|---|---|
| 1 | Participant answered or partially answered |
| 0 | Confirmed nonresponse |

Partial responses count as responses.

The function prioritizes `response_status` when available.

Reconstructed prompts and explicitly identified nonresponses are forced to zero.

### Important consideration

Missing or unrecognized response statuses are currently treated as nonresponses.

This assumes the input data has already distinguished genuine nonresponses from unavailable, invalid, or otherwise unknown observations.

If unknown response statuses exist, they should be resolved before modeling rather than automatically interpreted as nonresponse.

---

# 6. Cumulative Response History

These features describe the participant's overall response behavior up to the current prompt.

| Feature | Description | Definition |
|---|---|---|
| `cumulative_responses` | Total responses so far | Cumulative sum including current prompt |
| `cumulative_prompts` | Total prompts so far | Cumulative prompt count including current prompt |
| `cumulative_response_rate` | Overall response rate so far | Cumulative responses / cumulative prompts |
| `prior_cumulative_responses` | Previous responses | Cumulative responses excluding current prompt |
| `prior_cumulative_prompts` | Previous prompts | Cumulative prompts excluding current prompt |
| `prior_cumulative_response_rate` | Historical response rate | Previous responses / previous prompts |
| `prior_cumulative_nonresponses` | Previous missed prompts | Previous prompts minus previous responses |
| `prior_missed_beep_count` | Historical missed prompt count | Alias of `prior_cumulative_nonresponses` |
| `prior_smoothed_response_rate` | Smoothed historical response rate | Beta-smoothed previous response rate |
| `prior_response_history_count` | Number of historical observations | Previous prompt count |

### Smoothed response rate

The smoothed response rate uses a Beta prior:

\[
\text{Smoothed Rate} =
\frac{R+\alpha}{N+\alpha+\beta}
\]

Where:

- \(R\) = Previous responses
- \(N\) = Previous prompts
- \(\alpha\) = Beta prior alpha
- \(\beta\) = Beta prior beta

The default prior is Beta(1, 1).

This prevents extreme response-rate estimates when only a small number of observations are available.

### Interpretation

These features measure a participant's general historical compliance.

A participant with a consistently high response rate may be more likely to answer the next prompt.

### Predictor suitability

- `prior_cumulative_*` features: valid predictors.
- `prior_smoothed_response_rate`: valid predictor.
- `cumulative_*` features: retrospective, because they include the current response.

---

# 7. Previous Prompt Response History

These features capture short-term response behavior.

| Feature | Description | Definition |
|---|---|---|
| `prev_1_response` | Previous prompt response | Response to the immediately preceding prompt |
| `prev_2_response` | Two prompts ago | Response two prompts earlier |
| `prev_3_response` | Three prompts ago | Response three prompts earlier |
| `prev_4_response` | Four prompts ago | Response four prompts earlier |
| `prev_5_response` | Five prompts ago | Response five prompts earlier |
| `prev_6_response` | Six prompts ago | Response six prompts earlier |
| `prev_response` | Previous prompt response alias | Identical to `prev_1_response` |
| `prev_response_same_day` | Previous response within current day | Response to the immediately preceding prompt on the same day |

### Interpretation

These features capture short-term behavioral dependence.

For example, someone who missed their previous prompt may have an increased probability of missing their next prompt.

The features can help identify:

- Consecutive missed prompts.
- Short-term disengagement.
- Immediate response persistence.
- Within-day behavioral patterns.

---

# 8. Rolling Prior Prompt Response Rates

These features calculate response rates over a fixed number of previous prompts.

| Feature | Description |
|---|---|
| `prior_3_prompt_response_rate` | Response rate over the previous three prompts |
| `prior_6_prompt_response_rate` | Response rate over the previous six prompts |
| `prior_12_prompt_response_rate` | Response rate over the previous twelve prompts |

The windows are configurable through `rolling_windows`.

### Formula

\[
\text{Rolling Rate}_t =
\frac{\sum_{i=t-w}^{t-1}Y_i}{w}
\times 100
\]

Where:

- \(Y_i\) = Previous prompt response indicator
- \(w\) = Rolling window size

The implementation uses available historical observations when fewer than the requested number of prompts exist.

### Interpretation

Unlike cumulative response rates, rolling rates emphasize recent behavior.

A participant may have a historically high response rate but recently become less responsive.

Rolling rates can capture this deterioration.

---

# 9. Exponentially Weighted Response History

New features.

Exponentially weighted moving averages give more importance to recent observations while retaining information from older observations.

| Feature | Description |
|---|---|
| `prior_ewm_3_prompt_response_rate` | Exponentially weighted response rate with span 3 |
| `prior_ewm_6_prompt_response_rate` | Exponentially weighted response rate with span 6 |

The spans are configurable through `ewm_spans`.

### Interpretation

These features occupy a middle ground between cumulative and rolling response rates.

- Cumulative rates retain all previous observations equally.
- Rolling rates discard observations outside the window.
- EWM rates gradually reduce the influence of older observations.

This can help identify changing response tendencies without selecting a strict rolling-window boundary.

### Important consideration

The span is not a fixed number of observations after which older observations disappear.

Older observations continue contributing with progressively smaller weights.

---

# 10. Response Streaks

These features measure consecutive response behavior.

| Feature | Description |
|---|---|
| `prior_response_streak` | Number of consecutive previous prompts answered |
| `prior_nonresponse_streak` | Number of consecutive previous prompts missed |

Both exclude the current prompt.

### Interpretation

Response streaks capture persistence that may not be represented adequately by a response rate.

For example:

- A participant who missed four consecutive prompts may behave differently from someone who missed four scattered prompts.
- Two participants can have identical cumulative response rates but very different recent streaks.

---

# 11. Time Since Previous Response

| Feature | Description |
|---|---|
| `days_since_previous_response` | Number of study days since the participant's previous response |

This feature uses study-day information rather than actual timestamps.

If the participant has not previously responded, the value is missing.

### Interpretation

This measures how long the participant has gone without answering a prompt.

It can help identify sustained disengagement.

### Important distinction

This is not the elapsed time between the previous prompt and the previous response.

It is a difference in study-day identifiers.

---

# 12. Within-Day Response History

These features capture response behavior earlier in the current study day.

| Feature | Description |
|---|---|
| `prior_same_day_responses` | Number of previous responses today |
| `prior_same_day_nonresponses` | Number of previous nonresponses today |
| `prior_same_day_prompts` | Number of previous prompts today |
| `prior_same_day_response_rate` | Response rate across previous prompts today |
| `prior_same_day_smoothed_response_rate` | Beta-smoothed response rate today |
| `prior_same_day_response_streak` | Consecutive previous responses today |
| `prior_same_day_nonresponse_streak` | Consecutive previous nonresponses today |
| `prior_same_day_2_prompt_response_rate` | Response rate across previous two same-day prompts |
| `prior_same_day_3_prompt_response_rate` | Response rate across previous three same-day prompts |

### Interpretation

These features allow the model to recognize daily behavioral changes.

For example, a participant may generally respond well but become progressively less likely to answer later prompts on a particular day.

### Predictor suitability

All features in this section use only earlier prompts from the current day and are valid predictors.

---

# 13. Exponentially Weighted Within-Day History

New features.

| Feature | Description |
|---|---|
| `prior_same_day_ewm_3_response_rate` | Exponentially weighted response rate across previous same-day prompts |
| `prior_same_day_ewm_6_response_rate` | Longer-span exponentially weighted response rate across previous same-day prompts |

These features provide a recency-weighted measure of the participant's behavior within the current day.

They are particularly relevant when response probability changes over the course of a day.

---

# 14. Participant × Beep Historical Behavior

New feature group.

This is one of the most important additions to the feature engineering process.

These features capture a participant's historical response behavior at a particular beep position across study days.

For example, a participant may consistently respond to Beep 1 but frequently miss Beep 4.

General cumulative response rates do not capture this distinction.

## Features

| Feature | Description |
|---|---|
| `prior_same_beep_responses` | Previous responses at the current beep position |
| `prior_same_beep_nonresponses` | Previous nonresponses at the current beep position |
| `prior_same_beep_observations` | Previous occurrences of the current beep position |
| `prior_same_beep_response_rate` | Historical response rate at this beep position |
| `prior_same_beep_smoothed_rate` | Beta-smoothed response rate at this beep position |
| `prior_same_beep_response_streak` | Consecutive previous responses at this beep position |
| `prior_same_beep_nonresponse_streak` | Consecutive previous nonresponses at this beep position |
| `prev_same_beep_response` | Response at the previous occurrence of this beep position |
| `days_since_previous_same_beep` | Study-day difference since the previous occurrence |
| `prior_same_beep_ewm_3_response_rate` | Exponentially weighted response rate at this beep |
| `prior_same_beep_ewm_6_response_rate` | Longer-span exponentially weighted response rate at this beep |

### Example

Consider the following participant history:

| Study day | Beep 1 | Beep 2 | Beep 3 |
|---|---:|---:|---:|
| 1 | 1 | 1 | 0 |
| 2 | 1 | 0 | 0 |
| 3 | 1 | 1 | Target |

When predicting the third beep on Day 3, the model can use:

- Two previous nonresponses at Beep 3.
- Zero previous responses at Beep 3.
- Two previous observations at Beep 3.
- A historical response rate of 0%.
- A smoothed response rate of 25% under the default Beta(1, 1) prior.
- A previous occurrence response of 0.
- A previous occurrence gap of one study day.

### Important distinction

`prev_same_beep_response` uses the previous observed occurrence of that beep position.

It does not necessarily mean the previous calendar day.

For example, if Beep 3 was observed on Days 1 and 4, the previous occurrence for Day 4 is Day 1.

This differs from `prev_day_same_beep_response`, which specifically matches the immediately preceding numeric study day.

### Why this group matters

This feature group captures individual behavioral routines that general response history may miss.

It effectively creates a participant-specific response profile across beep positions.

---

# 15. Previous-Day Same-Beep Response

| Feature | Description |
|---|---|
| `prev_day_same_beep_response` | Response to the same beep number on the immediately preceding numeric study day |
| `prev_day_same_beep_available` | Whether the previous numeric study day contained that beep |

### Example

If predicting Beep 2 on Day 5:

- The function looks for Beep 2 on Day 4.
- If found, it retrieves that response.
- If not found, the response is missing and availability is zero.

### Important limitation

The implementation matches `current_day - 1`.

It does not automatically find the previous observed day.

If Day 4 is missing and Day 3 contains the previous Beep 2 observation, Day 3 is not used by this particular feature.

Use `prev_same_beep_response` and the historical same-beep features for broader previous-occurrence information.

---

# 16. Daily Participation History

These features summarize response behavior across entire study days.

They are calculated at the participant/day level and then mapped back to prompt-level observations.

## Daily summaries

| Feature | Description |
|---|---|
| `responded_any_today` | Whether the participant answered at least one prompt during the current day |
| `daily_response_rate` | Percentage of prompts answered during the current day |
| `missed_day` | Whether the participant missed every prompt during the current day |

These three features are retrospective summaries of the complete current day.

**Do not use them as predictors for individual prompts occurring earlier in that day.**

## Prior daily history

| Feature | Description |
|---|---|
| `prior_day_response_rate` | Mean response rate across all previous observed days |
| `prior_days_with_response` | Number of previous observed days with at least one response |
| `prior_days_observed` | Number of previous observed study days |
| `prior_day_participation_rate` | Percentage of previous observed days with at least one response |
| `prev_day_responded` | Whether the previous observed day contained at least one response |
| `prev_day_response_rate` | Response rate on the previous observed day |
| `days_since_previous_observed_day` | Difference between current and previous observed study-day identifiers |

### Interpretation

These features capture participation at a broader temporal level than prompt-level history.

A participant may have a high response rate within active days but frequently skip entire days.

Daily participation features help distinguish those patterns.

---

# 17. Consecutive Daily Participation Streaks

New features.

| Feature | Description |
|---|---|
| `prior_consecutive_active_days` | Consecutive previous observed days with at least one response |
| `prior_consecutive_inactive_days` | Consecutive previous observed days with no responses |

### Interpretation

These features measure sustained participation or disengagement across study days.

They are different from prompt-level response streaks.

For example, a participant could miss several prompts consecutively but still answer another prompt that day.

### Important limitation

The current implementation measures consecutive observed days, not necessarily consecutive numeric study days.

If observations exist on Days 1 and 3 but not Day 2, those two observed days are adjacent in the streak calculation.

Use `days_since_previous_observed_day` to identify gaps.

---

# 18. Rolling Daily Response Rates

These features calculate response behavior across previous observed study days.

| Feature | Description |
|---|---|
| `prior_3_day_response_rate` | Mean daily prompt response rate across the previous three observed days |
| `prior_7_day_response_rate` | Mean daily prompt response rate across the previous seven observed days |
| `prior_3_day_participation_rate` | Percentage of previous three observed days with at least one response |
| `prior_7_day_participation_rate` | Percentage of previous seven observed days with at least one response |

The windows are configurable through `daily_windows`.

### Important distinction

The daily response rate measures the average percentage of prompts answered.

The daily participation rate measures whether the participant answered anything at all during a day.

These are different behavioral characteristics.

---

# 19. Exponentially Weighted Daily Response History

New features.

| Feature | Description |
|---|---|
| `prior_ewm_3_day_response_rate` | Exponentially weighted average of previous daily response rates |
| `prior_ewm_6_day_response_rate` | Longer-span exponentially weighted average of previous daily response rates |

These features emphasize recent daily participation patterns while retaining older observations.

They can help identify changes in compliance over the course of a study.

---

# 20. Daily Response Rate Change

| Feature | Description |
|---|---|
| `prior_response_rate_change` | Difference between the recent three-observed-day average response rate and the historical average daily response rate |

### Formula

\[
\text{Rate Change} =
\text{Prior 3-Day Response Rate}
-
\text{Prior Overall Daily Response Rate}
\]

Positive values indicate that recent response behavior is better than the participant's historical average.

Negative values indicate that recent response behavior is worse.

This feature is calculated only when the three-day window is included in `daily_windows`.

---

# 21. Daily Cumulative Summary

The `daily_cumulative` output contains daily snapshots of cumulative participant response history.

| Feature | Description |
|---|---|
| `cumulative_responses` | Total responses through the final prompt of the day |
| `cumulative_prompts` | Total prompts through the final prompt of the day |
| `cumulative_response_rate` | Cumulative response rate through the final prompt of the day |
| `responded_any` | Whether the participant answered at least one prompt that day |
| `daily_response_rate` | Response rate for the complete day |

These are useful for exploratory analysis and visualization.

The cumulative values include the current day's outcomes and should not be used as predictors for earlier prompts.

---

# 22. Participant-Level Retrospective Summary

The `participant_status` output contains participant-level summaries.

| Feature | Description |
|---|---|
| `last_response_day` | Last study day on which the participant responded |
| `first_response_day` | First study day on which the participant responded |
| `total_observed_prompts` | Total observed prompts |
| `total_responses` | Total responses |
| `overall_response_rate` | Overall participant response rate |
| `never_responded` | Whether the participant never responded |

These features are retrospective and summarize the participant's complete observed study history.

They are useful for:

- Participant-level exploratory analysis.
- Descriptive statistics.
- Identifying extreme participation patterns.
- Understanding the composition of datasets.

They must not be used as predictors for earlier prompts because they contain future information.

---

# 23. Leakage and Predictor Suitability

Feature leakage is especially important in prompt-level prediction.

The objective is to estimate the probability of nonresponse before the current prompt's response is known.

Therefore, every predictor must be constructed exclusively from information available before the target prompt.

## Generally valid predictors

| Feature group | Predictor suitability |
|---|---|
| Basic temporal features | Generally valid |
| Schedule structure | Valid if schedule is known in advance |
| Prior cumulative response history | Valid |
| Previous prompt response lags | Valid |
| Rolling prior prompt response rates | Valid |
| Exponentially weighted prompt response rates | Valid |
| Prior response streaks | Valid |
| Time since previous response | Valid |
| Within-day prior response history | Valid |
| Participant × beep historical behavior | Valid |
| Previous-day same-beep response | Valid |
| Prior daily participation history | Valid |
| Prior daily response streaks | Valid |
| Prior rolling daily response rates | Valid |
| Prior exponentially weighted daily rates | Valid |

## Retrospective features

| Feature | Reason |
|---|---|
| `cumulative_responses` | Includes current response |
| `cumulative_prompts` | Includes current prompt |
| `cumulative_response_rate` | Includes current response |
| `responded_any_today` | Uses complete current-day outcomes |
| `daily_response_rate` | Uses complete current-day outcomes |
| `missed_day` | Uses complete current-day outcomes |
| `daily_cumulative` values | Includes current and previous outcomes |
| `participant_status` values | Uses complete participant history |
| `last_observed_day` | May depend on future observations |

### Important note on dataset construction

Even correctly calculated historical features can leak information if the underlying dataset contains future information in its construction.

For example:

- A participant's final observed day may only be known retrospectively.
- The number of prompts in a day may be derived from the completed dataset rather than a known schedule.
- Participant-level summaries may include outcomes from the evaluation period.

The prediction pipeline should preserve the information available at the actual prediction time.

---

# 24. Recommended Feature Groups for Modeling

I recommend testing the feature groups independently and incrementally.

This helps determine which types of historical information actually improve nonresponse prediction.

| Experiment | Feature groups |
|---|---|
| Baseline | Study day, beep number, schedule features |
| Model A | Baseline + cumulative response history |
| Model B | Model A + previous prompt lags |
| Model C | Model B + rolling response history |
| Model D | Model C + participant × beep history |
| Model E | Model D + EWM response history |
| Model F | Model E + daily participation history |
| Model G | Model F + daily streaks |

This is an ablation strategy: each successive experiment measures the incremental contribution of an additional feature group.

### Particularly interesting comparisons

**Cumulative versus rolling versus exponentially weighted history**

These represent different assumptions about how past behavior influences future behavior.

**General response history versus participant × beep history**

This tests whether participants have consistent response tendencies at particular beep positions.

**Prompt-level versus daily-level history**

This tests whether short-term response behavior and broader daily participation patterns provide distinct predictive information.

---

# 25. Recommended Evaluation

Because nonresponse is the primary event of interest, evaluation should emphasize the model's ability to identify nonresponses rather than simply predict the majority response class.

For evaluation, define nonresponse as the positive class.

| Metric | Purpose |
|---|---|
| Nonresponse PR-AUC | Measures precision-recall performance for missed prompts |
| Nonresponse recall | Percentage of actual nonresponses detected |
| Nonresponse precision | Percentage of predicted nonresponses that are genuine nonresponses |
| F1-score | Balance between precision and recall |
| F2-score | Places greater emphasis on recall |
| Balanced accuracy | Gives both classes equal importance |
| Log loss | Evaluates predicted probability quality |
| Brier score | Measures probability prediction error |
| Calibration | Determines whether predicted probabilities match observed frequencies |
| ROC-AUC | Measures ranking/discrimination ability |

## Recommended validation strategies

Evaluate generalization under different conditions:

1. **Chronological validation:** Train on earlier prompts and predict later prompts for known participants.
2. **Participant-held-out validation:** Train on some participants and predict responses from unseen participants.
3. **Dataset-held-out validation:** Train on some OpenESM datasets and evaluate on an entirely different dataset.

These answer different questions and should not be treated as interchangeable.

---

# 26. Important Implementation Considerations

## Small datasets

Many OpenESM datasets have relatively few participants and prompts.

Features with very little historical data can produce unstable estimates.

Beta smoothing helps with response rates, but it does not eliminate uncertainty.

For small datasets, consider prioritizing:

- Prior cumulative response rate.
- Previous prompt response.
- Previous-day same-beep response.
- Smoothed participant × beep response rate.
- Short rolling response rates.

## Redundant features

Some features contain overlapping information.

Examples:

- `prev_response` and `prev_1_response`.
- `scheduled_beeps_in_day` and `prompts_in_day`.
- `prior_missed_beep_count` and `prior_cumulative_nonresponses`.
- Cumulative, rolling, and EWM response rates.
- Raw and smoothed beep-specific response rates.

This is not necessarily a problem, especially when using regularized models, but it should be considered during feature selection and ablation.

## Missing values

Missing historical features are often meaningful.

For example:

- A missing previous response may mean there is no earlier prompt.
- A missing same-beep response rate may mean that beep has never occurred before.
- A missing days-since-previous-response value may mean the participant has never responded.

Avoid indiscriminately replacing all such missing values with zero.

Some models can handle missingness directly; others require imputation and potentially separate availability indicators.

---

# 27. Final Feature Engineering Objectives

The feature engineering process is intended to capture several distinct dimensions of nonresponse behavior:

| Dimension | What it represents |
|---|---|
| General participant compliance | Whether someone is generally responsive |
| Short-term response persistence | Whether recent responses predict subsequent responses |
| Beep-specific behavior | Whether a participant responds consistently to particular beep positions |
| Within-day behavior | Whether response probability changes throughout a day |
| Daily participation | Whether a participant maintains engagement across study days |
| Behavioral changes | Whether response behavior is improving or deteriorating |
| Temporal study effects | Whether response probability changes as the study progresses |

The most important research question is whether participant-specific historical behavior, particularly beep-specific response tendencies, provides predictive information beyond general cumulative response history.

The objective is not to maximize the number of features. It is to identify which behavioral patterns provide reproducible improvements in nonresponse prediction across individual OpenESM datasets.