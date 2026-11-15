# Regularized Logistic Regression with Nested Temporal Validation

## Methodology and implementation guide

**Scope:** Binary outcome prediction from chronologically ordered, repeated observations. This document is intentionally dataset-agnostic. It describes the implemented modeling procedure, its assumptions, its outputs, and its limitations, rather than results from any particular dataset.

## 1. Objective and prediction setting

The objective is to estimate the probability of a binary event at a specified prediction time using only information that would have been available at that time.

For observation \(i\), let \(Y_i\in\{0,1\}\), with **1 representing the event of interest (the positive class)**. The model estimates

\[
\hat p_i = P(Y_i=1\mid \mathbf{x}_i,\mathcal H_{<t_i}),
\]

where \(\mathbf{x}_i\) is the available feature vector and \(\mathcal H_{<t_i}\) denotes eligible history preceding the prediction. The probability is the primary model output; a binary decision can subsequently be obtained from a calibrated threshold.

Repeated observations may be grouped by subject, entity, or other unit. The evaluation described here is **future-observation prediction for a population whose prior observations may already be available**, not necessarily generalization to entirely unseen entities.

## 2. End-to-end workflow

1. Ingest a curated longitudinal table.
2. Validate the outcome, entity identifier, and chronological ordering variables.
3. Construct a consistent binary target and exclude known outcome-derived or otherwise unavailable fields.
4. Select numeric predictors.
5. Precompute outer and inner expanding-window folds and expected fitting workload.
6. For each outer fold, tune the regularization parameters using **only the outer training history**.
7. Generate historical out-of-sample probabilities inside that history and select a classification threshold.
8. Fit the selected regularized logistic model on the complete outer training history.
9. Predict probabilities and thresholded classes for the untouched outer assessment period.
10. Pool outer predictions for evaluation, inspect fold-level results, and export reproducibility artifacts.

```text
Chronologically ordered observations
                |
        Outer training history -------------------> Outer assessment
                |                                         ^
        Inner expanding-window folds                      |
                |                                         |
        Tune alpha and lambda                             |
                |                                         |
        Historical inner-fold predictions                 |
                |                                         |
        Calibrate classification threshold                |
                |                                         |
        Refit on all outer training history ---------------+
                |
        Predict outer assessment probabilities and classes
                |
        Repeat over later outer assessment periods
                |
        Aggregate out-of-sample metrics and export results
```

## 3. Statistical model

Regularized logistic regression models the log odds of the positive class as

\[
\operatorname{logit}(p_i) = \log\left(\frac{p_i}{1-p_i}\right)
= \beta_0 + \sum_{j=1}^{p}\beta_j x_{ij}.
\]

The inverse-logit converts the linear predictor into a probability between zero and one:

\[
p_i=\frac{1}{1+\exp[-(\beta_0+\mathbf{x}_i^\top\boldsymbol\beta)]}.
\]

The model is fit with an **elastic-net penalty**, allowing ridge, lasso, and intermediate regularization:

\[
\min_{\beta_0,\boldsymbol\beta}\left\{
-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\lambda\left[\frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2+\alpha\|\boldsymbol\beta\|_1\right]
\right\}.
\]

- **\(\alpha=0\): ridge**, which shrinks coefficients without generally setting them to zero.
- **\(0<\alpha<1\): elastic net**, combining ridge and lasso behavior.
- **\(\alpha=1\): lasso**, which may set coefficients to exactly zero.
- **\(\lambda>0\)** controls penalty strength: larger values generally imply more shrinkage.

The intercept is not penalized in the standard `glmnet` formulation. The implementation uses `glmnet::glmnet(family = "binomial", standardize = TRUE)`; the package standardizes predictors internally during fitting and returns coefficients on the original scale. It does **not** fit random effects, subject-specific intercepts, or a hierarchical model.

## 4. Features and fold-specific preprocessing

The implementation expects an already engineered longitudinal table. It excludes identifiers, chronology keys, the target, and an explicit list of known leakage fields from the predictors. The remaining numeric columns are candidates.

For **every inner training/assessment pair and every final outer training/assessment pair**, the following transformations are learned using training rows only:

1. Remove predictors with no finite training values.
2. Calculate the median of each retained predictor on finite training observations.
3. Replace missing or nonfinite values in both training and assessment rows using those **training medians**.
4. Calculate feature variance on the imputed training matrix.
5. Remove zero-variance or invalid-variance predictors using training statistics only.
6. Pass the resulting matrices to `glmnet`, which handles its own feature standardization.

These steps avoid learning imputation values or feature eligibility from assessment observations. They do **not** automatically make upstream engineered features leakage-free. Every lag, rolling statistic, cumulative count, aggregation, and external covariate must be audited for availability at the exact prediction time. In particular, precomputed full-day summaries can leak later observations into earlier predictions.

Only numeric predictors are modeled directly; categorical variables require appropriate upstream encoding. Feature selection here is a rule-based column filter, not a separate supervised feature-selection algorithm.

## 5. Expanding-window outer validation

The outer loop estimates future predictive performance. At each fold \(k\):

\[
\mathcal T_k=\{i:t_i<t_k\},\qquad
\mathcal A_k=\{i:t_i=t_k\},
\]

where the exact definition of \(t_k\) depends on the configured validation granularity.

**Across-period mode:** Train on all preceding periods and assess on all eligible observations in the next period.

**Within-period mode:** Train on all preceding periods **plus earlier ordered events within the current period**; assess on the current event position. For example, when the within-period order is indexed by a prompt number, observations from earlier prompts of that same period may be used for later prompts.

The history expands as validation progresses; observations are not shuffled. A configurable minimum number of initial periods must be available before outer assessment begins. Early rows used only for training will not receive outer out-of-sample predictions under this protocol.

**Simultaneity assumption:** All observations sharing an assessment time/event position are predicted together. Their outcomes cannot be used to predict each other within that fold. If observations are not truly available simultaneously, the temporal ordering needs to reflect the actual prediction schedule.

## 6. Nested inner temporal validation

For each outer fold, inner expanding-window folds are created **only from its training history**. The inner granularity matches the outer granularity in the implementation. Inner folds whose training partition does not contain both outcome classes are skipped. A minimum usable inner-fold count is required.

For each candidate \(\alpha\):

1. Fit one `glmnet` regularization path on each inner training partition.
2. Predict its corresponding future inner assessment partition at **every candidate \(\lambda\)**.
3. Compute binomial deviance for each fold/lambda combination.
4. Average the fold deviances for each lambda.
5. Compute the standard error across folds for each lambda.

The per-observation binomial deviance is

\[
d(y_i,\hat p_i)=-2\left[y_i\log(\hat p_i)+(1-y_i)\log(1-\hat p_i)\right],
\]

with predicted probabilities clipped away from exactly zero and one for numerical stability. The fold deviance is the mean across observations in that fold.

**Important aggregation detail:** The implementation uses the **unweighted mean of fold-level mean deviances**. Thus each fold has equal weight, even if the folds contain different numbers of observations. The standard error is the standard deviation of those fold-level scores divided by the square root of the number of folds. Temporal folds may overlap in training history and have dependent errors, so the one-standard-error heuristic should not be interpreted as a formal confidence interval.

### Lambda selection rules

For each alpha, two lambda candidates are derived:

- **`lambda.min`:** The lambda with the lowest mean inner-validation deviance.
- **`lambda.1se`:** The **largest lambda** whose mean deviance lies within one estimated standard error of the minimum.

The implementation then compares the selected `lambda.min` and `lambda.1se` candidates **across all alpha values**, choosing the candidate with the lowest mean deviance. In exact ties, it prefers `lambda.1se`, then larger alpha. This means `lambda.1se` is **not automatically** the final choice; `lambda.min` often wins because it attains the minimum within its alpha path.

A single fit at one alpha computes predictions over the full lambda path. Accordingly, the number of `glmnet` fitting calls is **not** the number of alpha-lambda combinations multiplied by the number of folds.

## 7. Historical temporal threshold calibration

A probability model needs no classification threshold to estimate risk. A threshold is required only to produce a binary decision:

\[
\hat y_i=\mathbb{1}(\hat p_i\ge\tau).
\]

The implementation selects a **separate threshold for each outer fold**, without consulting that fold's assessment outcomes:

1. Take the alpha/lambda combination selected by the inner tuning procedure.
2. Refit that selected alpha on each inner training partition, using the same candidate lambda path.
3. Predict each corresponding inner assessment partition at the selected lambda.
4. Pool the historical inner assessment probabilities and their true labels.
5. Evaluate a configurable grid of thresholds against a chosen classification metric.
6. Select the best-performing historical threshold and apply it to the subsequent outer assessment probabilities.

The default threshold-selection criterion in the examined implementation is **Matthews correlation coefficient (MCC)**, though the metric and grid are configurable. MCC incorporates all four confusion-matrix cells:

\[
\operatorname{MCC}=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
\]

When the denominator is zero, MCC is undefined; calibration therefore requires appropriate handling of degenerate threshold candidates and class distributions. The implementation requires both outcome classes in the pooled calibration labels and relies on the threshold utility to find a usable optimum.

### Critical interpretation

The calibration predictions are **out of sample relative to each inner training fit**, and the outer assessment labels are not used for threshold selection. Therefore the outer-fold classification evaluation does not directly tune its threshold on the same outer outcomes it evaluates.

However, **the same inner assessment observations are reused for hyperparameter selection and threshold calibration**. The resulting calibration score is not an independent unbiased estimate of threshold performance. A stricter design would reserve a separate later calibration block inside each outer training window, or use another level of nested temporal splitting. That is **not** what the examined implementation does.

Threshold calibration here is **decision-threshold selection**, not probability calibration. It does not fit Platt scaling, isotonic regression, or another mapping intended to correct probability reliability.

## 8. Final outer refit and prediction

After tuning and threshold calibration for outer fold \(k\):

1. Recompute preprocessing from **all** outer training observations.
2. Fit a `glmnet` path at the selected alpha.
3. Predict the outer assessment rows at the selected lambda.
4. Apply that fold's previously selected threshold to obtain binary classifications.
5. Store the probability, predicted class, threshold, selected hyperparameters, and fold metadata.

Each assessment row is predicted using only historical training labels. Later outer folds may legitimately train on observations that were assessments in earlier folds, because those outcomes would be known by the later prediction time.

The workflow repeatedly refits models; it is not an incremental online-learning update of a single coefficient vector.

## 9. Performance evaluation

### Threshold-independent probability metrics

The implementation produces pooled outer-validation probability metrics such as:

- **ROC-AUC:** Ranking performance across false-positive/true-positive tradeoffs.
- **PR-AUC:** Precision-recall behavior, especially relevant when positive events are relatively uncommon.
- **Log loss / binomial deviance:** Penalty for inaccurate probability assignments, with large penalties for confident errors.
- **Brier score:** Mean squared difference between predicted probabilities and binary outcomes.

These metrics evaluate the probability predictions without depending on the chosen classification threshold. They do not by themselves guarantee probability calibration; calibration curves or reliability diagnostics are useful additional analyses.

### Threshold-dependent classification metrics

The implementation also evaluates classifications obtained from the **fold-specific historical thresholds**, including metrics such as MCC, precision, recall, specificity, accuracy, balanced accuracy, F1/F-beta, and confusion-matrix counts.

Because thresholds can differ across outer folds, pooled calibrated metrics describe a **sequence of historically adapted decision rules**, not a single universal threshold.

### Exploratory pooled threshold analysis

The implementation separately searches for the best threshold using **all pooled outer predictions and their labels**. This is labeled **exploratory**. Its peak score is optimistically selected on the evaluation outcomes and must **not** be reported as unbiased deployed classification performance. It is **not** used to overwrite the fold-specific calibrated predictions.

### Dependence and uncertainty

Repeated observations from the same entities and adjacent times are correlated. A single pooled metric does not establish independent-sample uncertainty. Confidence intervals, if required, should account for the dependence structure (for example, by appropriate clustered or temporal resampling). Performance may also vary across entities, time periods, and event positions; stratified diagnostics are recommended.

## 10. Computational workload

The implementation precomputes the complete temporal fold plan before fitting so that progress reports use a fixed denominator.

Let:

- \(K\) = number of outer folds;
- \(A\) = number of alpha candidates;
- \(F_k\) = number of usable inner folds for outer fold \(k\);
- \(L\) = number of lambda values in each fitted path.

The number of `glmnet` fitting calls is

\[
N_{\mathrm{tuning}}=A\sum_{k=1}^{K}F_k,
\]

\[
N_{\mathrm{calibration}}=\sum_{k=1}^{K}F_k,
\]

\[
N_{\mathrm{outer}}=K,
\]

and therefore

\[
\boxed{N_{\mathrm{total}}=(A+1)\sum_{k=1}^{K}F_k+K.}
\]

The calibration term reflects **additional refits** of the selected alpha on historical inner training partitions. The number of lambda candidates \(L\) affects the size of each fitted path and prediction/deviance calculations, but it is **not an additional multiplier of the `glmnet` call count**.

Runtime depends on the number and sizes of folds, rows, retained predictors, sparsity, optimization behavior, regularization path length, and computing environment. Expanding-window validation repeatedly processes overlapping training histories and can be computationally expensive. Precomputing fold metadata itself can also take time before model fitting starts.

A **preflight mode** can restrict execution to an initial assessable portion of the timeline to test the pipeline before a full experiment. Preflight metrics are diagnostic and should not be treated as final performance estimates.

## 11. Saved artifacts and auditability

The examined implementation writes the following CSV outputs:

| File | Purpose |
|---|---|
| `predictions.csv` | Outer out-of-sample probabilities, calibrated classes, fold thresholds, and identifiers |
| `metrics.csv` | Threshold-grid and threshold-independent evaluation metrics |
| `calibrated_metrics.csv` | Metrics using the historical fold-specific decision thresholds |
| `threshold_performance.csv` | Exploratory pooled threshold performance curve |
| `tuning_results.csv` | Candidate alpha/lambda comparison for each outer fold |
| `selected_hyperparameters_by_fold.csv` | Chosen hyperparameters and associated inner-validation scores |
| `inner_temporal_folds.csv` | Inner fold metadata and fitting audit trail |
| `calibration_by_outer_fold.csv` | Fold-specific thresholds and calibration summary |
| `calibration_threshold_curves.csv` | Historical threshold-performance curves for each outer fold |
| `selected_alpha_counts.csv` | Frequency of selected alpha values |
| `selected_lambda_counts.csv` | Frequency of selected lambda rules |
| `candidate_features.csv` | Numeric predictor names admitted to modeling |
| `outer_workload_plan.csv` | Planned fold sizes and fitting counts |
| `model_configuration.csv` | Experiment settings and workload totals |

These files support inspection of temporal splits, hyperparameter stability, threshold variability, prediction quality, and execution workload. They are **evaluation artifacts**, not a serialized model ready for production inference.

## 12. Assumptions and limitations

1. **Correct chronology:** The time and within-period ordering fields must match when observations and labels actually become available.
2. **Point-in-time feature availability:** Feature engineering performed upstream must exclude future information, including target-derived summaries and data that arrive with reporting delays.
3. **Population transfer:** The protocol tests future observations of the observed population. It does not automatically estimate performance on new, never-seen entities; that requires a different evaluation design.
4. **Pooled coefficients:** Standard `glmnet` estimates a common coefficient vector, not hierarchical entity-specific effects. Entity heterogeneity may be represented through engineered features and interactions, but it is not modeled as random effects.
5. **Missingness handling:** Median imputation is simple and deterministic but may discard information carried by missingness; missingness indicators can be engineered upstream where justified.
6. **Calibration reuse:** Inner validation observations are used both to choose hyperparameters and to choose a threshold. Outer labels remain untouched, but inner calibration is not fully independent.
7. **No probability recalibration:** Selecting a threshold does not improve probability calibration.
8. **Temporal drift:** Changes in behavior or event prevalence may degrade model and threshold performance over time.
9. **Fold weighting:** Inner mean deviance gives equal weight to folds, not necessarily to observations.
10. **Metric choice:** MCC is one defensible criterion, but the best operational threshold may depend on asymmetric costs, capacity limits, or intervention policy.
11. **Early observations:** Rows before the first eligible outer assessment are training history and do not have outer out-of-sample predictions.
12. **No execution guarantee:** This methodology documents the inspected R implementation; it does not assert that the script and all companion modules have been successfully executed together in a specific R environment.

## 13. Recommended reporting language

> We evaluated a regularized logistic regression model using nested, expanding-window temporal validation. Within each outer training window, an inner temporal cross-validation procedure selected the elastic-net mixing parameter and regularization strength by mean binomial deviance. Missing-value imputation and predictor filtering were estimated exclusively from each training partition. Historical out-of-sample inner-fold predictions were then used to select a classification threshold, after which the selected model was refitted on the complete outer training window and applied to the next assessment period. Predictive performance was assessed from pooled outer out-of-sample probabilities and classifications using thresholds selected without access to the corresponding outer assessment outcomes. Hyperparameter tuning and threshold selection reused the same inner validation observations; consequently, calibration scores were treated as internal selection diagnostics rather than independent performance estimates.

## 14. Implementation reference

The procedure is implemented across a main experiment script and supporting modules for ingestion, longitudinal preprocessing, temporal validation, classification metrics, threshold optimization, and terminal progress tracking. The main implementation uses R and `glmnet`; evaluation utilities use `yardstick` where applicable.

**Document status:** Dataset-agnostic technical description of the inspected updated regularized logistic regression script. Parameter values, dataset identifiers, cohort sizes, and observed results are intentionally omitted.
