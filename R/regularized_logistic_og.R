# ============================================================
# OpenESM Nonresponse Prediction
# Tuned Regularized Logistic Regression
# Nested Expanding-Window Cross-Validation
# With Temporal Threshold Calibration
# ============================================================
# Run from repository root: Rscript R/regularized_logistic.R
# Target: 0 = response, 1 = nonresponse.
# Outer assessment outcomes are never used for tuning/calibration.
# ============================================================

# ============================================================
# 1. CONFIGURATION
# ============================================================
dataset_id <- "0006"
config_path <- "config/storage.yaml"
target_col <- "responded"
id_col <- "id"
day_col <- "day"
beep_col <- "beep"
validation_mode <- "within_day"
initial_days <- 10L
inner_initial_days <- 5L
inner_validation_mode <- validation_mode
alpha_grid <- c(0, 0.25, 0.50, 0.75, 1)
lambda_grid <- exp(seq(log(100), log(0.0001), length.out = 50L))
lambda_choices <- c("lambda.min", "lambda.1se")
minimum_inner_folds <- 3L
threshold_grid <- seq(0.01, 0.99, by = 0.02)
threshold_metric <- "MCC"
beta <- 1
random_seed <- 2026
# TRUE: use first assessable day only; in within_day mode it has several beeps.
preflight_only <- FALSE
output_dir <- file.path("analysis", "regularized_logistic",
                            paste0("dataset_", dataset_id))
if (preflight_only) output_dir <- file.path(output_dir, "preflight")

# ============================================================
# 2. LOAD REQUIRED MODULES
# ============================================================
module_paths <- c("R/src/ingestion.R", "R/src/preprocessing.R",
                  "R/src/validations.R", "R/src/metrics.R",
                  "R/src/thresholds.R", "R/utils.R")
missing_modules <- module_paths[!file.exists(module_paths)]
if (length(missing_modules)) {
  stop("Missing R modules: ", paste(missing_modules, collapse = ", "),
       ". Run from repository root.")
}
for (module_path in module_paths) source(module_path)

# ============================================================
# 3. CHECK REQUIRED PACKAGES
# ============================================================
required_packages <- c("glmnet", "yardstick", "arrow", "aws.s3")
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Missing packages: ", paste(missing_packages, collapse = ", "))
}

# ============================================================
# 4. VALIDATE CONFIGURATION
# ============================================================
if (!is.numeric(alpha_grid) || !length(alpha_grid) ||
    any(!is.finite(alpha_grid)) || any(alpha_grid < 0 | alpha_grid > 1) ||
    anyDuplicated(alpha_grid)) stop("Invalid alpha_grid.")
if (!is.numeric(lambda_grid) || length(lambda_grid) < 2L ||
    any(!is.finite(lambda_grid)) || any(lambda_grid <= 0) ||
    any(diff(lambda_grid) >= 0)) stop("lambda_grid must decrease strictly.")
if (!length(lambda_choices) || anyDuplicated(lambda_choices) ||
    any(!lambda_choices %in% c("lambda.min", "lambda.1se"))) {
  stop("Invalid lambda_choices.")
}
if (!validation_mode %in% c("within_day", "across_day") ||
    inner_validation_mode != validation_mode) stop("Invalid validation modes.")
for (name in c("initial_days", "inner_initial_days", "minimum_inner_folds")) {
  value <- get(name)
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) ||
      value != floor(value) ||
      value < (if (name == "minimum_inner_folds") 2 else 1)) {
    stop(name, " must be a positive integer (minimum_inner_folds >= 2).")
  }
}
if (!is.numeric(threshold_grid) || !length(threshold_grid) ||
    any(!is.finite(threshold_grid)) || any(threshold_grid < 0 | threshold_grid > 1)) {
  stop("Invalid threshold_grid.")
}
if (length(beta) != 1L || !is.numeric(beta) || !is.finite(beta) || beta <= 0) {
  stop("beta must be positive.")
}
set.seed(random_seed)

# ============================================================
# 5. LOAD ONE CURATED DATASET
# ============================================================
message("Loading curated dataset: ", dataset_id)
raw_data <- load_curated_dataset(dataset_id = dataset_id, config_path = config_path)
if (!is.data.frame(raw_data)) stop("Ingestion must return a data.frame.")
message("Loaded ", nrow(raw_data), " observations and ", ncol(raw_data), " columns.")

# ============================================================
# 6. PREPROCESS DATASET
# ============================================================
message("Preprocessing longitudinal dataset.")
prepared_data <- preprocess_longitudinal_data(
  df = raw_data, target_col = target_col, id_col = id_col,
  day_col = day_col, beep_col = beep_col, drop_missing_target = TRUE)
rm(raw_data)
if (!is.data.frame(prepared_data) || !"nonresponse" %in% names(prepared_data)) {
  stop("Preprocessing must return a data.frame with nonresponse.")
}
message("Prepared observations: ", nrow(prepared_data))
message("Participants: ", length(unique(prepared_data[[id_col]])))
message("Study days: ", length(unique(prepared_data[[day_col]])))
message("Responses: ", sum(prepared_data$nonresponse == 0))
message("Nonresponses: ", sum(prepared_data$nonresponse == 1))
if (preflight_only) {
  days <- sort(unique(prepared_data[[day_col]]))
  if (length(days) <= initial_days) stop("Insufficient days for preflight.")
  last_day <- days[initial_days + 1L]
  prepared_data <- prepared_data[prepared_data[[day_col]] <= last_day, , drop = FALSE]
  rownames(prepared_data) <- NULL
  message("PREFLIGHT: restricted to first ", initial_days + 1L, " days.")
}

# ============================================================
# 7. SELECT PREDICTOR FEATURES
# ============================================================
known_leakage_features <- c(
  "responded_any", "responded_any_today", "daily_response_rate",
  "missed_day", "cumulative_responses", "cumulative_prompts",
  "cumulative_response_rate", "last_observed_day", "last_response_day",
  "first_response_day", "total_observed_prompts", "total_responses",
  "overall_response_rate", "never_responded")
excluded_features <- unique(c(id_col, day_col, beep_col, target_col,
                              "nonresponse", known_leakage_features))
candidate_features <- setdiff(names(prepared_data), excluded_features)
features <- candidate_features[vapply(prepared_data[candidate_features],
                                      is.numeric, logical(1))]
if (!length(features)) stop("No numeric predictors found.")
message("Selected numeric predictor features: ", length(features))
print(features)
message("Verify every selected feature is known before its prediction beep.")

# ============================================================
# 8. CREATE INNER EXPANDING-WINDOW TEMPORAL FOLDS
# ============================================================
make_inner_temporal_splits <- function(data, mode, initial_days,
                                      id_col, day_col, beep_col) {
  unique_days <- sort(unique(data[[day_col]]))
  if (length(unique_days) <= initial_days) return(list())
  splits <- list()
  for (current_day in unique_days[seq.int(initial_days + 1L, length(unique_days))]) {
    if (mode == "across_day") {
      train <- data[data[[day_col]] < current_day, , drop = FALSE]
      test <- data[data[[day_col]] == current_day, , drop = FALSE]
      if (nrow(train) && nrow(test)) {
        splits[[length(splits) + 1L]] <- list(
          train_data = train, test_data = test,
          fold = paste0("inner_across_day_", current_day),
          validation_day = current_day, validation_beep = NA_real_)
      }
    } else {
      beeps <- sort(unique(data[[beep_col]][data[[day_col]] == current_day]))
      for (current_beep in beeps) {
        train <- data[data[[day_col]] < current_day |
                        (data[[day_col]] == current_day &
                           data[[beep_col]] < current_beep), , drop = FALSE]
        test <- data[data[[day_col]] == current_day &
                       data[[beep_col]] == current_beep, , drop = FALSE]
        if (nrow(train) && nrow(test)) {
          splits[[length(splits) + 1L]] <- list(
            train_data = train, test_data = test,
            fold = paste0("inner_within_day_", current_day, "_beep_", current_beep),
            validation_day = current_day, validation_beep = current_beep)
        }
      }
    }
  }
  splits
}

# ============================================================
# 9. FILTER USABLE INNER FOLDS
# ============================================================
filter_usable_inner_splits <- function(splits, target) {
  Filter(function(split) {
    y <- as.integer(split$train_data[[target]])
    !anyNA(y) && length(unique(y)) == 2L
  }, splits)
}

# ============================================================
# 10. PRECOMPUTE GLOBAL OUTER WORKLOAD
# ============================================================
build_outer_workload_plan <- function(data, mode, initial_days,
                                      inner_initial_days, id_col, day_col,
                                      beep_col, target, minimum_inner_folds,
                                      alpha_count) {
  unique_days <- sort(unique(data[[day_col]]))
  if (length(unique_days) <= initial_days) stop("Insufficient outer days.")
  plan_rows <- list()
  for (day in unique_days[seq.int(initial_days + 1L, length(unique_days))]) {
    beeps <- if (mode == "within_day") {
      sort(unique(data[[beep_col]][data[[day_col]] == day]))
    } else NA_real_
    for (beep in beeps) {
      if (mode == "within_day") {
        train <- data[data[[day_col]] < day |
                        (data[[day_col]] == day & data[[beep_col]] < beep), , drop = FALSE]
        test <- data[data[[day_col]] == day & data[[beep_col]] == beep, , drop = FALSE]
      } else {
        train <- data[data[[day_col]] < day, , drop = FALSE]
        test <- data[data[[day_col]] == day, , drop = FALSE]
      }
      if (!nrow(train) || !nrow(test)) next
      y <- as.integer(train[[target]])
      if (anyNA(y) || length(unique(y)) != 2L) {
        stop("Outer training lacks both classes at day ", day, ", beep ", beep)
      }
      inner <- filter_usable_inner_splits(make_inner_temporal_splits(
        train, mode, inner_initial_days, id_col, day_col, beep_col), target)
      if (length(inner) < minimum_inner_folds) {
        stop("Insufficient inner folds at day ", day, ", beep ", beep)
      }
      plan_rows[[length(plan_rows) + 1L]] <- data.frame(
        outer_fold = length(plan_rows) + 1L,
        assessment_day = day, assessment_beep = beep,
        training_max_day = max(train[[day_col]]),
        training_rows = nrow(train), assessment_rows = nrow(test),
        inner_fold_count = length(inner),
        inner_fit_count = length(inner) * alpha_count,
        calibration_fit_count = length(inner), stringsAsFactors = FALSE)
    }
  }
  if (!length(plan_rows)) stop("No outer folds generated.")
  plan <- do.call(rbind, plan_rows)
  rownames(plan) <- NULL
  plan
}

# ============================================================
# 11. PREPARE MODEL MATRICES
# ============================================================
prepare_model_matrices <- function(train_data, test_data, features) {
  if (!nrow(train_data) || !nrow(test_data)) stop("Empty model split.")
  x_train <- as.matrix(train_data[, features, drop = FALSE])
  x_test <- as.matrix(test_data[, features, drop = FALSE])
  storage.mode(x_train) <- "double"
  storage.mode(x_test) <- "double"
  if (!identical(colnames(x_train), colnames(x_test))) stop("Matrix columns differ.")
  usable <- vapply(seq_len(ncol(x_train)), function(j) {
    any(is.finite(x_train[, j]))
  }, logical(1))
  x_train <- x_train[, usable, drop = FALSE]
  x_test <- x_test[, usable, drop = FALSE]
  if (!ncol(x_train)) stop("No usable predictors.")
  medians <- vapply(seq_len(ncol(x_train)), function(j) {
    stats::median(x_train[is.finite(x_train[, j]), j])
  }, numeric(1))
  for (j in seq_len(ncol(x_train))) {
    x_train[!is.finite(x_train[, j]), j] <- medians[j]
    x_test[!is.finite(x_test[, j]), j] <- medians[j]
  }
  variable <- vapply(seq_len(ncol(x_train)), function(j) {
    v <- stats::var(x_train[, j])
    is.finite(v) && v > 0
  }, logical(1))
  x_train <- x_train[, variable, drop = FALSE]
  x_test <- x_test[, variable, drop = FALSE]
  if (!ncol(x_train)) stop("All predictors are constant.")
  list(x_train = x_train, x_test = x_test,
       retained_features = colnames(x_train))
}

# ============================================================
# 12. BINOMIAL DEVIANCE FUNCTION
# ============================================================
calculate_binomial_deviance <- function(y_true, y_probability) {
  if (!length(y_true) || length(y_true) != length(y_probability) ||
      anyNA(y_true) || !all(y_true %in% c(0L, 1L)) ||
      any(!is.finite(y_probability)) ||
      any(y_probability < 0 | y_probability > 1)) {
    stop("Invalid binomial deviance inputs.")
  }
  p <- pmin(pmax(y_probability, 1e-15), 1 - 1e-15)
  -2 * mean(y_true * log(p) + (1 - y_true) * log1p(-p))
}

# ============================================================
# 13. GLOBAL PROGRESS AND LOG STATE
# ============================================================
tuning_log <- list()
inner_fold_log <- list()
calibration_log <- list()
calibration_curve_log <- list()
outer_completed <- 0L
global_progress <- NULL
outer_fold_plan <- NULL
planned_inner_fits <- 0
planned_calibration_fits <- 0
planned_outer_refits <- 0
planned_total_fits <- 0

# ============================================================
# 14. TUNE ALPHA/LAMBDA AND CALIBRATE TEMPORAL THRESHOLD
# ============================================================
fit_tuned_regularized_logistic <- function(train_data, test_data, features, target) {
  y_outer_train <- as.integer(train_data[[target]])
  if (anyNA(y_outer_train) || length(unique(y_outer_train)) != 2L) {
    stop("Outer training target must contain both binary classes.")
  }

  # 14A. Identify and validate the outer fold against the workload plan.
  current_outer_fold <- outer_completed + 1L
  if (current_outer_fold > nrow(outer_fold_plan)) stop("Too many outer folds.")
  planned <- outer_fold_plan[current_outer_fold, , drop = FALSE]
  assessment_day <- unique(test_data[[day_col]])
  assessment_beeps <- unique(test_data[[beep_col]])
  if (length(assessment_day) != 1L || assessment_day != planned$assessment_day ||
      nrow(train_data) != planned$training_rows ||
      nrow(test_data) != planned$assessment_rows) {
    stop("Outer fold does not match workload plan.")
  }
  if (validation_mode == "within_day" &&
      (length(assessment_beeps) != 1L ||
       assessment_beeps != planned$assessment_beep)) {
    stop("Assessment beep differs from workload plan.")
  }

  # 14B. Recreate inner folds using only outer training history.
  inner_splits <- filter_usable_inner_splits(make_inner_temporal_splits(
    train_data, inner_validation_mode, inner_initial_days,
    id_col, day_col, beep_col), target)
  n_inner <- length(inner_splits)
  if (n_inner < minimum_inner_folds || n_inner != planned$inner_fold_count) {
    stop("Inner fold count differs from workload plan.")
  }
  n_lambda <- length(lambda_grid)

  # 14C. Evaluate every alpha/lambda path on identical temporal folds.
  alpha_results <- vector("list", length(alpha_grid))
  alpha_fits <- vector("list", length(alpha_grid))
  for (alpha_index in seq_along(alpha_grid)) {
    alpha_value <- alpha_grid[alpha_index]
    fold_deviance <- matrix(NA_real_, nrow = n_inner, ncol = n_lambda)
    for (fold_index in seq_along(inner_splits)) {
      split <- inner_splits[[fold_index]]
      y_train <- as.integer(split$train_data[[target]])
      y_test <- as.integer(split$test_data[[target]])
      if (anyNA(y_test) || !all(y_test %in% c(0L, 1L))) {
        stop("Invalid inner assessment targets: ", split$fold)
      }
      matrices <- prepare_model_matrices(split$train_data, split$test_data, features)
      fit <- glmnet::glmnet(x = matrices$x_train, y = y_train,
                            family = "binomial", alpha = alpha_value,
                            lambda = lambda_grid, standardize = TRUE)
      pred <- as.matrix(stats::predict(fit, newx = matrices$x_test,
                                       s = lambda_grid, type = "response"))
      if (nrow(pred) != length(y_test) || ncol(pred) != n_lambda) {
        stop("Unexpected lambda prediction dimensions: ", split$fold)
      }
      for (j in seq_len(n_lambda)) {
        fold_deviance[fold_index, j] <- calculate_binomial_deviance(y_test, pred[, j])
      }
      global_progress$update_inner(current_outer_fold, nrow(outer_fold_plan),
                                   fold_index, n_inner, alpha_value)
      inner_fold_log[[length(inner_fold_log) + 1L]] <<- data.frame(
        alpha = alpha_value, outer_training_max_day = max(train_data[[day_col]]),
        outer_fold = current_outer_fold, inner_fold = split$fold,
        inner_training_rows = nrow(split$train_data),
        inner_validation_rows = nrow(split$test_data),
        inner_training_max_day = max(split$train_data[[day_col]]),
        inner_validation_day = split$validation_day,
        inner_validation_beep = split$validation_beep, stringsAsFactors = FALSE)
    }
    if (any(!is.finite(fold_deviance))) stop("Nonfinite inner deviance.")
    mean_deviance <- colMeans(fold_deviance)
    deviance_se <- apply(fold_deviance, 2, stats::sd) / sqrt(n_inner)
    minimum_index <- which.min(mean_deviance)
    one_se_limit <- mean_deviance[minimum_index] + deviance_se[minimum_index]
    eligible_one_se <- which(mean_deviance <= one_se_limit)
    one_se_index <- eligible_one_se[which.max(lambda_grid[eligible_one_se])]
    alpha_results[[alpha_index]] <- data.frame(
      alpha = alpha_value, lambda_index = seq_len(n_lambda),
      lambda = lambda_grid, mean_cv_deviance = mean_deviance,
      cv_deviance_se = deviance_se, stringsAsFactors = FALSE)
    alpha_fits[[alpha_index]] <- list(
      fold_deviance = fold_deviance, mean_deviance = mean_deviance,
      deviance_se = deviance_se, lambda_min_index = minimum_index,
      lambda_1se_index = one_se_index)
  }

  # 14D. Compare alpha/lambda candidates as in the original script.
  candidate_results <- list()
  for (alpha_index in seq_along(alpha_grid)) {
    alpha_fit <- alpha_fits[[alpha_index]]
    for (lambda_name in lambda_choices) {
      selected_index <- if (lambda_name == "lambda.min") {
        alpha_fit$lambda_min_index
      } else alpha_fit$lambda_1se_index
      candidate_results[[length(candidate_results) + 1L]] <- data.frame(
        alpha = alpha_grid[alpha_index], lambda_choice = lambda_name,
        lambda = lambda_grid[selected_index], lambda_index = selected_index,
        cv_deviance = alpha_fit$mean_deviance[selected_index],
        cv_deviance_se = alpha_fit$deviance_se[selected_index],
        number_of_inner_folds = n_inner,
        number_of_inner_validation_observations = sum(vapply(
          inner_splits, function(x) nrow(x$test_data), integer(1))),
        number_of_features = length(features), stringsAsFactors = FALSE)
    }
  }
  candidate_table <- do.call(rbind, candidate_results)
  rownames(candidate_table) <- NULL
  # Preserve original minimum-deviance selection (lambda.min usually wins).
  candidate_order <- order(candidate_table$cv_deviance,
                           candidate_table$lambda_choice != "lambda.1se",
                           -candidate_table$alpha)
  best_candidate_index <- candidate_order[1L]
  best_candidate <- candidate_table[best_candidate_index, , drop = FALSE]

  # 14E. Preserve complete candidate comparison log.
  candidate_table$outer_training_max_day <- max(train_data[[day_col]])
  candidate_table$assessment_day <- assessment_day
  candidate_table$assessment_beep <- if (length(assessment_beeps) == 1L) {
    assessment_beeps
  } else NA_real_
  candidate_table$outer_fold_scope <- if (length(assessment_beeps) == 1L) {
    "single_day_single_beep"
  } else "single_day_multiple_beeps"
  candidate_table$outer_fold <- current_outer_fold
  candidate_table$selected <- seq_len(nrow(candidate_table)) == best_candidate_index
  tuning_log[[length(tuning_log) + 1L]] <<- candidate_table

  # 14F. Calibration: for the selected alpha/lambda, REFIT on each
  # inner training fold, predict its later validation fold, then
  # optimize MCC using only those historical out-of-sample predictions.
  # This is leakage-free relative to the outer assessment, though
  # calibration and hyperparameter selection reuse inner validation data.
  calibration_truth <- integer(0)
  calibration_prob <- numeric(0)
  for (fold_index in seq_along(inner_splits)) {
    split <- inner_splits[[fold_index]]
    matrices <- prepare_model_matrices(split$train_data, split$test_data, features)
    fit <- glmnet::glmnet(
      x = matrices$x_train, y = as.integer(split$train_data[[target]]),
      family = "binomial", alpha = best_candidate$alpha,
      lambda = lambda_grid, standardize = TRUE)
    p <- as.numeric(stats::predict(
      fit, newx = matrices$x_test, s = best_candidate$lambda,
      type = "response"))
    y <- as.integer(split$test_data[[target]])
    if (length(p) != length(y) || any(!is.finite(p)) ||
        any(p < 0 | p > 1)) stop("Invalid calibration predictions.")
    calibration_truth <- c(calibration_truth, y)
    calibration_prob <- c(calibration_prob, p)
    global_progress$update_calibration(current_outer_fold, nrow(outer_fold_plan),
                                       fold_index, n_inner)
  }
  if (length(unique(calibration_truth)) < 2L) {
    stop("Calibration outcomes lack both classes in outer fold ", current_outer_fold)
  }
  calibration <- find_optimal_threshold(
    y_true = calibration_truth, y_prob = calibration_prob,
    metric = threshold_metric, beta = beta, thresholds = threshold_grid)
  calibrated_threshold <- calibration$optimal_threshold
  calibration_log[[length(calibration_log) + 1L]] <<- data.frame(
    outer_fold = current_outer_fold, assessment_day = assessment_day,
    assessment_beep = if (length(assessment_beeps) == 1L) assessment_beeps else NA_real_,
    selected_alpha = best_candidate$alpha,
    selected_lambda = best_candidate$lambda,
    selected_lambda_choice = best_candidate$lambda_choice,
    calibrated_threshold = calibrated_threshold,
    calibration_score = calibration$peak_score,
    calibration_rows = length(calibration_truth),
    calibration_folds = n_inner, stringsAsFactors = FALSE)
  curve <- calibration$performance_curve
  curve$outer_fold <- current_outer_fold
  calibration_curve_log[[length(calibration_curve_log) + 1L]] <<- curve

  # 14G. Final outer refit on the entire outer training window.
  outer_matrices <- prepare_model_matrices(train_data, test_data, features)
  final_fit <- glmnet::glmnet(
    x = outer_matrices$x_train, y = y_outer_train,
    family = "binomial", alpha = best_candidate$alpha,
    lambda = lambda_grid, standardize = TRUE)
  predicted_prob <- as.numeric(stats::predict(
    final_fit, newx = outer_matrices$x_test,
    s = best_candidate$lambda, type = "response"))
  if (length(predicted_prob) != nrow(test_data) ||
      any(!is.finite(predicted_prob)) ||
      any(predicted_prob < 0 | predicted_prob > 1)) {
    stop("Invalid final outer probabilities.")
  }
  global_progress$update_refit(current_outer_fold, nrow(outer_fold_plan))
  outer_completed <<- outer_completed + 1L

  # validations.R preserves metadata when model_func returns a data.frame.
  data.frame(
    .validation_row_id = test_data$.validation_row_id,
    predicted_prob = predicted_prob,
    predicted_class = as.integer(predicted_prob >= calibrated_threshold),
    calibrated_threshold = rep(calibrated_threshold, nrow(test_data)),
    selected_alpha = rep(best_candidate$alpha, nrow(test_data)),
    selected_lambda = rep(best_candidate$lambda, nrow(test_data)),
    selected_lambda_choice = rep(best_candidate$lambda_choice, nrow(test_data)),
    stringsAsFactors = FALSE)
}

# ============================================================
# 15. PRECOMPUTE GLOBAL WORKLOAD
# ============================================================
message("\nPreparing global workload estimate.")
message("Outer mode: ", validation_mode)
message("Outer initial days: ", initial_days)
message("Inner mode: ", inner_validation_mode)
message("Inner initial days: ", inner_initial_days)
message("Alpha candidates: ", paste(alpha_grid, collapse = ", "))
message("Lambda candidates: ", length(lambda_grid))
outer_fold_plan <- build_outer_workload_plan(
  data = prepared_data, mode = validation_mode, initial_days = initial_days,
  inner_initial_days = inner_initial_days, id_col = id_col,
  day_col = day_col, beep_col = beep_col, target = "nonresponse",
  minimum_inner_folds = minimum_inner_folds, alpha_count = length(alpha_grid))
outer_fold_count <- nrow(outer_fold_plan)
planned_inner_fits <- sum(outer_fold_plan$inner_fit_count)
planned_calibration_fits <- sum(outer_fold_plan$calibration_fit_count)
planned_outer_refits <- outer_fold_count
planned_total_fits <- planned_inner_fits + planned_calibration_fits + planned_outer_refits
planned_inner_folds <- sum(outer_fold_plan$inner_fold_count)
message("\n================================================")
message("PRECOMPUTED MODEL FITTING WORKLOAD")
message("================================================")
message("Outer validation folds: ", outer_fold_count)
message("Unique inner temporal folds: ", planned_inner_folds)
message("Inner glmnet fits: ", planned_inner_fits)
message("Calibration glmnet fits: ", planned_calibration_fits)
message("Final outer refits: ", planned_outer_refits)
message("Total glmnet calls: ", planned_total_fits)
message("================================================")

# ============================================================
# 16. INITIALIZE GLOBAL PROGRESS TRACKER
# ============================================================
global_progress <- create_global_progress(
  total_inner_fits = planned_inner_fits,
  total_outer_fits = planned_outer_refits,
  total_calibration_fits = planned_calibration_fits,
  refresh_seconds = 1
)

outer_completed <- 0L

# ============================================================
# 17. RUN OUTER WALK-FORWARD VALIDATION
# ============================================================
message("\nStarting nested temporal validation with calibration.")
validation_results <- run_walk_forward_validation(
  df = prepared_data, features = features, target = "nonresponse",
  model_func = fit_tuned_regularized_logistic,
  mode = validation_mode, initial_days = initial_days,
  id_col = id_col, day_col = day_col, beep_col = beep_col)
global_progress$close()
message("")
if (outer_completed != outer_fold_count) stop("Outer fold count differs from plan.")
message("Outer validation completed. Assessment observations: ", nrow(validation_results))

# ============================================================
# 18. RECOVER OUTER PREDICTIONS AND ACTUAL OUTCOMES
# ============================================================
row_id <- validation_results$.validation_row_id
prediction_results <- data.frame(
  .validation_row_id = row_id,
  id = prepared_data[[id_col]][row_id],
  day = prepared_data[[day_col]][row_id],
  beep = prepared_data[[beep_col]][row_id],
  actual_nonresponse = prepared_data$nonresponse[row_id],
  predicted_probability = validation_results$predicted_prob,
  predicted_class = validation_results$predicted_class,
  calibrated_threshold = validation_results$calibrated_threshold,
  selected_alpha = validation_results$selected_alpha,
  selected_lambda = validation_results$selected_lambda,
  selected_lambda_choice = validation_results$selected_lambda_choice,
  fold = validation_results$fold,
  training_max_day = validation_results$training_max_day,
  test_day = validation_results$test_day,
  test_beep = validation_results$test_beep,
  stringsAsFactors = FALSE)
if (anyDuplicated(row_id) || anyNA(prediction_results$predicted_probability) ||
    anyNA(prediction_results$actual_nonresponse) ||
    anyNA(prediction_results$predicted_class)) stop("Invalid prediction results.")
if (!length(tuning_log) || !length(inner_fold_log) || !length(calibration_log)) {
  stop("One or more model logs are empty.")
}
tuning_results <- do.call(rbind, tuning_log)
inner_fold_results <- do.call(rbind, inner_fold_log)
calibration_results <- do.call(rbind, calibration_log)
calibration_curves <- do.call(rbind, calibration_curve_log)
rownames(tuning_results) <- NULL
rownames(inner_fold_results) <- NULL
rownames(calibration_results) <- NULL
rownames(calibration_curves) <- NULL

# ============================================================
# 19. CALCULATE POOLED OUTER-VALIDATION METRICS
# ============================================================
message("Calculating outer-validation metrics.")
metric_results <- calculate_threshold_metrics(
  y_true = prediction_results$actual_nonresponse,
  y_prob = prediction_results$predicted_probability,
  predicted_class = prediction_results$predicted_class,
  beta = beta, thresholds = threshold_grid)
calibrated_metrics <- attr(metric_results, "calibrated_metrics")
if (is.null(calibrated_metrics)) {
  stop("metrics.R must support predicted_class and calibrated_metrics attribute.")
}
message("Calibrated outer-validation metrics:")
print(calibrated_metrics)

# ============================================================
# 20. EXPLORATORY OPTIMAL THRESHOLD (NOT USED FOR EVALUATION)
# ============================================================
message("Finding EXPLORATORY pooled threshold using ", threshold_metric, ".")
threshold_results <- find_optimal_threshold(
  y_true = prediction_results$actual_nonresponse,
  y_prob = prediction_results$predicted_probability,
  metric = threshold_metric, beta = beta, thresholds = threshold_grid)
optimal_threshold <- threshold_results$optimal_threshold
message("Exploratory optimal threshold: ", round(optimal_threshold, 4))
message("Exploratory peak ", threshold_metric, ": ",
        round(threshold_results$peak_score, 4))
# Do not overwrite predicted_class with this exploratory threshold.

# ============================================================
# 21. SUMMARIZE SELECTED HYPERPARAMETERS
# ============================================================
selected_tuning_results <- tuning_results[tuning_results$selected, , drop = FALSE]
if (!nrow(selected_tuning_results)) stop("No selected hyperparameters.")
selected_alpha_counts <- as.data.frame(table(selected_tuning_results$alpha),
                                       stringsAsFactors = FALSE)
names(selected_alpha_counts) <- c("alpha", "number_of_outer_folds")
selected_lambda_counts <- as.data.frame(table(selected_tuning_results$lambda_choice),
                                        stringsAsFactors = FALSE)
names(selected_lambda_counts) <- c("lambda_choice", "number_of_outer_folds")

# ============================================================
# 22. COUNT MODEL FITS
# ============================================================
number_of_outer_folds <- outer_fold_count
number_of_inner_model_fits <- planned_inner_folds
number_of_alpha_inner_fits <- nrow(inner_fold_results)
number_of_calibration_fits <- sum(calibration_results$calibration_folds)
number_of_final_outer_fits <- number_of_outer_folds
total_glmnet_fits <- number_of_alpha_inner_fits +
  number_of_calibration_fits + number_of_final_outer_fits
if (number_of_alpha_inner_fits != planned_inner_fits ||
    number_of_calibration_fits != planned_calibration_fits ||
    total_glmnet_fits != planned_total_fits) stop("Actual fits differ from plan.")
message("\nModel fitting workload:")
message("Outer assessment folds: ", number_of_outer_folds)
message("Unique inner temporal folds: ", number_of_inner_model_fits)
message("Inner glmnet fits: ", number_of_alpha_inner_fits)
message("Calibration glmnet fits: ", number_of_calibration_fits)
message("Final outer refits: ", number_of_final_outer_fits)
message("Total glmnet fits: ", total_glmnet_fits)

# ============================================================
# 23. SAVE RESULTS
# ============================================================
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
save_csv <- function(object, filename) {
  utils::write.csv(object, file.path(output_dir, filename), row.names = FALSE)
}
save_csv(prediction_results, "predictions.csv")
save_csv(metric_results, "metrics.csv")
save_csv(calibrated_metrics, "calibrated_metrics.csv")
save_csv(threshold_results$performance_curve, "threshold_performance.csv")
save_csv(tuning_results, "tuning_results.csv")
save_csv(selected_tuning_results, "selected_hyperparameters_by_fold.csv")
save_csv(inner_fold_results, "inner_temporal_folds.csv")
save_csv(calibration_results, "calibration_by_outer_fold.csv")
save_csv(calibration_curves, "calibration_threshold_curves.csv")
save_csv(selected_alpha_counts, "selected_alpha_counts.csv")
save_csv(selected_lambda_counts, "selected_lambda_counts.csv")
save_csv(data.frame(feature = features), "candidate_features.csv")
save_csv(outer_fold_plan, "outer_workload_plan.csv")

# ============================================================
# 24. SAVE MODEL CONFIGURATION
# ============================================================
model_configuration <- data.frame(
  dataset_id = dataset_id,
  model = "Nested Temporal Regularized Logistic Regression",
  validation_mode = validation_mode,
  outer_initial_days = initial_days,
  inner_validation_mode = inner_validation_mode,
  inner_initial_days = inner_initial_days,
  alpha_grid = paste(alpha_grid, collapse = ";"),
  lambda_choices = paste(lambda_choices, collapse = ";"),
  lambda_grid_size = length(lambda_grid),
  lambda_grid_min = min(lambda_grid),
  lambda_grid_max = max(lambda_grid),
  minimum_inner_folds = minimum_inner_folds,
  number_of_candidate_features = length(features),
  number_of_outer_predictions = nrow(prediction_results),
  number_of_outer_folds = number_of_outer_folds,
  number_of_inner_temporal_folds = number_of_inner_model_fits,
  number_of_inner_glmnet_fits = number_of_alpha_inner_fits,
  number_of_calibration_glmnet_fits = number_of_calibration_fits,
  number_of_final_outer_fits = number_of_final_outer_fits,
  total_glmnet_fits = total_glmnet_fits,
  threshold_metric = threshold_metric,
  threshold_calibration = "historical inner-fold out-of-sample predictions",
  exploratory_optimal_threshold = optimal_threshold,
  exploratory_peak_score = threshold_results$peak_score,
  random_seed = random_seed, preflight_only = preflight_only,
  stringsAsFactors = FALSE)
save_csv(model_configuration, "model_configuration.csv")

# ============================================================
# 25. FINAL SUMMARY
# ============================================================
message("\n================================================")
message("NESTED TEMPORAL REGULARIZED LOGISTIC COMPLETE")
message("================================================")
message("Dataset: ", dataset_id)
message("Outer validation: ", validation_mode)
message("Outer initial days: ", initial_days)
message("Inner validation: ", inner_validation_mode)
message("Inner initial days: ", inner_initial_days)
message("Alpha candidates: ", paste(alpha_grid, collapse = ", "))
message("Lambda selection rules: ", paste(lambda_choices, collapse = ", "))
message("Lambda grid size: ", length(lambda_grid))
message("Predictor count: ", length(features))
message("Outer assessment predictions: ", nrow(prediction_results))
message("Outer validation folds: ", number_of_outer_folds)
message("Inner temporal folds: ", number_of_inner_model_fits)
message("Inner glmnet fits: ", number_of_alpha_inner_fits)
message("Calibration glmnet fits: ", number_of_calibration_fits)
message("Final outer refits: ", number_of_final_outer_fits)
message("Total glmnet fits: ", total_glmnet_fits)
message("Selected alpha counts: ", paste(paste0(
  selected_alpha_counts$alpha, "=", selected_alpha_counts$number_of_outer_folds),
  collapse = ", "))
message("Selected lambda counts: ", paste(paste0(
  selected_lambda_counts$lambda_choice, "=",
  selected_lambda_counts$number_of_outer_folds), collapse = ", "))
message("Exploratory threshold: ", round(optimal_threshold, 4))
message("Exploratory optimized ", threshold_metric, ": ",
        round(threshold_results$peak_score, 4))
message("Output directory: ", output_dir)
message("================================================")
