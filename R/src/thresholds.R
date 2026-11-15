
#' Find the Optimal Binary Classification Threshold
#'
#' Evaluates classification thresholds using yardstick.
#'
#' Target encoding:
#'   0 = response
#'   1 = nonresponse (positive class)
#'
#' IMPORTANT:
#'   This function does not enforce temporal separation.
#'   The caller must provide calibration observations
#'   that precede the corresponding outer assessment.
#'
#' @param y_true Actual binary outcomes.
#' @param y_prob Predicted probabilities of nonresponse.
#' @param metric Metric to optimize.
#' @param beta Positive F-beta weighting parameter.
#' @param step_size Threshold increment when thresholds is NULL.
#' @param thresholds Optional explicit threshold vector.
#'
#' @return List containing optimal_threshold, peak_score,
#'   metric, and performance_curve.
find_optimal_threshold <- function(
    y_true,
    y_prob,
    metric = "MCC",
    beta = 1,
    step_size = 0.01,
    thresholds = NULL) {

  # ==========================================================
  # 1. CHECK DEPENDENCIES
  # ==========================================================

  if (!requireNamespace("yardstick", quietly = TRUE)) {
    stop("Package 'yardstick' is required.")
  }

  # ==========================================================
  # 2. VALIDATE TARGET AND PROBABILITIES
  # ==========================================================

  if (
    length(y_true) == 0L ||
    length(y_true) != length(y_prob)
  ) {
    stop(
      "y_true and y_prob must have equal nonzero lengths."
    )
  }

  y_true <- as.character(y_true)

  if (
    anyNA(y_true) ||
    !all(y_true %in% c("0", "1"))
  ) {
    stop("y_true must contain only 0 and 1.")
  }

  if (
    !is.numeric(y_prob) ||
    anyNA(y_prob) ||
    any(!is.finite(y_prob)) ||
    any(y_prob < 0 | y_prob > 1)
  ) {
    stop(
      "y_prob must contain finite probabilities ",
      "between 0 and 1."
    )
  }

  # ==========================================================
  # 3. VALIDATE METRIC
  # ==========================================================

  supported_metrics <- c(
    "MCC",
    "Precision",
    "Recall",
    "Accuracy",
    "Balanced_Accuracy",
    "F1",
    "F_Beta",
    "Jaccard",
    "Youden_J",
    "Kappa"
  )

  if (
    !is.character(metric) ||
    length(metric) != 1L ||
    is.na(metric) ||
    !metric %in% supported_metrics
  ) {
    stop(
      "Unsupported threshold metric: ",
      paste(metric, collapse = ", ")
    )
  }

  if (
    !is.numeric(beta) ||
    length(beta) != 1L ||
    !is.finite(beta) ||
    beta <= 0
  ) {
    stop("beta must be a positive finite number.")
  }

  # ==========================================================
  # 4. PREPARE THRESHOLD GRID
  # ==========================================================

  if (is.null(thresholds)) {

    if (
      !is.numeric(step_size) ||
      length(step_size) != 1L ||
      !is.finite(step_size) ||
      step_size <= 0 ||
      step_size > 1
    ) {
      stop(
        "step_size must be a positive number ",
        "between 0 and 1."
      )
    }

    thresholds <- seq(
      from = step_size,
      to = 1 - step_size,
      by = step_size
    )
  }

  if (
    !is.numeric(thresholds) ||
    length(thresholds) == 0L ||
    anyNA(thresholds) ||
    any(!is.finite(thresholds)) ||
    any(thresholds < 0 | thresholds > 1)
  ) {
    stop(
      "thresholds must contain finite probabilities ",
      "between 0 and 1."
    )
  }

  thresholds <- sort(
    unique(thresholds)
  )

  # ==========================================================
  # 5. PREPARE YARDSTICK TARGET
  # ==========================================================

  truth <- factor(
    y_true,
    levels = c("1", "0")
  )

  # ==========================================================
  # 6. DEFINE SAFE METRIC EVALUATION
  # ==========================================================

  safe_metric <- function(expr) {

    result <- tryCatch(
      suppressWarnings(expr),
      error = function(e) NA_real_
    )

    if (
      !is.numeric(result) ||
      length(result) != 1L ||
      !is.finite(result)
    ) {
      return(NA_real_)
    }

    as.numeric(result)
  }

  # ==========================================================
  # 7. CALCULATE THRESHOLD PERFORMANCE
  # ==========================================================

  evaluate_threshold <- function(threshold) {

    estimate <- factor(
      as.character(
        as.integer(y_prob >= threshold)
      ),
      levels = c("1", "0")
    )

    tp <- sum(truth == "1" & estimate == "1")
    tn <- sum(truth == "0" & estimate == "0")
    fp <- sum(truth == "0" & estimate == "1")
    fn <- sum(truth == "1" & estimate == "0")

    precision <- safe_metric(
      yardstick::precision_vec(
        truth,
        estimate,
        event_level = "first"
      )
    )

    recall <- safe_metric(
      yardstick::recall_vec(
        truth,
        estimate,
        event_level = "first"
      )
    )

    specificity <- safe_metric(
      yardstick::spec_vec(
        truth,
        estimate,
        event_level = "first"
      )
    )

    accuracy <- safe_metric(
      yardstick::accuracy_vec(
        truth,
        estimate
      )
    )

    balanced_accuracy <- safe_metric(
      yardstick::bal_accuracy_vec(
        truth,
        estimate,
        event_level = "first"
      )
    )

    f1 <- safe_metric(
      yardstick::f_meas_vec(
        truth,
        estimate,
        beta = 1,
        event_level = "first"
      )
    )

    f_beta <- safe_metric(
      yardstick::f_meas_vec(
        truth,
        estimate,
        beta = beta,
        event_level = "first"
      )
    )

    mcc <- safe_metric(
      yardstick::mcc_vec(
        truth,
        estimate
      )
    )

    kappa <- safe_metric(
      yardstick::kap_vec(
        truth,
        estimate
      )
    )

    jaccard <- if ((tp + fp + fn) > 0) {
      tp / (tp + fp + fn)
    } else {
      NA_real_
    }

    youden_j <- if (
      is.finite(recall) &&
      is.finite(specificity)
    ) {
      recall + specificity - 1
    } else {
      NA_real_
    }

    data.frame(
      threshold = threshold,
      MCC = mcc,
      Precision = precision,
      Recall = recall,
      Specificity = specificity,
      Accuracy = accuracy,
      Balanced_Accuracy = balanced_accuracy,
      F1 = f1,
      F_Beta = f_beta,
      Jaccard = jaccard,
      Youden_J = youden_j,
      Kappa = kappa,
      TP = tp,
      TN = tn,
      FP = fp,
      FN = fn,
      stringsAsFactors = FALSE
    )
  }

  performance_rows <- lapply(
    thresholds,
    evaluate_threshold
  )

  performance_curve <- do.call(
    rbind,
    performance_rows
  )

  rownames(performance_curve) <- NULL

  # ==========================================================
  # 8. IDENTIFY OPTIMAL THRESHOLD
  # ==========================================================

  metric_scores <- performance_curve[[metric]]

  valid_indices <- which(
    is.finite(metric_scores)
  )

  if (length(valid_indices) == 0L) {
    stop(
      "The selected metric is undefined at every threshold."
    )
  }

  peak_score <- max(
    metric_scores[valid_indices]
  )

  best_indices <- valid_indices[
    metric_scores[valid_indices] == peak_score
  ]

  # Deterministic tie-break:
  # choose the lowest threshold.

  best_index <- best_indices[
    which.min(
      performance_curve$threshold[best_indices]
    )
  ]

  optimal_threshold <- performance_curve$threshold[
    best_index
  ]

  # ==========================================================
  # 9. RETURN CALIBRATION RESULTS
  # ==========================================================

  list(
    optimal_threshold = optimal_threshold,
    peak_score = peak_score,
    metric = metric,
    performance_curve = performance_curve
  )
}
