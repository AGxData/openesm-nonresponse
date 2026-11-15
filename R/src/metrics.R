
#' Calculate Binary Classification Metrics Across Thresholds
#'
#' Uses yardstick for classification and probability metrics.
#'
#' Target encoding:
#'   0 = response
#'   1 = nonresponse (positive class)
#'
#' If predicted_class is supplied, the function also calculates
#' metrics for those existing classifications without optimizing
#' or modifying their thresholds.
#'
#' @param y_true Actual binary outcomes.
#' @param y_prob Predicted probabilities of nonresponse.
#' @param beta Positive F-beta weighting parameter.
#' @param thresholds Threshold grid for exploratory evaluation.
#' @param predicted_class Optional precomputed binary predictions.
#'
#' @return Data.frame containing threshold-dependent metrics,
#'   probability metrics, and optional calibrated metrics.
calculate_threshold_metrics <- function(
    y_true,
    y_prob,
    beta = 1,
    thresholds = seq(0.01, 0.99, by = 0.01),
    predicted_class = NULL) {

  # ==========================================================
  # 1. CHECK PACKAGE DEPENDENCIES
  # ==========================================================

  if (!requireNamespace("yardstick", quietly = TRUE)) {
    stop("Package 'yardstick' is required.")
  }

  # ==========================================================
  # 2. VALIDATE INPUTS
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

  if (
    !is.numeric(beta) ||
    length(beta) != 1L ||
    !is.finite(beta) ||
    beta <= 0
  ) {
    stop("beta must be a positive finite number.")
  }

  if (
    !is.numeric(thresholds) ||
    length(thresholds) == 0L ||
    anyNA(thresholds) ||
    any(!is.finite(thresholds)) ||
    any(thresholds < 0 | thresholds > 1)
  ) {
    stop(
      "thresholds must contain finite values between 0 and 1."
    )
  }

  if (!is.null(predicted_class)) {

    if (length(predicted_class) != length(y_true)) {
      stop(
        "predicted_class must have the same length as y_true."
      )
    }

    predicted_class <- as.character(predicted_class)

    if (
      anyNA(predicted_class) ||
      !all(predicted_class %in% c("0", "1"))
    ) {
      stop("predicted_class must contain only 0 and 1.")
    }
  }

  # ==========================================================
  # 3. CREATE YARDSTICK FACTORS
  # ==========================================================

  truth <- factor(
    y_true,
    levels = c("1", "0")
  )

  probability_data <- data.frame(
    truth = truth,
    .pred_nonresponse = as.numeric(y_prob)
  )

  # ==========================================================
  # 4. CALCULATE THRESHOLD-INDEPENDENT METRICS
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

  roc_auc <- safe_metric(
    yardstick::roc_auc_vec(
      truth = truth,
      estimate = y_prob,
      event_level = "first"
    )
  )

  pr_auc <- safe_metric(
    yardstick::pr_auc_vec(
      truth = truth,
      estimate = y_prob,
      event_level = "first"
    )
  )

  log_loss <- safe_metric(
    yardstick::mn_log_loss_vec(
      truth = truth,
      estimate = cbind(
        y_prob,
        1 - y_prob
      )
    )
  )

  brier_score <- mean(
    (as.integer(y_true) - y_prob)^2
  )

  # ==========================================================
  # 5. DEFINE CLASSIFICATION METRIC CALCULATOR
  # ==========================================================

  calculate_class_metrics <- function(predicted) {

    estimate <- factor(
      as.character(predicted),
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
      FN = fn
    )
  }

  # ==========================================================
  # 6. CALCULATE EXPLORATORY THRESHOLD CURVE
  # ==========================================================

  metric_rows <- lapply(
    thresholds,
    function(threshold) {

      predicted <- as.integer(
        y_prob >= threshold
      )

      classification_metrics <- calculate_class_metrics(
        predicted
      )

      data.frame(
        threshold = threshold,
        classification_metrics,
        ROC_AUC = roc_auc,
        PR_AUC = pr_auc,
        Log_Loss = log_loss,
        Brier_Score = brier_score,
        stringsAsFactors = FALSE
      )
    }
  )

  metric_results <- do.call(
    rbind,
    metric_rows
  )

  rownames(metric_results) <- NULL

  # ==========================================================
  # 7. EVALUATE PRECOMPUTED CALIBRATED CLASSIFICATIONS
  # ==========================================================

  if (!is.null(predicted_class)) {

    calibrated_metrics <- calculate_class_metrics(
      predicted_class
    )

    calibrated_metrics$ROC_AUC <- roc_auc
    calibrated_metrics$PR_AUC <- pr_auc
    calibrated_metrics$Log_Loss <- log_loss
    calibrated_metrics$Brier_Score <- brier_score

    attr(
      metric_results,
      "calibrated_metrics"
    ) <- calibrated_metrics
  }

  # ==========================================================
  # 8. RETURN METRICS
  # ==========================================================

  metric_results
}
