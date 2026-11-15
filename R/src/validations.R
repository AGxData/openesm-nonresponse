
#' Execute Longitudinal Walk-Forward Validation
#'
#' Performs expanding-window validation using either:
#'   - Across-day validation
#'   - Within-day validation
#'
#' Across-day:
#'   Train on all previous days and predict the next day.
#'
#' Within-day:
#'   Train on all previous days and earlier beeps from
#'   the current day. Predict all participants at the
#'   current beep simultaneously.
#'
#' Model-specific hyperparameter tuning, threshold
#' calibration, and fitting belong inside model_func.
#'
#' The model function may return:
#'   1. A numeric probability vector.
#'   2. A data.frame containing .validation_row_id,
#'      predicted_prob, and optional prediction metadata.
#'
#' @param df Prepared longitudinal data.frame.
#' @param features Character vector of predictor columns.
#' @param target Binary target: 0 = response, 1 = nonresponse.
#' @param model_func Model fitting and prediction function.
#' @param mode "within_day" or "across_day".
#' @param start_pct Initial training proportion.
#' @param id_col Participant identifier column.
#' @param day_col Study-day column.
#' @param beep_col Prompt/beep column.
#' @param initial_days Optional initial training days.
#' @param fold_loader Optional function(fold_label, train_data, test_data,
#'   features, target) returning NULL (cache miss) or model predictions
#'   in the original model_func return format. The caller must verify
#'   checkpoint provenance and integrity before returning predictions.
#' @param fold_saver Optional function(fold_label, train_data, test_data,
#'   features, target, fold_results) to persist a validated new fold.
#'   Must return normally only when persistence is complete.
#'
#' @return Data.frame of out-of-sample predictions and metadata.
#'
#' @export
run_walk_forward_validation <- function(
    df,
    features,
    target,
    model_func,
    mode = "within_day",
    start_pct = 0.25,
    id_col = "id",
    day_col = "day",
    beep_col = "beep",
    initial_days = NULL,
    fold_loader = NULL,
    fold_saver = NULL) {

  # ==========================================================
  # 1. VALIDATE INPUT DATA
  # ==========================================================

  if (!is.data.frame(df) || nrow(df) == 0L) {
    stop("df must be a nonempty data.frame.")
  }

  if (anyDuplicated(names(df))) {
    stop("df cannot contain duplicate column names.")
  }

  if (
    !is.character(features) ||
    length(features) == 0L ||
    anyNA(features) ||
    any(!nzchar(trimws(features))) ||
    anyDuplicated(features)
  ) {
    stop(
      "features must contain unique, nonempty column names."
    )
  }

  if (
    !is.character(target) ||
    length(target) != 1L ||
    is.na(target) ||
    !nzchar(trimws(target))
  ) {
    stop("target must be one nonempty column name.")
  }

  if (!is.function(model_func)) {
    stop("model_func must be a function.")
  }

  if (
    !is.numeric(start_pct) ||
    length(start_pct) != 1L ||
    !is.finite(start_pct) ||
    start_pct <= 0 ||
    start_pct >= 1
  ) {
    stop("start_pct must be between 0 and 1.")
  }

  mode <- match.arg(
    mode,
    choices = c("within_day", "across_day")
  )

  # ADDITIVE CHECKPOINT HOOK VALIDATION (optional; no-op by default)
  if (!is.null(fold_loader) && !is.function(fold_loader)) {
    stop("fold_loader must be NULL or a function.")
  }
  if (!is.null(fold_saver) && !is.function(fold_saver)) {
    stop("fold_saver must be NULL or a function.")
  }

  # ==========================================================
  # 2. VALIDATE STRUCTURAL COLUMNS
  # ==========================================================

  structural_cols <- c(
    id_col,
    day_col,
    beep_col,
    target
  )

  if (
    length(structural_cols) != 4L ||
    !is.character(structural_cols) ||
    anyNA(structural_cols) ||
    any(!nzchar(trimws(structural_cols))) ||
    anyDuplicated(structural_cols)
  ) {
    stop(
      "Identifier, temporal, and target columns ",
      "must be distinct nonempty strings."
    )
  }

  required_cols <- unique(
    c(structural_cols, features)
  )

  missing_cols <- setdiff(
    required_cols,
    names(df)
  )

  if (length(missing_cols) > 0L) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  # ==========================================================
  # 3. PROTECT AGAINST TARGET LEAKAGE
  # ==========================================================

  if (target %in% features) {
    stop("The target cannot be included in features.")
  }

  if (
    identical(target, "nonresponse") &&
    "responded" %in% features
  ) {
    stop(
      "'responded' is the exact inverse of 'nonresponse' ",
      "and cannot be used as a predictor."
    )
  }

  # ==========================================================
  # 4. PROTECT RESERVED OUTPUT COLUMNS
  # ==========================================================

  reserved_cols <- c(
    ".validation_row_id",
    "predicted_prob",
    "fold",
    "training_max_day",
    "test_day",
    "test_beep"
  )

  conflicts <- intersect(
    reserved_cols,
    names(df)
  )

  if (length(conflicts) > 0L) {
    stop(
      "Input contains reserved validation columns: ",
      paste(conflicts, collapse = ", ")
    )
  }

  # ==========================================================
  # 5. VALIDATE IDENTIFIERS AND TIME
  # ==========================================================

  if (anyNA(df[[id_col]])) {
    stop("Participant identifiers cannot be missing.")
  }

  if (
    is.character(df[[id_col]]) &&
    any(!nzchar(trimws(df[[id_col]])))
  ) {
    stop("Participant identifiers cannot be empty.")
  }

  for (column in c(day_col, beep_col)) {

    values <- df[[column]]

    if (
      !is.numeric(values) ||
      anyNA(values) ||
      any(!is.finite(values)) ||
      any(values <= 0)
    ) {
      stop(
        column,
        " must contain positive finite numeric values."
      )
    }
  }

  # ==========================================================
  # 6. VALIDATE TARGET
  # ==========================================================

  target_values <- as.character(
    df[[target]]
  )

  if (
    anyNA(target_values) ||
    !all(target_values %in% c("0", "1"))
  ) {
    stop(
      "Target must contain nonmissing binary values: ",
      "0 = response, 1 = nonresponse."
    )
  }

  df[[target]] <- as.integer(
    target_values
  )

  # ==========================================================
  # 7. CHECK OBSERVATION UNIQUENESS
  # ==========================================================

  observation_keys <- df[
    ,
    c(id_col, day_col, beep_col),
    drop = FALSE
  ]

  if (anyDuplicated(observation_keys)) {
    stop(
      "Duplicate participant/day/beep combinations detected."
    )
  }

  # ==========================================================
  # 8. DETERMINE INITIAL TRAINING WINDOW
  # ==========================================================

  unique_days <- sort(
    unique(df[[day_col]])
  )

  number_of_days <- length(unique_days)

  if (number_of_days < 2L) {
    stop("At least two study days are required.")
  }

  if (is.null(initial_days)) {

    initial_days <- max(
      1L,
      floor(number_of_days * start_pct)
    )

  } else {

    if (
      !is.numeric(initial_days) ||
      length(initial_days) != 1L ||
      !is.finite(initial_days) ||
      initial_days != floor(initial_days) ||
      initial_days < 1 ||
      initial_days >= number_of_days
    ) {
      stop(
        "initial_days must be an integer between 1 ",
        "and the number of study days minus 1."
      )
    }

    initial_days <- as.integer(initial_days)
  }

  # ==========================================================
  # 9. PRESERVE ORIGINAL ROW IDENTITIES
  # ==========================================================

  df[[".validation_row_id"]] <- seq_len(
    nrow(df)
  )

  # ==========================================================
  # 10. SORT LONGITUDINAL DATA
  # ==========================================================

  sort_longitudinal_data <- function(data) {

    ordering <- order(
      data[[day_col]],
      data[[beep_col]],
      data[[id_col]]
    )

    data <- data[
      ordering,
      ,
      drop = FALSE
    ]

    rownames(data) <- NULL

    data
  }

  # ==========================================================
  # 11. EVALUATE ONE OUTER SPLIT
  # ==========================================================

  evaluate_split <- function(
      train_data,
      test_data,
      fold_label) {

    if (
      nrow(train_data) == 0L ||
      nrow(test_data) == 0L
    ) {
      stop(
        "Empty training or assessment data in fold: ",
        fold_label
      )
    }

    train_data <- sort_longitudinal_data(train_data)
    test_data <- sort_longitudinal_data(test_data)

    training_max_day <- max(
      train_data[[day_col]]
    )

    test_days <- unique(
      test_data[[day_col]]
    )

    if (length(test_days) != 1L) {
      stop(
        "Each outer fold must assess exactly one day."
      )
    }

    test_day <- test_days[1L]

    # --------------------------------------------------------
    # Temporal separation
    # --------------------------------------------------------

    if (training_max_day > test_day) {
      stop(
        "Temporal leakage: training contains ",
        "observations after the assessment day."
      )
    }

    if (mode == "across_day") {

      if (
        any(train_data[[day_col]] >= test_day)
      ) {
        stop(
          "Across-day training must contain only ",
          "earlier days."
        )
      }

    } else {

      test_beeps <- unique(
        test_data[[beep_col]]
      )

      if (length(test_beeps) != 1L) {
        stop(
          "Each within-day assessment fold must ",
          "contain exactly one beep."
        )
      }

      current_beep <- test_beeps[1L]

      if (
        any(
          train_data[[day_col]] == test_day &
          train_data[[beep_col]] >= current_beep
        )
      ) {
        stop(
          "Temporal leakage: same-day training ",
          "contains the current or a future beep."
        )
      }
    }

    # --------------------------------------------------------
    # Execute model function
    # --------------------------------------------------------

    # ADDITIVE: load only a caller-verified checkpoint when available.
    # A NULL return is a cache miss. Loaded predictions are subjected
    # to all the same alignment, schema, and probability checks below.
    recovered_from_checkpoint <- FALSE
    predictions <- NULL

    if (!is.null(fold_loader)) {
      predictions <- fold_loader(
        fold_label = fold_label,
        train_data = train_data,
        test_data = test_data,
        features = features,
        target = target
      )
      recovered_from_checkpoint <- !is.null(predictions)
    }

    if (!recovered_from_checkpoint) {
      predictions <- model_func(
        train_data = train_data,
        test_data = test_data,
        features = features,
        target = target
      )
    }

    # ========================================================
    # 12. VALIDATE AND ALIGN PREDICTIONS
    # ========================================================

    prediction_metadata <- NULL

    if (is.data.frame(predictions)) {

      if (anyDuplicated(names(predictions))) {
        stop(
          "Prediction output contains duplicate columns."
        )
      }

      required_prediction_cols <- c(
        ".validation_row_id",
        "predicted_prob"
      )

      missing_prediction_cols <- setdiff(
        required_prediction_cols,
        names(predictions)
      )

      if (length(missing_prediction_cols) > 0L) {
        stop(
          "Prediction data.frame must contain ",
          ".validation_row_id and predicted_prob."
        )
      }

      prohibited_cols <- c(
        "fold",
        "training_max_day",
        "test_day",
        "test_beep"
      )

      conflicting_cols <- intersect(
        prohibited_cols,
        names(predictions)
      )

      if (length(conflicting_cols) > 0L) {
        stop(
          "Model output conflicts with reserved columns: ",
          paste(conflicting_cols, collapse = ", ")
        )
      }

      expected_ids <- test_data[[
        ".validation_row_id"
      ]]

      prediction_ids <- predictions[[
        ".validation_row_id"
      ]]

      if (
        length(prediction_ids) != nrow(test_data) ||
        anyNA(prediction_ids) ||
        anyDuplicated(prediction_ids) ||
        !setequal(prediction_ids, expected_ids)
      ) {
        stop(
          "Prediction row IDs must match assessment rows ",
          "exactly once."
        )
      }

      prediction_order <- match(
        expected_ids,
        prediction_ids
      )

      predictions <- predictions[
        prediction_order,
        ,
        drop = FALSE
      ]

      predicted_prob <- predictions[[
        "predicted_prob"
      ]]

      metadata_cols <- setdiff(
        names(predictions),
        required_prediction_cols
      )

      if (length(metadata_cols) > 0L) {
        prediction_metadata <- predictions[
          ,
          metadata_cols,
          drop = FALSE
        ]
      }

    } else {

      if (
        !is.numeric(predictions) ||
        !is.null(dim(predictions)) ||
        length(predictions) != nrow(test_data)
      ) {
        stop(
          "model_func must return a numeric probability ",
          "vector or an aligned prediction data.frame."
        )
      }

      predicted_prob <- predictions
    }

    # ========================================================
    # 13. VALIDATE PROBABILITIES
    # ========================================================

    if (
      !is.numeric(predicted_prob) ||
      !is.null(dim(predicted_prob)) ||
      length(predicted_prob) != nrow(test_data) ||
      anyNA(predicted_prob) ||
      any(!is.finite(predicted_prob)) ||
      any(predicted_prob < 0 | predicted_prob > 1)
    ) {
      stop(
        "Predicted probabilities must be finite ",
        "numeric values between 0 and 1."
      )
    }

    # ========================================================
    # 14. BUILD FOLD RESULTS
    # ========================================================

    fold_results <- data.frame(
      .validation_row_id =
        test_data[[".validation_row_id"]],
      predicted_prob =
        as.numeric(predicted_prob),
      fold =
        rep(fold_label, nrow(test_data)),
      training_max_day =
        rep(training_max_day, nrow(test_data)),
      test_day =
        test_data[[day_col]],
      test_beep =
        test_data[[beep_col]],
      stringsAsFactors = FALSE
    )

    if (!is.null(prediction_metadata)) {

      conflicts <- intersect(
        names(prediction_metadata),
        names(fold_results)
      )

      if (length(conflicts) > 0L) {
        stop(
          "Prediction metadata conflicts with ",
          "validation columns: ",
          paste(conflicts, collapse = ", ")
        )
      }

      fold_results <- cbind(
        fold_results,
        prediction_metadata
      )
    }

    rownames(fold_results) <- NULL

    # ADDITIVE: persist only newly computed, fully validated folds.
    # Save failures propagate and halt the experiment; an incomplete
    # checkpoint must never be treated as a completed fold.
    if (!recovered_from_checkpoint && !is.null(fold_saver)) {
      fold_saver(
        fold_label = fold_label,
        train_data = train_data,
        test_data = test_data,
        features = features,
        target = target,
        fold_results = fold_results
      )
    }

    fold_results
  }

  # ==========================================================
  # 15. EXECUTE OUTER EXPANDING-WINDOW VALIDATION
  # ==========================================================

  results_list <- list()
  result_index <- 1L

  validation_days <- unique_days[
    seq.int(
      initial_days + 1L,
      number_of_days
    )
  ]

  for (current_day in validation_days) {

    current_day_data <- df[
      df[[day_col]] == current_day,
      ,
      drop = FALSE
    ]

    if (mode == "across_day") {

      train_data <- df[
        df[[day_col]] < current_day,
        ,
        drop = FALSE
      ]

      test_data <- current_day_data

      fold_label <- paste0(
        "across_day_",
        current_day
      )

      results_list[[result_index]] <- evaluate_split(
        train_data,
        test_data,
        fold_label
      )

      result_index <- result_index + 1L

    } else {

      current_beeps <- sort(
        unique(current_day_data[[beep_col]])
      )

      for (current_beep in current_beeps) {

        train_data <- df[
          df[[day_col]] < current_day |
          (
            df[[day_col]] == current_day &
            df[[beep_col]] < current_beep
          ),
          ,
          drop = FALSE
        ]

        test_data <- current_day_data[
          current_day_data[[beep_col]] == current_beep,
          ,
          drop = FALSE
        ]

        fold_label <- paste0(
          "within_day_",
          current_day,
          "_beep_",
          current_beep
        )

        results_list[[result_index]] <- evaluate_split(
          train_data,
          test_data,
          fold_label
        )

        result_index <- result_index + 1L
      }
    }
  }

  # ==========================================================
  # 16. COMBINE VALIDATION RESULTS
  # ==========================================================

  if (length(results_list) == 0L) {
    stop("No validation folds were generated.")
  }

  reference_columns <- names(
    results_list[[1L]]
  )

  for (i in seq_along(results_list)) {

    if (
      !identical(
        names(results_list[[i]]),
        reference_columns
      )
    ) {
      stop(
        "Prediction metadata must have the same ",
        "columns in every outer fold."
      )
    }
  }

  validation_results <- do.call(
    rbind,
    results_list
  )

  rownames(validation_results) <- NULL

  # ==========================================================
  # 17. VERIFY COMPLETE ASSESSMENT COVERAGE
  # ==========================================================

  assessed_ids <- validation_results[[
    ".validation_row_id"
  ]]

  expected_ids <- df[[
    ".validation_row_id"
  ]][
    df[[day_col]] > unique_days[initial_days]
  ]

  if (
    anyNA(assessed_ids) ||
    anyDuplicated(assessed_ids) ||
    length(assessed_ids) != length(expected_ids) ||
    !setequal(assessed_ids, expected_ids)
  ) {
    stop(
      "Outer validation did not assess every eligible ",
      "observation exactly once."
    )
  }

  # ==========================================================
  # 18. RESTORE ORIGINAL ROW ORDER
  # ==========================================================

  validation_results <- validation_results[
    order(
      validation_results[[
        ".validation_row_id"
      ]]
    ),
    ,
    drop = FALSE
  ]

  rownames(validation_results) <- NULL

  # ==========================================================
  # 19. FINAL VALIDATION
  # ==========================================================

  if (
    nrow(validation_results) != length(expected_ids) ||
    anyNA(validation_results$predicted_prob) ||
    any(!is.finite(validation_results$predicted_prob))
  ) {
    stop("Final validation output is invalid.")
  }

  # ==========================================================
  # 20. RETURN OUT-OF-SAMPLE PREDICTIONS
  # ==========================================================

  validation_results
}
