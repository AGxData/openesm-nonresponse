
#' Validate and prepare an already-engineered longitudinal dataset
#'
#' Python is responsible for feature engineering. This function
#' validates the curated dataset without recomputing features.
#'
#' Target encoding:
#'   Python: responded = 1, nonresponse = 0
#'   R:      nonresponse = 1, response = 0
#'
#' @param df Curated longitudinal data.frame.
#' @param target_col Original response indicator.
#' @param id_col Participant identifier column.
#' @param day_col Study-day column.
#' @param beep_col Prompt/beep column.
#' @param leakage_cols Columns to exclude from modeling.
#' @param drop_missing_target Remove rows with missing outcomes.
#'
#' @return Chronologically ordered data.frame.
#'
#' @export
preprocess_longitudinal_data <- function(
    df,
    target_col = "responded",
    id_col = "id",
    day_col = "day",
    beep_col = "beep",
    leakage_cols = c(
      "responded_any",
      "responded_any_today",
      "daily_response_rate",
      "missed_day",
      "cumulative_responses",
      "cumulative_prompts",
      "cumulative_response_rate",
      "last_observed_day",
      "last_response_day",
      "first_response_day",
      "total_observed_prompts",
      "total_responses",
      "overall_response_rate",
      "never_responded",
      "days_since_first_observed",
      "first_observed_day",
      "prompts_in_day",
      "first_beep_of_day",
      "last_beep_of_day",
      "beep_position",
      "scheduled_beeps_in_day",
      "scheduled_beep_position",
      "scheduled_beep_position_rate"
    ),
    drop_missing_target = TRUE) {

  # ==========================================================
  # 1. VALIDATE DATASET
  # ==========================================================

  if (!is.data.frame(df)) {
    stop("df must be a data.frame.")
  }

  if (nrow(df) == 0L) {
    stop("df contains no observations.")
  }

  if (anyDuplicated(names(df))) {
    stop("df contains duplicate column names.")
  }

  # ==========================================================
  # 2. VALIDATE COLUMN NAMES
  # ==========================================================

  structural_cols <- c(
    target_col,
    id_col,
    day_col,
    beep_col
  )

  if (
    !is.character(structural_cols) ||
    length(structural_cols) != 4L ||
    anyNA(structural_cols) ||
    any(!nzchar(trimws(structural_cols))) ||
    anyDuplicated(structural_cols)
  ) {
    stop(
      "Target, participant, day, and beep columns ",
      "must be distinct, nonempty character values."
    )
  }

  missing_cols <- setdiff(
    structural_cols,
    names(df)
  )

  if (length(missing_cols) > 0L) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  if ("nonresponse" %in% names(df)) {
    stop(
      "Input data already contains 'nonresponse'. ",
      "Remove or rename it before preprocessing."
    )
  }

  # ==========================================================
  # 3. VALIDATE OPTIONS
  # ==========================================================

  if (
    !is.logical(drop_missing_target) ||
    length(drop_missing_target) != 1L ||
    is.na(drop_missing_target)
  ) {
    stop("drop_missing_target must be TRUE or FALSE.")
  }

  if (
    !is.character(leakage_cols) ||
    anyNA(leakage_cols) ||
    any(!nzchar(trimws(leakage_cols)))
  ) {
    stop(
      "leakage_cols must be a character vector ",
      "without missing or empty names."
    )
  }

  protected_cols <- c(
    structural_cols,
    "nonresponse"
  )

  if (any(leakage_cols %in% structural_cols)) {
    stop(
      "leakage_cols cannot include required ",
      "structural or target columns."
    )
  }

  # ==========================================================
  # 4. VALIDATE PARTICIPANT IDENTIFIERS
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

  # ==========================================================
  # 5. VALIDATE TEMPORAL COLUMNS
  # ==========================================================

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
        " must contain positive, finite, ",
        "nonmissing numeric values."
      )
    }
  }

  # ==========================================================
  # 6. VALIDATE AND CONVERT RESPONSE TARGET
  # ==========================================================

  response_values <- as.character(
    df[[target_col]]
  )

  observed_values <- response_values[
    !is.na(response_values)
  ]

  if (
    length(observed_values) == 0L ||
    !all(observed_values %in% c("0", "1"))
  ) {
    stop(
      target_col,
      " must contain binary values: ",
      "1 = response, 0 = nonresponse."
    )
  }

  if (drop_missing_target) {

    df <- df[
      !is.na(response_values),
      ,
      drop = FALSE
    ]

    response_values <- response_values[
      !is.na(response_values)
    ]
  }

  if (nrow(df) == 0L) {
    stop("No observations remain after target filtering.")
  }

  df[[target_col]] <- as.integer(
    response_values
  )

  # ==========================================================
  # 7. CREATE NONRESPONSE TARGET
  # ==========================================================

  df[["nonresponse"]] <- 1L -
    df[[target_col]]

  # ==========================================================
  # 8. REMOVE EXPLICIT LEAKAGE COLUMNS
  # ==========================================================

  columns_to_remove <- intersect(
    leakage_cols,
    names(df)
  )

  columns_to_remove <- setdiff(
    columns_to_remove,
    protected_cols
  )

  if (length(columns_to_remove) > 0L) {

    df <- df[
      ,
      !names(df) %in% columns_to_remove,
      drop = FALSE
    ]
  }

  # ==========================================================
  # 9. CHECK OBSERVATION UNIQUENESS
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
  # 10. SORT CHRONOLOGICALLY
  # ==========================================================

  ordering <- order(
    df[[day_col]],
    df[[beep_col]],
    df[[id_col]]
  )

  df <- df[
    ordering,
    ,
    drop = FALSE
  ]

  rownames(df) <- NULL

  # ==========================================================
  # 11. FINAL TARGET VALIDATION
  # ==========================================================

  if (drop_missing_target) {

    if (
      anyNA(df[["nonresponse"]]) ||
      !all(df[["nonresponse"]] %in% c(0L, 1L))
    ) {
      stop(
        "The generated nonresponse target ",
        "contains invalid values."
      )
    }
  }

  # ==========================================================
  # 12. RETURN PREPARED DATA
  # ==========================================================

  df
}
