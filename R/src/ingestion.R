#' Load a curated OpenESM dataset directly from S3
#'
#' @param dataset_id Dataset identifier, e.g. "0006", "6", or 6.
#' @param config_path Optional path to config/storage.yaml.
#'
#' @return A data.frame containing the curated dataset.
load_curated_dataset <- function(dataset_id, config_path = NULL) {

  # 1. Validate and normalize dataset ID
  if (length(dataset_id) != 1 || is.na(dataset_id)) {
    stop("dataset_id must be a single, non-missing value.")
  }

  dataset_id <- trimws(as.character(dataset_id))

  if (!grepl("^[0-9]{1,4}$", dataset_id)) {
    stop("dataset_id must contain between one and four digits.")
  }

  dataset_id <- sprintf("%04d", as.integer(dataset_id))

  # 2. Locate configuration
  if (!is.null(config_path)) {

    if (!file.exists(config_path)) {
      stop("Configuration file not found: ", config_path)
    }

    config_path <- normalizePath(
      config_path,
      winslash = "/",
      mustWork = TRUE
    )

  } else {

    current_path <- normalizePath(
      getwd(),
      winslash = "/",
      mustWork = TRUE
    )

    config_path <- NULL

    repeat {

      candidate <- file.path(
        current_path,
        "config",
        "storage.yaml"
      )

      if (file.exists(candidate)) {
        config_path <- candidate
        break
      }

      parent_path <- dirname(current_path)

      if (identical(parent_path, current_path)) {
        break
      }

      current_path <- parent_path
    }

    if (is.null(config_path)) {
      stop("Could not locate config/storage.yaml.")
    }
  }

  # 3. Read configuration
  config <- yaml::read_yaml(config_path)

  bucket <- config$aws$bucket
  prefix <- config$s3$prefixes$curated

  if (is.null(bucket) || !nzchar(trimws(bucket))) {
    stop("storage.yaml must define aws$bucket.")
  }

  if (is.null(prefix) || length(prefix) != 1) {
    stop("storage.yaml must define s3$prefixes$curated.")
  }

  prefix <- gsub("^/+|/+$", "", trimws(prefix))

  if (nzchar(prefix)) {
    prefix <- paste0(prefix, "/")
  }

  # 4. Configure AWS credentials for R

  env_names <- c(
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN"
  )

  old_env <- Sys.getenv(
    env_names,
    unset = NA_character_
  )

  on.exit({

    for (i in seq_along(env_names)) {

      if (is.na(old_env[i])) {
        Sys.unsetenv(env_names[i])
      } else {
        do.call(
          Sys.setenv,
          stats::setNames(
            list(old_env[i]),
            env_names[i]
          )
        )
      }
    }

  }, add = TRUE)

  # Determine the configured AWS profile
  profile <- config$aws$profile

  if (is.null(profile) || !nzchar(trimws(profile))) {
    profile <- "default"
  }

  # Export credentials using the AWS CLI
  credential_output <- tryCatch(
    system2(
      "aws",
      args = c(
        "configure",
        "export-credentials",
        "--profile",
        profile,
        "--format",
        "process"
      ),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(e) {
      stop(
        "Unable to obtain AWS credentials through the AWS CLI: ",
        conditionMessage(e)
      )
    }
  )

  command_status <- attr(credential_output, "status")

  if (!is.null(command_status) && command_status != 0) {
    stop(
      "AWS CLI credential export failed: ",
      paste(credential_output, collapse = "\n")
    )
  }

  # Parse exported credentials
  credentials <- tryCatch(
    jsonlite::fromJSON(
      paste(credential_output, collapse = "\n")
    ),
    error = function(e) {
      stop(
        "Unable to parse AWS CLI credentials: ",
        conditionMessage(e)
      )
    }
  )

  # Validate required credentials
  required_credentials <- c(
    "AccessKeyId",
    "SecretAccessKey"
  )

  if (!all(required_credentials %in% names(credentials))) {
    stop("AWS CLI did not return the required credentials.")
  }

  if (
    is.null(credentials$AccessKeyId) ||
    is.null(credentials$SecretAccessKey)
  ) {
    stop("AWS CLI returned incomplete credentials.")
  }

  # Make credentials available to aws.s3
  Sys.setenv(
    AWS_ACCESS_KEY_ID = credentials$AccessKeyId,
    AWS_SECRET_ACCESS_KEY = credentials$SecretAccessKey
  )

  if (
    !is.null(credentials$SessionToken) &&
    nzchar(credentials$SessionToken)
  ) {
    Sys.setenv(
      AWS_SESSION_TOKEN = credentials$SessionToken
    )
  }

  if (!is.null(config$aws$region)) {
    Sys.setenv(
      AWS_DEFAULT_REGION = config$aws$region
    )
  }

  message(
    "AWS credentials successfully loaded for profile: ",
    profile
  )

  # 5. Locate matching S3 object
  bucket_objects <- tryCatch(
    aws.s3::get_bucket(
      bucket = bucket,
      prefix = prefix,
      max = Inf
    ),
    error = function(e) {
      stop("Unable to list S3 objects: ", conditionMessage(e))
    }
  )

  matching_keys <- character()

  for (i in seq_along(bucket_objects)) {

    object_key <- bucket_objects[[i]]$Key

    if (is.null(object_key)) {
      next
    }

    filename <- basename(object_key)

    if (
      startsWith(filename, paste0(dataset_id, "_")) &&
      endsWith(filename, "_curated.parquet")
    ) {
      matching_keys <- c(matching_keys, object_key)
    }
  }

  if (length(matching_keys) == 0) {
    stop(
      sprintf(
        "Dataset %s was not found in s3://%s/%s",
        dataset_id,
        bucket,
        prefix
      )
    )
  }

  if (length(matching_keys) > 1) {
    stop(
      "Multiple curated datasets found: ",
      paste(matching_keys, collapse = ", ")
    )
  }

  # 6. Download Parquet object
  parquet_bytes <- tryCatch(
    aws.s3::get_object(
      bucket = bucket,
      object = matching_keys[1]
    ),
    error = function(e) {
      stop("Unable to download Parquet object: ", conditionMessage(e))
    }
  )

  if (!is.raw(parquet_bytes) || length(parquet_bytes) == 0) {
    stop("S3 download returned an empty or invalid object.")
  }

  # 7. Read Parquet data
  temp_file <- tempfile(fileext = ".parquet")

  on.exit(unlink(temp_file), add = TRUE)

  writeBin(parquet_bytes, temp_file)

  dataset <- tryCatch(
    arrow::read_parquet(temp_file),
    error = function(e) {
      stop("Unable to read Parquet dataset: ", conditionMessage(e))
    }
  )

  as.data.frame(dataset)
}