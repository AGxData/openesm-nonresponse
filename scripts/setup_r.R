# ============================================================
# OpenESM: Reproducible R Environment Setup
# R 4.5.1
# ============================================================

options(
  repos = c(CRAN = "https://cloud.r-project.org"),
  timeout = 600
)

cat("\n========================================\n")
cat("OpenESM R environment setup\n")
cat("========================================\n\n")

cat("R version:", R.version.string, "\n")
cat("Project:", normalizePath(getwd()), "\n\n")

# ------------------------------------------------------------
# 1. Install and initialize renv
# ------------------------------------------------------------

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

if (!file.exists("renv/activate.R")) {
  renv::init(
    project = getwd(),
    bare = TRUE,
    restart = FALSE
  )
}

source("renv/activate.R")

cat(
  "renv version:",
  as.character(packageVersion("renv")),
  "\n"
)

# ------------------------------------------------------------
# 2. Explicit package manifest
# ------------------------------------------------------------

packages <- c(
  # Core infrastructure
  "renv",
  "tidyverse",
  "data.table",
  "arrow",
  "janitor",
  "lubridate",
  "here",
  "jsonlite",

  # Existing OpenESM project packages
  "openesm",
  "tvem",
  "PRROC",

  # Regularized logistic regression
  "glmnet",
  "LiblineaR",

  # Mixed-effects and penalized mixed-effects models
  "lme4",
  "glmmTMB",
  "nlme",
  "glmmLasso",
  "glmmPen",

  # Other statistical models
  "mgcv",
  "survival",

  # Hidden Markov models
  "hmmTMB",
  "depmixS4",

  # Bayesian modeling
  "brms",

  # Classification metrics
  "yardstick",
  "pROC",
  "precrec",
  "MLmetrics",

  # Calibration
  "probably",
  "caret",

  # Feature selection
  "Boruta",
  "vip",

  # Model interpretation
  "DALEX",
  "iml",
  "shapviz",

  # Diagnostics and model interpretation
  "DHARMa",
  "performance",
  "marginaleffects",
  "broom",
  "broom.mixed",
  "emmeans",
  "parameters",

  # Temporal validation
  "rsample",
  "slider",
  "timetk",

  # Modeling workflows
  "recipes",
  "workflows",
  "tune",
  "targets",
  "tarchetypes",

  # Parallel processing
  "future",
  "furrr",
  "doParallel",
  "doRNG",

  # AWS S3 access
  "aws.s3"
  "paws.storage",

  # Development and reproducibility
  "testthat",
  "knitr",
  "rmarkdown",
  "sessioninfo",
  "lintr",
  "styler",
  "jsonvalidate"
)

packages <- unique(packages)

# ------------------------------------------------------------
# 3. Install packages
# ------------------------------------------------------------

install_results <- data.frame(
  package = character(),
  status = character(),
  message = character(),
  stringsAsFactors = FALSE
)

install_one <- function(package_name) {
  cat("\n----------------------------------------\n")
  cat("Installing:", package_name, "\n")
  cat("----------------------------------------\n")

  result <- tryCatch(
    {
      renv::install(package_name)

      data.frame(
        package = package_name,
        status = "installed",
        message = "",
        stringsAsFactors = FALSE
      )
    },
    error = function(e) {
      data.frame(
        package = package_name,
        status = "failed",
        message = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    }
  )

  if (result$status == "failed") {
    cat("INSTALLATION FAILED:", result$message, "\n")
  } else {
    cat("Installation completed.\n")
  }

  result
}

for (package_name in packages) {
  install_results <- rbind(
    install_results,
    install_one(package_name)
  )
}

# ------------------------------------------------------------
# 4. Snapshot environment
# ------------------------------------------------------------

cat("\nCreating renv.lock...\n")

renv::snapshot(
  type = "all",
  prompt = FALSE
)

# ------------------------------------------------------------
# 5. Package load tests
# ------------------------------------------------------------

cat("\n========================================\n")
cat("PACKAGE LOAD TESTS\n")
cat("========================================\n")

load_results <- data.frame(
  package = character(),
  load_status = character(),
  message = character(),
  stringsAsFactors = FALSE
)

for (package_name in packages) {
  load_result <- tryCatch(
    {
      if (requireNamespace(package_name, quietly = TRUE)) {
        data.frame(
          package = package_name,
          load_status = "loaded",
          message = "",
          stringsAsFactors = FALSE
        )
      } else {
        data.frame(
          package = package_name,
          load_status = "not available",
          message = "Package could not be loaded.",
          stringsAsFactors = FALSE
        )
      }
    },
    error = function(e) {
      data.frame(
        package = package_name,
        load_status = "failed",
        message = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    }
  )

  load_results <- rbind(
    load_results,
    load_result
  )

  cat(
    package_name,
    ":",
    load_result$load_status,
    "\n"
  )
}

# ------------------------------------------------------------
# 6. Version and dependency checks
# ------------------------------------------------------------

cat("\n========================================\n")
cat("PACKAGE VERSIONS\n")
cat("========================================\n")

installed_packages <- installed.packages()

version_results <- data.frame(
  package = packages,
  installed = packages %in% rownames(installed_packages),
  version = NA_character_,
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(version_results))) {
  package_name <- version_results$package[i]

  if (version_results$installed[i]) {
    version_results$version[i] <-
      installed_packages[package_name, "Version"]
  }
}

print(version_results, row.names = FALSE)

cat("\n========================================\n")
cat("DEPENDENCY STATUS\n")
cat("========================================\n")

renv::status()

# ------------------------------------------------------------
# 7. Save reproducibility information
# ------------------------------------------------------------

writeLines(
  capture.output(sessioninfo::session_info()),
  "session_info.txt"
)

writeLines(
  capture.output(renv::status()),
  "renv_status.txt"
)

# ------------------------------------------------------------
# 8. AWS CLI availability check
# ------------------------------------------------------------

cat("\n========================================\n")
cat("AWS CLI CHECK\n")
cat("========================================\n")

aws_executable <- Sys.which("aws")

if (nzchar(aws_executable)) {
  cat("AWS CLI found:", aws_executable, "\n")

  aws_version <- system2(
    "aws",
    "--version",
    stdout = TRUE,
    stderr = TRUE
  )

  cat(aws_version, sep = "\n")
} else {
  cat("AWS CLI was not found on PATH.\n")
}

# ------------------------------------------------------------
# 9. Final summary
# ------------------------------------------------------------

cat("\n========================================\n")
cat("R ENVIRONMENT SETUP COMPLETE\n")
cat("========================================\n")

cat("R version:", R.version.string, "\n")

cat(
  "renv version:",
  as.character(packageVersion("renv")),
  "\n"
)

cat("Packages requested:", length(packages), "\n")

cat(
  "Installation successes:",
  sum(install_results$status == "installed"),
  "\n"
)

cat(
  "Installation failures:",
  sum(install_results$status == "failed"),
  "\n"
)

cat(
  "Packages successfully loadable:",
  sum(load_results$load_status == "loaded"),
  "\n"
)

cat("\nInstallation failures:\n")

print(
  install_results[
    install_results$status == "failed",
  ],
  row.names = FALSE
)

cat("\nPackages not successfully loadable:\n")

print(
  load_results[
    load_results$load_status != "loaded",
  ],
  row.names = FALSE
)

cat("\nGenerated files:\n")
cat("- renv.lock\n")
cat("- session_info.txt\n")
cat("- renv_status.txt\n")

cat("\nReview installation failures before proceeding.\n")