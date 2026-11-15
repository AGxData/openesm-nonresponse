# OpenESM global progress tracker.
# Existing update signatures are preserved. Resume support adds restore().
create_global_progress <- function(total_inner_fits, total_outer_fits,
                                   total_calibration_fits = 0L,
                                   refresh_seconds = 1) {
  validate_count <- function(x, name) {
    if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
        x < 0 || x != floor(x)) stop(name, ' must be a nonnegative integer.')
    as.numeric(x)
  }
  total_inner_fits <- validate_count(total_inner_fits, 'total_inner_fits')
  total_outer_fits <- validate_count(total_outer_fits, 'total_outer_fits')
  total_calibration_fits <- validate_count(total_calibration_fits, 'total_calibration_fits')
  total_fits <- total_inner_fits + total_outer_fits + total_calibration_fits
  if (total_fits < 1) stop('The total fitting workload must be positive.')
  if (!is.numeric(refresh_seconds) || length(refresh_seconds) != 1L ||
      !is.finite(refresh_seconds) || refresh_seconds <= 0) {
    stop('refresh_seconds must be a positive number.')
  }
  state <- new.env(parent = emptyenv())
  state$inner <- 0
  state$calibration <- 0
  state$outer <- 0
  state$recovered <- c(inner = 0, calibration = 0, outer = 0)
  state$started <- Sys.time()
  state$last_refresh <- as.POSIXct(NA)
  state$last_width <- 0L
  state$last_stage <- ''
  state$closed <- FALSE
  fmt_time <- function(seconds) {
    if (!is.finite(seconds) || seconds < 0) return('--:--:--')
    seconds <- floor(seconds)
    sprintf('%02d:%02d:%02d', floor(seconds / 3600),
            floor((seconds %% 3600) / 60), seconds %% 60)
  }
  render <- function(stage, details = '', force = FALSE) {
    now <- Sys.time()
    done <- state$inner + state$calibration + state$outer
    due <- is.na(state$last_refresh) ||
      as.numeric(difftime(now, state$last_refresh, units = 'secs')) >= refresh_seconds
    if (!force && !due && identical(stage, state$last_stage) && done < total_fits) {
      return(invisible(NULL))
    }
    elapsed <- as.numeric(difftime(now, state$started, units = 'secs'))
    newly_done <- done - sum(state$recovered)
    remaining <- total_fits - done
    eta <- if (remaining == 0) 0 else if (newly_done > 0) {
      elapsed * remaining / newly_done
    } else NA_real_
    line <- paste0(
      sprintf('%5.1f%% (%s/%s)', 100 * done / total_fits,
              format(done, scientific = FALSE, trim = TRUE),
              format(total_fits, scientific = FALSE, trim = TRUE)),
      ' | ', stage,
      if (nzchar(details)) paste0(' | ', details) else '',
      ' | recovered ', format(sum(state$recovered), scientific = FALSE, trim = TRUE),
      ' | ', fmt_time(elapsed), ' elapsed | ', fmt_time(eta), ' ETA'
    )
    width <- nchar(line, type = 'width')
    cat('\r', line, strrep(' ', max(0L, state$last_width - width)), sep = '')
    flush.console()
    state$last_width <- width
    state$last_refresh <- now
    state$last_stage <- stage
    invisible(NULL)
  }
  restore <- function(inner_fits = 0L, calibration_fits = 0L,
                      outer_fits = 0L) {
    if (state$closed) stop('Cannot restore a closed tracker.')
    if (state$inner + state$calibration + state$outer != 0) {
      stop('restore() must be called before any fitting updates.')
    }
    counts <- c(inner = validate_count(inner_fits, 'inner_fits'),
                calibration = validate_count(calibration_fits, 'calibration_fits'),
                outer = validate_count(outer_fits, 'outer_fits'))
    limits <- c(inner = total_inner_fits, calibration = total_calibration_fits,
                outer = total_outer_fits)
    if (any(counts > limits)) stop('Recovered fits exceed the workload plan.')
    state$inner <- counts[['inner']]
    state$calibration <- counts[['calibration']]
    state$outer <- counts[['outer']]
    state$recovered <- counts
    render('Recovered checkpoints', force = TRUE)
    invisible(NULL)
  }
  snapshot <- function() {
    done <- c(inner = state$inner, calibration = state$calibration,
              outer = state$outer)
    list(planned = c(inner = total_inner_fits,
                     calibration = total_calibration_fits,
                     outer = total_outer_fits),
         completed = done,
         recovered = state$recovered,
         executed_this_session = done - state$recovered,
         complete = sum(done) == total_fits)
  }
  update_inner <- function(outer_fold, outer_total, inner_fold, inner_total, alpha) {
    if (state$closed) stop('Cannot update a closed progress tracker.')
    if (state$inner >= total_inner_fits) stop('Inner workload exceeded plan.')
    state$inner <- state$inner + 1
    render('Tuning', paste0('Outer ', outer_fold, '/', outer_total,
                           ' | Inner ', inner_fold, '/', inner_total,
                           ' | Alpha ', alpha))
    invisible(NULL)
  }
  update_calibration <- function(outer_fold, outer_total,
                                 calibration_fold, calibration_total) {
    if (state$closed) stop('Cannot update a closed progress tracker.')
    if (state$calibration >= total_calibration_fits) {
      stop('Calibration workload exceeded plan.')
    }
    state$calibration <- state$calibration + 1
    render('Calibration', paste0('Outer ', outer_fold, '/', outer_total,
                                ' | Fold ', calibration_fold, '/', calibration_total))
    invisible(NULL)
  }
  update_refit <- function(outer_fold, outer_total) {
    if (state$closed) stop('Cannot update a closed progress tracker.')
    if (state$outer >= total_outer_fits) stop('Outer refits exceeded plan.')
    state$outer <- state$outer + 1
    render('Outer refit', paste0('Outer ', outer_fold, '/', outer_total))
    invisible(NULL)
  }
  close_progress <- function() {
    if (state$closed) return(invisible(NULL))
    done <- state$inner + state$calibration + state$outer
    render(if (done == total_fits) 'Complete' else 'Incomplete', force = TRUE)
    cat('\n')
    state$closed <- TRUE
    if (done != total_fits) warning('Progress tracker closed after ', done,
                                   ' of ', total_fits, ' planned fits.', call. = FALSE)
    invisible(NULL)
  }
  render('Starting', force = TRUE)
  list(update_inner = update_inner, update_calibration = update_calibration,
       update_refit = update_refit, restore = restore, snapshot = snapshot,
       close = close_progress)
}