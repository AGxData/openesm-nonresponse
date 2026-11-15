# OpenESM configuration resolution. Additive; does not source or modify models.
openesm_read_yaml <- function(path) {
  if (!requireNamespace('yaml', quietly = TRUE)) stop('Install package: yaml')
  if (!file.exists(path)) stop('Missing configuration: ', path)
  out <- yaml::read_yaml(path)
  if (!is.list(out)) stop('YAML root must be a mapping: ', path)
  out
}

openesm_merge <- function(base, override, path = '') {
  if (is.null(override)) return(base)
  if (!is.list(base) || !is.list(override) || is.null(names(base)) || is.null(names(override))) return(override)
  for (key in names(override)) {
    child <- if (nzchar(path)) paste(path, key, sep='.') else key
    if (key %in% names(base)) base[[key]] <- openesm_merge(base[[key]], override[[key]], child)
    else base[[key]] <- override[[key]]
  }
  base
}

openesm_assert_known_keys <- function(base, override, path='') {
  if (!is.list(override) || is.null(names(override))) return(invisible(TRUE))
  for (key in names(override)) {
    child <- if (nzchar(path)) paste(path,key,sep='.') else key
    if (!key %in% names(base)) stop('Unknown configuration key: ', child)
    if (is.list(override[[key]]) && !is.null(names(override[[key]])) &&
        is.list(base[[key]]) && !is.null(names(base[[key]]))) {
      openesm_assert_known_keys(base[[key]], override[[key]], child)
    }
  }
  invisible(TRUE)
}

openesm_hash <- function(object) {
  if (!requireNamespace('digest', quietly=TRUE)) stop('Install package: digest')
  digest::digest(object, algo='sha256', serialize=TRUE)
}

openesm_protocol_validation <- function(protocols, name) {
  p <- protocols$protocols[[name]]
  if (is.null(p)) stop('Unknown validation protocol: ', name)
  if (identical(p$inner$mode, 'inherit')) p$inner$mode <- p$outer$mode
  p$protocol <- name
  p
}

# Strict overrides are checked against model defaults, with validation protocol
# fields allowed separately. Named protocol is selected before merging overrides.
openesm_resolve_config <- function(dataset_id, model_family='regularized_logistic',
                                   config_dir='config', run_overrides=list()) {
  dataset_id <- as.character(dataset_id)
  if (!grepl('^[0-9]{4}$', dataset_id)) stop('dataset_id must be four digits, e.g. 0006')
  global <- openesm_read_yaml(file.path(config_dir,'modeling.yaml'))
  storage <- openesm_read_yaml(file.path(config_dir,'storage.yaml'))
  defaults <- openesm_read_yaml(file.path(config_dir,'models',paste0(model_family,'.yaml')))
  protocols <- openesm_read_yaml(file.path(config_dir,'validation_protocols.yaml'))
  overrides <- openesm_read_yaml(file.path(config_dir,'datasets','modeling_overrides.yaml'))
  if (!identical(defaults$model$family, model_family)) stop('Model family mismatch')
  entry <- overrides$datasets[[dataset_id]]
  all_override <- if (is.null(entry$all_models)) list() else entry$all_models
  model_override <- if (is.null(entry$models[[model_family]])) list() else entry$models[[model_family]]
  if (!is.list(run_overrides)) stop('run_overrides must be a list')
  # Establish the winning protocol name, then expand it exactly once.
  name <- defaults$validation$protocol
  for (ov in list(all_override,model_override,run_overrides)) {
    if (!is.null(ov$validation$protocol)) name <- ov$validation$protocol
  }
  expanded <- openesm_protocol_validation(protocols, name)
  # Protocol defines legal nested validation keys, model defines other keys.
  base <- defaults
  base$validation <- openesm_merge(expanded, defaults$validation[names(defaults$validation) != 'protocol'])
  base$validation$protocol <- name
  for (ov in list(all_override,model_override,run_overrides)) {
    if (isTRUE(global$schema$strict_validation)) openesm_assert_known_keys(base, ov)
    base <- openesm_merge(base, ov)
  }
  base$validation$protocol <- name
  if (!identical(base$model$family, model_family)) stop('Model family cannot change via override')
  if (!identical(base$model$target, 'nonresponse')) stop('Unsupported target')
  if (!base$validation$outer$mode %in% c('within_day','across_day')) stop('Invalid outer mode')
  if (!base$validation$inner$mode %in% c('within_day','across_day')) stop('Invalid inner mode')
  if (!identical(base$validation$outer$window, 'expanding') ||
      !identical(base$validation$inner$window, 'expanding')) stop('Only expanding windows currently supported')
  if (!identical(base$validation$outer$temporal_gap_beeps, 0L) &&
      !identical(as.numeric(base$validation$outer$temporal_gap_beeps), 0))
    stop('Nonzero temporal gaps are not supported by existing validations.R')
  if (base$validation$outer$initial_days < 1 || base$validation$inner$initial_days < 1) stop('Invalid initial_days')
  if (base$tuning$minimum_inner_folds < 2) stop('Invalid minimum_inner_folds')
  if (base$tuning$lambda$count < 2 || base$tuning$lambda$minimum <= 0 ||
      base$tuning$lambda$maximum <= base$tuning$lambda$minimum) stop('Invalid lambda grid')
  if (any(base$tuning$alpha_grid < 0 | base$tuning$alpha_grid > 1)) stop('Invalid alpha grid')
  if (base$calibration$threshold_grid$step <= 0) stop('Invalid threshold step')
  resolved <- list(dataset_id=dataset_id, model_family=model_family,
                   model=base, global=global, storage=storage)
  resolved$config_hash <- openesm_hash(resolved)
  resolved
}

openesm_model_parameters <- function(resolved) {
  c <- resolved$model
  list(dataset_id=resolved$dataset_id, validation_mode=c$validation$outer$mode,
       initial_days=as.integer(c$validation$outer$initial_days),
       inner_validation_mode=c$validation$inner$mode,
       inner_initial_days=as.integer(c$validation$inner$initial_days),
       minimum_inner_folds=as.integer(c$tuning$minimum_inner_folds),
       alpha_grid=as.numeric(unlist(c$tuning$alpha_grid)),
       lambda_grid=exp(seq(log(c$tuning$lambda$maximum),log(c$tuning$lambda$minimum),
                           length.out=as.integer(c$tuning$lambda$count))),
       lambda_choices=unlist(c$tuning$lambda_choices),
       threshold_grid=seq(c$calibration$threshold_grid$start,
                          c$calibration$threshold_grid$stop,
                          by=c$calibration$threshold_grid$step),
       threshold_metric=c$calibration$threshold_metric,
       beta=c$evaluation$beta, random_seed=c$execution$random_seed,
       preflight_only=isTRUE(c$execution$preflight_only))
}
