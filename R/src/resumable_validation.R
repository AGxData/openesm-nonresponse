# OpenESM fold-level S3 checkpoint adapter for the existing validation API.
# Source AFTER artifact_storage.R and experiment_tracking.R.
# Does not change the statistical fitting algorithm.

openesm_fold_signature <- function(train_data, test_data, features, target,
                                   resolved, code_hash) {
  if (!requireNamespace('digest', quietly=TRUE)) stop('Install digest')
  digest::digest(list(
    train_keys=train_data[,c('.validation_row_id','id','day','beep',target),drop=FALSE],
    test_keys=test_data[,c('.validation_row_id','id','day','beep',target),drop=FALSE],
    features=features, config_hash=resolved$config_hash, code_hash=code_hash
  ), algo='sha256')
}

openesm_checkpoint_exists <- function(resolved, experiment_id, fold_id) {
  prefix <- openesm_s3_key(resolved$storage$s3$modeling$checkpoints,
                           resolved$model_family,resolved$dataset_id,experiment_id)
  openesm_s3_exists(resolved$storage$aws$bucket,
                    paste0(prefix,fold_id,'.json'),resolved$storage)
}

# Stateful wrapper: caller supplies getter/setter closures for existing logs and
# fold counter. A completed fold is checkpointed before its prediction is returned.
openesm_resumable_model_func <- function(model_func, resolved, experiment_id,
                                         features, target, code_hash,
                                         get_state, set_state,
                                         allow_resume=TRUE) {
  stopifnot(is.function(model_func), is.function(get_state), is.function(set_state))
  force(model_func); force(resolved); force(experiment_id)
  function(train_data,test_data,features,target) {
    state_before <- get_state()
    fold_number <- as.integer(state_before$outer_completed) + 1L
    fold_id <- sprintf('outer_%05d',fold_number)
    signature <- openesm_fold_signature(train_data,test_data,features,target,
                                        resolved,code_hash)
    if (isTRUE(allow_resume) && openesm_checkpoint_exists(resolved,experiment_id,fold_id)) {
      checkpoint <- openesm_load_fold_checkpoint(resolved,experiment_id,fold_id)
      if (!identical(checkpoint$signature,signature))
        stop('Checkpoint signature mismatch for ',fold_id,
             ': data, code or fold membership changed')
      if (!identical(as.integer(checkpoint$fold_number),fold_number))
        stop('Checkpoint order mismatch: ',fold_id)
      if (!identical(as.integer(checkpoint$state$outer_completed),fold_number))
        stop('Checkpoint state invalid: ',fold_id)
      if (!identical(as.integer(checkpoint$prediction$.validation_row_id),
                     as.integer(test_data$.validation_row_id)))
        stop('Checkpoint prediction row IDs differ: ',fold_id)
      set_state(checkpoint$state)
      message('Resumed ',fold_id,' from verified S3 checkpoint')
      return(checkpoint$prediction)
    }
    prediction <- model_func(train_data,test_data,features,target)
    if (!is.data.frame(prediction) ||
        !all(c('.validation_row_id','predicted_prob') %in% names(prediction)))
      stop('Model must return a prediction data.frame with row IDs')
    if (!identical(as.integer(prediction$.validation_row_id),
                   as.integer(test_data$.validation_row_id)))
      stop('Prediction IDs do not match assessment rows')
    state_after <- get_state()
    if (!identical(as.integer(state_after$outer_completed),fold_number))
      stop('Model fold counter did not advance by one')
    payload <- list(fold_number=fold_number,signature=signature,
                    prediction=prediction,state=state_after)
    openesm_save_fold_checkpoint(resolved,experiment_id,fold_id,payload)
    message('Checkpointed ',fold_id,' to S3')
    prediction
  }
}

# Record a unique experiment ID outside the training script and pass it via env.
# Reusing the same ID resumes; changing it starts a new experiment.
openesm_experiment_id_from_env <- function() {
  id <- Sys.getenv('OPENESM_EXPERIMENT_ID','')
  if (!nzchar(id)) stop('Set OPENESM_EXPERIMENT_ID to a UUID before training')
  if (!grepl('^[A-Za-z0-9-]+$',id)) stop('Unsafe experiment ID')
  id
}
