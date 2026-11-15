# Explicit experiment lifecycle and immutable per-fold checkpoint artifacts.
openesm_new_experiment_id <- function() {
  if (!requireNamespace('uuid',quietly=TRUE)) stop('Install package: uuid')
  uuid::UUIDgenerate()
}

openesm_experiment_prefix <- function(resolved,experiment_id) {
  openesm_s3_key(resolved$storage$s3$modeling$experiments,
                 resolved$model_family,resolved$dataset_id,experiment_id)
}

openesm_experiment_manifest <- function(resolved,experiment_id,status='pending') {
  if (!status %in% c('pending','running','interrupted','failed','completed','cancelled'))
    stop('Invalid experiment status')
  list(experiment_id=experiment_id,dataset_id=resolved$dataset_id,
       model_family=resolved$model_family,config_hash=resolved$config_hash,
       status=status,created_at=format(Sys.time(),'%Y-%m-%dT%H:%M:%SZ',tz='UTC'),
       artifacts=list())
}

# A checkpoint consists of a single immutable RDS payload and its manifest.
# The manifest is uploaded last. No overwrites are permitted.
openesm_save_fold_checkpoint <- function(resolved,experiment_id,fold_id,payload) {
  if (!is.list(payload)) stop('Checkpoint payload must be a named list')
  if (!grepl('^[A-Za-z0-9_-]+$',fold_id)) stop('Unsafe fold ID')
  prefix <- openesm_s3_key(resolved$storage$s3$modeling$checkpoints,
                           resolved$model_family,resolved$dataset_id,experiment_id)
  key <- paste0(prefix,fold_id,'.rds')
  manifest_key <- paste0(prefix,fold_id,'.json')
  if (openesm_s3_exists(resolved$storage$aws$bucket,manifest_key,resolved$storage))
    stop('Fold already checkpointed: ',fold_id)
  payload$checkpoint_metadata <- list(fold_id=fold_id,
      experiment_id=experiment_id,config_hash=resolved$config_hash)
  artifact <- openesm_upload_artifact(payload,key,'rds',resolved$storage,immutable=TRUE)
  manifest <- c(list(fold_id=fold_id,experiment_id=experiment_id,
                     config_hash=resolved$config_hash,format='rds'),artifact)
  openesm_upload_artifact(manifest,manifest_key,'json',resolved$storage,immutable=TRUE)
  manifest
}

openesm_load_fold_checkpoint <- function(resolved,experiment_id,fold_id) {
  prefix <- openesm_s3_key(resolved$storage$s3$modeling$checkpoints,
                           resolved$model_family,resolved$dataset_id,experiment_id)
  manifest_path <- tempfile(fileext='.json'); data_path <- tempfile(fileext='.rds')
  on.exit(unlink(c(manifest_path,data_path)),add=TRUE)
  openesm_download_file(paste0(prefix,fold_id,'.json'),manifest_path,resolved$storage)
  manifest <- openesm_deserialize(manifest_path,'json')
  if (!identical(manifest$config_hash,resolved$config_hash)) stop('Checkpoint config hash mismatch')
  openesm_download_file(manifest$key,data_path,resolved$storage)
  if (!identical(openesm_file_sha256(data_path),manifest$sha256)) stop('Checkpoint checksum mismatch')
  payload <- readRDS(data_path)
  if (!identical(payload$checkpoint_metadata$fold_id,fold_id)) stop('Checkpoint fold mismatch')
  payload
}

# Only completed experiment manifests are publishable. Caller must supply a
# verified list of expected folds and artifacts; this function does not infer it.
openesm_publish_completion <- function(resolved,experiment_id,manifest,expected_fold_ids) {
  if (!length(expected_fold_ids)) stop('Expected fold IDs required')
  if (!identical(manifest$config_hash,resolved$config_hash)) stop('Config mismatch')
  if (!identical(sort(unique(manifest$completed_fold_ids)),sort(unique(expected_fold_ids))))
    stop('Incomplete fold list')
  prefix <- openesm_experiment_prefix(resolved,experiment_id)
  manifest$status <- 'completed'
  manifest$completed_at <- format(Sys.time(),'%Y-%m-%dT%H:%M:%SZ',tz='UTC')
  openesm_upload_artifact(manifest,paste0(prefix,'COMPLETED.json'),'json',resolved$storage,immutable=TRUE)
}
