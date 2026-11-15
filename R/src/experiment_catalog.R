# Catalog building blocks. Never overwrites an existing catalog partition.
# Catalog publication/concurrency is intentionally not automated here.
openesm_catalog_row <- function(manifest,artifact_uri) {
  required <- c('experiment_id','dataset_id','model_family','config_hash','status')
  missing <- setdiff(required,names(manifest))
  if (length(missing)) stop('Missing manifest fields: ',paste(missing,collapse=', '))
  data.frame(experiment_id=manifest$experiment_id,
             dataset_id=as.character(manifest$dataset_id),
             model_family=manifest$model_family,
             config_hash=manifest$config_hash,
             status=manifest$status,
             manifest_uri=artifact_uri,stringsAsFactors=FALSE)
}

openesm_validate_catalog <- function(rows) {
  required <- c('experiment_id','dataset_id','model_family','config_hash','status','manifest_uri')
  if (!is.data.frame(rows) || !all(required %in% names(rows)))
    stop('Invalid catalog schema')
  if (anyNA(rows[required]) || anyDuplicated(rows$experiment_id))
    stop('Catalog has missing fields or duplicate experiment IDs')
  invisible(TRUE)
}

# Caller supplies authoritative completed manifests; does not scan S3 or
# silently promote unfinished experiments. Returns a data.frame for inspection.
openesm_build_experiment_catalog <- function(manifests,uris) {
  if (length(manifests)!=length(uris)) stop('Manifest/URI lengths differ')
  if (!length(manifests)) return(data.frame(
    experiment_id=character(),dataset_id=character(),model_family=character(),
    config_hash=character(),status=character(),manifest_uri=character()))
  if (any(!vapply(manifests,function(x) identical(x$status,'completed'),logical(1))))
    stop('Only completed experiments can be cataloged')
  rows <- do.call(rbind,Map(openesm_catalog_row,manifests,uris))
  openesm_validate_catalog(rows)
  rows
}

openesm_publish_catalog_snapshot <- function(rows,resolved,snapshot_id) {
  openesm_validate_catalog(rows)
  if (!grepl('^[A-Za-z0-9_-]+$',snapshot_id)) stop('Unsafe snapshot ID')
  prefix <- resolved$storage$s3$modeling$catalog
  key <- paste0(prefix,'snapshots/',snapshot_id,'/experiments.parquet')
  openesm_upload_artifact(rows,key,'parquet',resolved$storage,immutable=TRUE)
}
