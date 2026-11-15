# S3-first artifact utilities. All writes are explicit; no side effects on source().
openesm_s3_key <- function(template, model_family, dataset_id, experiment_id=NULL, model_version=NULL) {
  values <- list(model_family=model_family,dataset_id=dataset_id,
                 experiment_id=experiment_id,model_version=model_version)
  key <- template
  for (name in names(values)) if (!is.null(values[[name]]))
    key <- gsub(paste0('{',name,'}'),as.character(values[[name]]),key,fixed=TRUE)
  if (grepl('\\{[^}]+\\}',key)) stop('Unresolved S3 prefix template: ',key)
  if (grepl('(^/|\\.\\.|\\\\)',key)) stop('Unsafe S3 key: ',key)
  key
}

openesm_s3_uri <- function(bucket,key) paste0('s3://',bucket,'/',key)

openesm_aws_env <- function(storage_config) {
  aws <- storage_config$aws
  if (!nzchar(aws$bucket)) stop('AWS bucket missing')
  # R's aws.s3 package honors AWS_PROFILE and AWS_DEFAULT_REGION.
  # No temporary long-lived credentials are exported or logged here.
  c(AWS_PROFILE=aws$profile, AWS_DEFAULT_REGION=aws$region)
}

openesm_with_aws <- function(storage_config, expr) {
  vars <- openesm_aws_env(storage_config)
  keys <- names(vars)
  old <- Sys.getenv(keys,unset=NA_character_)
  on.exit(for (i in seq_along(keys)) {
    if (is.na(old[[i]])) Sys.unsetenv(keys[[i]])
    else do.call(Sys.setenv,setNames(list(old[[i]]),keys[[i]]))
  },add=TRUE)
  do.call(Sys.setenv,as.list(vars))
  force(expr)
}

openesm_file_sha256 <- function(path) {
  if (!requireNamespace('digest',quietly=TRUE)) stop('Install package: digest')
  digest::digest(file=path,algo='sha256')
}

openesm_serialize <- function(object, path, format=c('parquet','rds','json','yaml','jsonl')) {
  format <- match.arg(format)
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  if (format=='parquet') {
    if (!requireNamespace('arrow',quietly=TRUE)) stop('Install package: arrow')
    arrow::write_parquet(object,path,compression='zstd')
  } else if (format=='rds') saveRDS(object,path)
  else if (format=='yaml') {
    if (!requireNamespace('yaml',quietly=TRUE)) stop('Install package: yaml')
    yaml::write_yaml(object,path)
  } else {
    if (!requireNamespace('jsonlite',quietly=TRUE)) stop('Install package: jsonlite')
    if (format=='json') jsonlite::write_json(object,path,auto_unbox=TRUE,pretty=TRUE,null='null')
    else {
      con <- file(path,'wt'); on.exit(close(con),add=TRUE)
      for (item in object) writeLines(jsonlite::toJSON(item,auto_unbox=TRUE,null='null'),con)
    }
  }
  invisible(path)
}

openesm_deserialize <- function(path,format=c('parquet','rds','json','yaml','jsonl')) {
  format <- match.arg(format)
  if (format=='parquet') return(as.data.frame(arrow::read_parquet(path)))
  if (format=='rds') return(readRDS(path))
  if (format=='yaml') return(yaml::read_yaml(path))
  if (format=='json') return(jsonlite::read_json(path,simplifyVector=TRUE))
  lapply(readLines(path,warn=FALSE),jsonlite::fromJSON)
}

openesm_s3_exists <- function(bucket,key,storage_config) {
  if (!requireNamespace('aws.s3',quietly=TRUE)) stop('Install package: aws.s3')
  openesm_with_aws(storage_config, {
    x <- tryCatch(aws.s3::head_object(object=key,bucket=bucket),error=function(e) NULL)
    !is.null(x) && !identical(x,FALSE)
  })
}

# Upload with local SHA256 and an optional verified download. A HEAD/ETag is
# not a trustworthy content SHA256 for multipart uploads.
openesm_upload_file <- function(path,key,storage_config,immutable=TRUE,verify=TRUE) {
  if (!file.exists(path)) stop('Missing local artifact: ',path)
  if (!requireNamespace('aws.s3',quietly=TRUE)) stop('Install package: aws.s3')
  bucket <- storage_config$aws$bucket
  sha <- openesm_file_sha256(path)
  if (immutable && openesm_s3_exists(bucket,key,storage_config))
    stop('Immutable artifact already exists: ',openesm_s3_uri(bucket,key))
  retries <- storage_config$storage$writes$max_retries
  if (is.null(retries)) retries <- 3L
  for (attempt in seq_len(as.integer(retries)+1L)) {
    ok <- tryCatch(openesm_with_aws(storage_config,
       aws.s3::put_object(file=path,object=key,bucket=bucket)),error=function(e) FALSE)
    if (isTRUE(ok)) break
    if (attempt > retries) stop('Upload failed after retries: ',key)
    Sys.sleep(min(2^(attempt-1),30))
  }
  if (verify) {
    downloaded <- tempfile(); on.exit(unlink(downloaded),add=TRUE)
    openesm_download_file(key,downloaded,storage_config,overwrite=TRUE)
    if (!identical(sha,openesm_file_sha256(downloaded))) stop('S3 checksum mismatch: ',key)
  }
  list(s3_uri=openesm_s3_uri(bucket,key),key=key,sha256=sha,
       size_bytes=unname(file.info(path)$size))
}

openesm_download_file <- function(key,path,storage_config,overwrite=FALSE) {
  if (file.exists(path) && !overwrite) stop('Local path exists: ',path)
  if (!requireNamespace('aws.s3',quietly=TRUE)) stop('Install package: aws.s3')
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  ok <- openesm_with_aws(storage_config,
    aws.s3::save_object(object=key,bucket=storage_config$aws$bucket,file=path))
  if (!isTRUE(ok) || !file.exists(path)) stop('Download failed: ',key)
  invisible(path)
}

openesm_upload_artifact <- function(object,key,format,storage_config,immutable=TRUE) {
  path <- tempfile(fileext=paste0('.',format)); on.exit(unlink(path),add=TRUE)
  openesm_serialize(object,path,format)
  openesm_upload_file(path,key,storage_config,immutable=immutable,
                      verify=isTRUE(storage_config$storage$integrity$verify_uploads))
}
