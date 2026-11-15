# Explicit winner selection and promotion; never auto-promotes.
openesm_rank_experiments <- function(metrics,primary_metric='log_loss',direction='minimize') {
  required <- c('experiment_id','dataset_id','model_family','protocol_hash',
                'assessment_population_hash',primary_metric,'status')
  if (!is.data.frame(metrics) || !all(required %in% names(metrics)))
    stop('Missing registry comparison fields: ',paste(setdiff(required,names(metrics)),collapse=', '))
  if (!direction %in% c('minimize','maximize')) stop('Invalid direction')
  if (any(metrics$status!='completed')) stop('Only completed experiments can be ranked')
  if (length(unique(metrics$dataset_id))!=1L ||
      length(unique(metrics$protocol_hash))!=1L ||
      length(unique(metrics$assessment_population_hash))!=1L)
    stop('Experiments must share dataset, protocol, and assessment population')
  if (anyNA(metrics[[primary_metric]]) || any(!is.finite(metrics[[primary_metric]])))
    stop('Ranking metric must be finite')
  metrics[order(metrics[[primary_metric]],decreasing=identical(direction,'maximize')), ,drop=FALSE]
}

openesm_promotion_record <- function(experiment_id,model_family,dataset_id,
                                     model_version,rationale,metric_name,metric_value,
                                     protocol_hash,approved_by) {
  vals <- c(experiment_id,model_family,dataset_id,model_version,rationale,
            metric_name,protocol_hash,approved_by)
  if (anyNA(vals) || any(!nzchar(vals))) stop('Promotion requires all fields and explicit approver')
  if (!is.finite(metric_value)) stop('Invalid metric')
  list(experiment_id=experiment_id,model_family=model_family,dataset_id=dataset_id,
       model_version=model_version,rationale=rationale,metric_name=metric_name,
       metric_value=metric_value,protocol_hash=protocol_hash,approved_by=approved_by,
       promoted_at=format(Sys.time(),'%Y-%m-%dT%H:%M:%SZ',tz='UTC'))
}

# Immutable version record only. The registry's mutable current pointer is
# deliberately not implemented until concurrency control is available.
openesm_publish_model_version <- function(record,resolved) {
  if (!identical(record$dataset_id,resolved$dataset_id) ||
      !identical(record$model_family,resolved$model_family)) stop('Promotion scope mismatch')
  prefix <- openesm_s3_key(resolved$storage$s3$modeling$best_versions,
      resolved$model_family,resolved$dataset_id,model_version=record$model_version)
  openesm_upload_artifact(record,paste0(prefix,'promotion.json'),'json',resolved$storage,immutable=TRUE)
}
