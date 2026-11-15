#!/usr/bin/env Rscript
# Safe by default: resolve/inspect only. Does NOT launch regularized_logistic.R.
# Usage: Rscript R/modeling_runner.R 0006 regularized_logistic
args <- commandArgs(trailingOnly=TRUE)
if (length(args) > 2L) stop('Usage: Rscript R/modeling_runner.R [dataset_id] [model_family]')
dataset_id <- if (length(args)>=1L) args[[1L]] else '0006'
model_family <- if (length(args)>=2L) args[[2L]] else 'regularized_logistic'
if (!file.exists('R/src/configuration.R')) stop('Run from repository root')
source('R/src/configuration.R')
resolved <- openesm_resolve_config(dataset_id,model_family)
params <- openesm_model_parameters(resolved)
cat('OpenESM configuration inspection (no model fitting)\n')
cat('Dataset:',resolved$dataset_id,'\nModel:',resolved$model_family,
    '\nConfig hash:',resolved$config_hash,'\n')
print(params)
cat('\nNo data read, models fitted, or S3 objects written.\n')
