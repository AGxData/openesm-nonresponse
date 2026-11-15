output "s3_bucket_name" {
  description = "S3 bucket for project data"
  value       = aws_s3_bucket.project_data.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the project bucket"
  value       = aws_s3_bucket.project_data.arn
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}