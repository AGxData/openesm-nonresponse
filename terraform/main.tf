
# ============================================================
# OpenESM - S3 Bucket Infrastructure
# ============================================================

resource "aws_s3_bucket" "project_data" {
  bucket_prefix = "openesm-"
  force_destroy = false

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Allow public bucket policies while blocking public ACLs.

resource "aws_s3_bucket_public_access_block" "project_data" {
  bucket = aws_s3_bucket.project_data.id

  block_public_acls       = true
  block_public_policy     = false
  ignore_public_acls      = true
  restrict_public_buckets = false
}

# Keep encryption enabled.

resource "aws_s3_bucket_server_side_encryption_configuration" "project_data" {
  bucket = aws_s3_bucket.project_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Keep versioning enabled.

resource "aws_s3_bucket_versioning" "project_data" {
  bucket = aws_s3_bucket.project_data.id

  versioning_configuration {
    status = "Enabled"
  }
}
