
# ============================================================
# OpenESM - Public Read-Only S3 Access
# ============================================================
#
# Public permissions:
#   - List all bucket objects
#   - Download all bucket objects
#
# No public permissions to upload, overwrite, or delete.
#
# WARNING:
# All existing and future objects will be publicly
# readable. Verify redistribution rights before applying.
# ============================================================

data "aws_iam_policy_document" "public_read" {

  statement {
    sid    = "AllowPublicBucketListing"
    effect = "Allow"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = [
      "s3:ListBucket"
    ]

    resources = [
      aws_s3_bucket.project_data.arn
    ]
  }

  statement {
    sid    = "AllowPublicObjectReading"
    effect = "Allow"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = [
      "s3:GetObject"
    ]

    resources = [
      "${aws_s3_bucket.project_data.arn}/*"
    ]
  }
}

resource "aws_s3_bucket_policy" "public_read" {
  bucket = aws_s3_bucket.project_data.id

  policy = data.aws_iam_policy_document.public_read.json

  depends_on = [
    aws_s3_bucket_public_access_block.project_data
  ]
}
