import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = "openesm-8917a696ff634755199b686587"

session = boto3.Session(
    profile_name="openesm-terraform",
    region_name="us-east-2",
)

s3 = session.client("s3")

try:
    s3.head_bucket(Bucket=BUCKET_NAME)
    print(f"Success: connected to bucket {BUCKET_NAME}")
except ClientError as e:
    print(f"S3 request failed: {e}")
