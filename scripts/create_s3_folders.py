
import boto3

BUCKET_NAME = "openesm-8917a696ff634755199b686587"
REGION = "us-east-2"

session = boto3.Session(
    profile_name="openesm-terraform",
    region_name=REGION,
)

s3 = session.client("s3")

folders = ["metadata/", "raw/", "processed/", "curated/", "modeling/"]

for folder in folders:
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=folder,
        Body=b"",
    )
    print(f"Created s3://{BUCKET_NAME}/{folder}")
