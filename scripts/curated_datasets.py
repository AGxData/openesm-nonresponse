from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd

# ============================================================
# Configuration
# ============================================================

AWS_PROFILE = "openesm-terraform"
AWS_REGION = "us-east-2"
BUCKET = "openesm-8917a696ff634755199b686587"
PREFIX = "curated/"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "curated_dataset_summary.csv"

# ============================================================
# AWS S3 Connection
# ============================================================

session = boto3.Session(
    profile_name=AWS_PROFILE,
    region_name=AWS_REGION,
)

s3 = session.client("s3")
paginator = s3.get_paginator("list_objects_v2")

# ============================================================
# Process Curated Datasets
# ============================================================

rows = []

for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]

        if not key.endswith(".parquet"):
            continue

        dataset_id = Path(key).stem

        print(f"Processing {dataset_id}...")

        response = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(BytesIO(response["Body"].read()))

        rows.append(
            {
                "id": dataset_id,
                "n_participants": df["id"].nunique(),
                "n_days": df["day"].nunique(),
                "n_beeps": df["beep"].nunique(),
                "n_rows": len(df),
               "response_rate": df["responded"].mean() * 100,
            }
        )

# ============================================================
# Create Summary
# ============================================================

columns = [
    "id",
    "n_participants",
    "n_days",
    "n_beeps",
    "n_rows",
    "response_rate",
]

summary_df = pd.DataFrame(rows, columns=columns)

summary_df = (
    summary_df.sort_values("id")
    .reset_index(drop=True)
)

summary_df["response_rate"] = summary_df["response_rate"].round(2)

# ============================================================
# Export CSV
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

summary_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nProcessed {len(summary_df)} curated datasets.")
print(f"Saved to: {OUTPUT_PATH}")

print("\nDataset Summary:")
print(summary_df.to_string(index=False))