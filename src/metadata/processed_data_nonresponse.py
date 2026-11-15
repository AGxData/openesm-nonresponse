
# Imports
from pathlib import Path
from io import BytesIO

import boto3
import pandas as pd


# Configuration
BUCKET = "openesm-8917a696ff634755199b686587"
PREFIX = "curated/"
PROFILE = "openesm-terraform"
REGION = "us-east-2"

RESPONSE_COLUMN = "responded"


# Local Output Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "metadata"
OUTPUT_FILENAME = "response_rate_summary.parquet"
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME


# S3 Connection
session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION,
)

s3 = session.client("s3")

paginator = s3.get_paginator("list_objects_v2")


# Helper Functions
def get_dataset_id(key):
    relative_key = key[len(PREFIX):]
    first_directory = relative_key.split("/")[0]

    if first_directory.startswith("dataset_id="):
        return first_directory.replace("dataset_id=", "")

    return first_directory.split(".")[0]


def normalize_responded(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype("string").str.strip().str.lower()

    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}

    unknown_values = normalized.dropna().loc[
        ~normalized.dropna().isin(true_values | false_values)
    ]

    if not unknown_values.empty:
        raise ValueError(
            f"Unexpected values in '{RESPONSE_COLUMN}': "
            f"{unknown_values.unique().tolist()}"
        )

    return normalized.isin(true_values)


# Process Curated Datasets
dataset_results = []

print("Reading curated datasets from S3...\n")

for page in paginator.paginate(
    Bucket=BUCKET,
    Prefix=PREFIX,
):
    for obj in page.get("Contents", []):

        key = obj["Key"]

        if not key.endswith(".parquet"):
            continue

        dataset_id = get_dataset_id(key)

        print(f"Processing dataset {dataset_id}: {key}")

        response = s3.get_object(
            Bucket=BUCKET,
            Key=key,
        )

        df = pd.read_parquet(
            BytesIO(response["Body"].read())
        )

        if RESPONSE_COLUMN not in df.columns:
            print(
                f"Skipping {key}: "
                f"'{RESPONSE_COLUMN}' column not found.\n"
            )
            continue

        responded = normalize_responded(df[RESPONSE_COLUMN])

        dataset_results.append({
            "dataset_id": dataset_id,
            "total_prompts": len(df),
            "responded": int(responded.sum()),
            "not_responded": int((~responded).sum()),
        })


# Aggregate Results by Dataset
if not dataset_results:
    raise RuntimeError(
        f"No usable Parquet datasets with a '{RESPONSE_COLUMN}' "
        f"column were found under s3://{BUCKET}/{PREFIX}"
    )

summary = (
    pd.DataFrame(dataset_results)
    .groupby("dataset_id", as_index=False)
    .agg({
        "total_prompts": "sum",
        "responded": "sum",
        "not_responded": "sum",
    })
)

summary["responded_pct"] = (
    summary["responded"] / summary["total_prompts"] * 100
).round(2)

summary["not_responded_pct"] = (
    summary["not_responded"] / summary["total_prompts"] * 100
).round(2)

summary = summary.sort_values("dataset_id").reset_index(drop=True)


# Display Results
print("\n" + "=" * 65)
print("CURATED DATASET RESPONSE RATES")
print("=" * 65)

print(
    summary[
        [
            "dataset_id",
            "responded_pct",
            "not_responded_pct",
        ]
    ].to_string(index=False)
)


# Save Results Locally
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary[
    [
        "dataset_id",
        "total_prompts",
        "responded",
        "not_responded",
        "responded_pct",
        "not_responded_pct",
    ]
].to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 65)
print("OUTPUT")
print("=" * 65)

print(f"Saved locally: {OUTPUT_PATH}")
print(f"Datasets processed: {len(summary)}")
