
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import boto3
import polars as pl
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = PROJECT_ROOT / "metadata" / "catalog.parquet"
STORAGE_CONFIG_PATH = PROJECT_ROOT / "config" / "storage.yaml"
OUTPUT_PATH = PROJECT_ROOT / "metadata" / "column_inventory.json"


def normalize_id(value):
    value = str(value).strip()
    return value.zfill(4) if value.isdigit() else value


def normalize_column_name(name):
    """Apply minimal normalization for exact-name matching."""
    return name.strip().lower()


def main():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalogue not found: {CATALOG_PATH}")

    config = yaml.safe_load(
        STORAGE_CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}

    aws = config["aws"]
    prefixes = config["s3"]["prefixes"]

    bucket = aws["bucket"]
    raw_prefix = prefixes["raw"].strip("/")

    catalog = pl.read_parquet(CATALOG_PATH)

    id_column = next(
        (
            name for name in catalog.columns
            if name.lower() in {"dataset_id", "id"}
        ),
        None,
    )

    if id_column is None:
        raise ValueError(
            f"No dataset ID column found in catalogue: {catalog.columns}"
        )

    dataset_ids = sorted({
        normalize_id(value)
        for value in catalog[id_column].to_list()
        if value is not None
    })

    s3 = boto3.Session(
        profile_name=aws["profile"],
        region_name=aws["region"],
    ).client("s3")

    # normalized name -> information accumulated across datasets
    inventory = defaultdict(
        lambda: {
            "name_variants": set(),
            "datasets": {},
            "dtypes": set(),
        }
    )

    scanned = []
    missing = []
    errors = []

    for dataset_id in dataset_ids:
        key = f"{raw_prefix}/{dataset_id}.parquet"

        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            parquet_bytes = response["Body"].read()

            # Read the S3 object in memory; do not write dataset files locally.
            df = pl.read_parquet(BytesIO(parquet_bytes))

            for column_name, dtype in df.schema.items():
                normalized = normalize_column_name(column_name)
                dtype_name = str(dtype)

                item = inventory[normalized]
                item["name_variants"].add(column_name)
                item["dtypes"].add(dtype_name)
                item["datasets"][dataset_id] = {
                    "column_name": column_name,
                    "dtype": dtype_name,
                }

            scanned.append(dataset_id)
            print(
                f"Scanned {dataset_id}: "
                f"{df.height} rows, {df.width} columns"
            )

        except Exception as exc:
            error_text = str(exc)
            if (
                "NoSuchKey" in error_text
                or "404" in error_text
                or "Not Found" in error_text
            ):
                missing.append({
                    "dataset_id": dataset_id,
                    "s3_key": key,
                })
            else:
                errors.append({
                    "dataset_id": dataset_id,
                    "s3_key": key,
                    "error": error_text,
                })

            print(f"FAILED {dataset_id}: {error_text}")

    complete = len(scanned) == len(dataset_ids)

    all_variables = {}

    for normalized, item in sorted(inventory.items()):
        dataset_count = len(item["datasets"])

        all_variables[normalized] = {
            "name_variants": sorted(item["name_variants"]),
            "dataset_count": dataset_count,
            "datasets": dict(sorted(item["datasets"].items())),
            "dtypes": sorted(item["dtypes"]),
            "shared_across_scanned_datasets": (
                dataset_count == len(scanned)
            ),
        }

    shared_variables = {
        name: details
        for name, details in all_variables.items()
        if details["shared_across_scanned_datasets"]
    }

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "raw_prefix": f"{raw_prefix}/",
        "matching": {
            "method": "lowercase_and_strip_whitespace",
            "regex_used": False,
            "nlp_used": False,
        },
        "summary": {
            "catalog_datasets": len(dataset_ids),
            "datasets_scanned": len(scanned),
            "datasets_missing": len(missing),
            "datasets_with_errors": len(errors),
            "unique_normalized_variables": len(all_variables),
            "variables_shared_across_scanned_datasets": len(
                shared_variables
            ),
            "complete_inventory": complete,
            "shared_across_all_catalog_datasets_verified": complete,
        },
        "scanned_dataset_ids": scanned,
        "missing_datasets": missing,
        "errors": errors,
        "shared_variables": shared_variables,
        "all_variables": all_variables,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nInventory complete.")
    print(f"Datasets in catalogue: {len(dataset_ids)}")
    print(f"Datasets scanned: {len(scanned)}")
    print(f"Missing objects: {len(missing)}")
    print(f"Other errors: {len(errors)}")
    print(f"Variables found: {len(all_variables)}")
    print(f"Shared across scanned datasets: {len(shared_variables)}")
    print(f"Complete inventory: {complete}")
    print(f"Output: {OUTPUT_PATH}")
    print("No dataset files were saved locally.")


if __name__ == "__main__":
    main()
