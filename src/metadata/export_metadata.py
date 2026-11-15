
import json
import math
from datetime import date, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import boto3
import openesm
import polars as pl

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "metadata" / "metadata.json"


def json_safe(value):
    if value is None:
        return None

    if isinstance(value, pl.DataFrame):
        return json_safe(value.to_dicts())

    if isinstance(value, pl.Series):
        return json_safe(value.to_list())

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if hasattr(value, "item"):
        return json_safe(value.item())

    return value


def normalize_dataset_id(value) -> str:
    dataset_id = str(value).strip()

    if dataset_id.isdigit():
        return dataset_id.zfill(4)

    return dataset_id


def find_id_column(catalogue: pl.DataFrame) -> str:
    candidates = {"dataset_id", "datasetid", "id"}

    for column in catalogue.columns:
        if column.lower() in candidates:
            return column

    raise ValueError(
        f"Could not identify dataset ID column: {catalogue.columns}"
    )


def load_existing_metadata() -> dict:
    if not OUTPUT_PATH.exists():
        return {}

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Expected a JSON object indexed by dataset ID: {OUTPUT_PATH}"
        )

    return metadata


def get_s3_frame(s3, bucket: str, key: str) -> pl.DataFrame:
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read()

    return pl.read_parquet(BytesIO(content))


def profile_dataset(frame: pl.DataFrame) -> dict:
    columns = []

    for name in frame.columns:
        series = frame.get_column(name)
        null_count = series.null_count()

        columns.append({
            "name": name,
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": (
                round(null_count / frame.height * 100, 4)
                if frame.height
                else None
            ),
            "unique_count": series.n_unique(),
        })

    return {
        "n_rows": frame.height,
        "n_columns": frame.width,
        "columns": columns,
    }


def main() -> None:
    config = load_config("storage.yaml")

    aws = config["aws"]
    prefixes = config["s3"]["prefixes"]

    session = boto3.Session(
        profile_name=aws["profile"],
        region_name=aws["region"],
    )
    s3 = session.client("s3")
    bucket = aws["bucket"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    catalogue = openesm.list_datasets()
    id_column = find_id_column(catalogue)

    existing_metadata = load_existing_metadata()
    metadata = {}
    failed = []

    print(f"Catalogue records: {catalogue.height}")

    for record in catalogue.to_dicts():
        dataset_id = normalize_dataset_id(record[id_column])
        print(f"Processing {dataset_id}...")

        try:
            safe_id = dataset_id

            raw_key = (
                f'{prefixes["raw"]}{safe_id}.parquet'
            )

            frame = get_s3_frame(s3, bucket, raw_key)

            source_metadata = dict(record)
            source_metadata.pop(id_column, None)
            original_features = source_metadata.pop("features", None)

            previous = existing_metadata.get(dataset_id, {})
            previous_design = previous.get("study_design", {})
            previous_features = previous.get("features", {})
            previous_derived = previous.get("derived_features", {})

            # Preserve existing derived work while refreshing the base profile.
            derived_features = {
                **previous_derived,
                **profile_dataset(frame),
            }

            metadata[dataset_id] = {
                "study_design": {
                    **json_safe(source_metadata),
                    "dataset_id": dataset_id,
                },
                "features": (
                    json_safe(original_features)
                    if original_features is not None
                    else json_safe(previous_features)
                ),
                "derived_features": json_safe(derived_features),
            }

            # Preserve any additional study-design fields that are not
            # currently present in the catalogue.
            for key, value in previous_design.items():
                metadata[dataset_id]["study_design"].setdefault(
                    key, value
                )

            print(
                f"Processed {dataset_id}: "
                f"{frame.height} rows, {frame.width} columns"
            )

        except Exception as error:
            # Keep a previous record if the current refresh fails.
            if dataset_id in existing_metadata:
                metadata[dataset_id] = existing_metadata[dataset_id]

            failed.append({
                "dataset_id": dataset_id,
                "error": str(error),
            })

            print(f"FAILED {dataset_id}: {error}")

    # Serialize once so local and cloud copies have identical contents.
    content = json.dumps(
        json_safe(metadata),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"\nSaved locally: {OUTPUT_PATH}")

    metadata_key = f'{prefixes["metadata"]}metadata.json'

    s3.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=content.encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )

    print(f"Uploaded: s3://{bucket}/{metadata_key}")

    print("\nMetadata export summary")
    print("-----------------------")
    print(f"Catalogue records: {catalogue.height}")
    print(f"Records in JSON: {len(metadata)}")
    print(f"Failures: {len(failed)}")

    if failed:
        print("\nFailures:")
        for item in failed:
            print(f'{item["dataset_id"]}: {item["error"]}')


if __name__ == "__main__":
    main()
