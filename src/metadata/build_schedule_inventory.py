
from pathlib import Path
import json

import polars as pl
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = PROJECT_ROOT / "metadata" / "catalog.parquet"
METADATA_PATH = PROJECT_ROOT / "metadata" / "metadata.json"
STORAGE_CONFIG_PATH = PROJECT_ROOT / "config" / "storage.yaml"
OUTPUT_PATH = PROJECT_ROOT / "config" / "prompt_schedules.yaml"


def normalize_id(value):
    """Normalize numeric dataset IDs to four digits."""
    value = str(value).strip()
    return value.zfill(4) if value.isdigit() else value


def find_column(columns, *candidates):
    """Find a column name without case sensitivity."""
    lookup = {column.lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def get_value(record, *keys):
    """Return the first value that is not None."""
    if not isinstance(record, dict):
        return None

    for key in keys:
        value = record.get(key)
        if value is not None:
            return value

    return None


def clean(value):
    """Convert metadata values to YAML-safe Python values."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "to_list"):
        return [clean(item) for item in value.to_list()]

    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        return clean(value.item())

    return str(value)


def load_metadata():
    """Load existing study metadata without modifying it."""
    if not METADATA_PATH.exists():
        return {}

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Expected a JSON object in {METADATA_PATH}"
        )

    return metadata


def load_storage_config():
    """Load S3 settings from the project's storage configuration."""
    if not STORAGE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Storage configuration not found: {STORAGE_CONFIG_PATH}"
        )

    config = yaml.safe_load(
        STORAGE_CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}

    aws = config.get("aws", {})
    s3_config = config.get("s3", {})
    prefixes = s3_config.get("prefixes", {})

    bucket = aws.get("bucket")
    raw_prefix = prefixes.get("raw")
    processed_prefix = prefixes.get("processed")

    if not bucket:
        raise ValueError("Missing aws.bucket in storage.yaml.")

    if raw_prefix is None:
        raise ValueError("Missing s3.prefixes.raw in storage.yaml.")

    if processed_prefix is None:
        raise ValueError(
            "Missing s3.prefixes.processed in storage.yaml."
        )

    return {
        "bucket": bucket,
        "raw_prefix": raw_prefix,
        "processed_prefix": processed_prefix,
    }


def main():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalogue not found: {CATALOG_PATH}"
        )

    catalog = pl.read_parquet(CATALOG_PATH)
    metadata = load_metadata()
    storage = load_storage_config()

    id_column = find_column(
        catalog.columns, "dataset_id", "id"
    )

    if id_column is None:
        raise ValueError(
            f"No dataset ID column found. Columns: {catalog.columns}"
        )

    datasets = {}

    for row in catalog.to_dicts():
        raw_id = row.get(id_column)

        if raw_id is None:
            raise ValueError(
                f"Catalogue row has no dataset ID: {row}"
            )

        dataset_id = normalize_id(raw_id)

        if dataset_id in datasets:
            raise ValueError(
                f"Duplicate dataset ID in catalogue: {dataset_id}"
            )

        existing = metadata.get(dataset_id, {})
        study = existing.get("study_design", {})

        # Prefer study metadata when it is not None.
        # Otherwise, fall back to the catalogue.
        def value(*keys):
            study_value = get_value(study, *keys)

            if study_value is not None:
                return study_value

            return get_value(row, *keys)

        design_hints = {
            "sampling_scheme": clean(value("sampling_scheme")),
            "n_days": clean(value("n_days")),
            "n_beeps_per_day": clean(value("n_beeps_per_day")),
            "n_time_points": clean(value("n_time_points")),
            "participants": clean(value("participants")),
            "raw_time_stamp": clean(value("raw_time_stamp")),
        }

        design_hints = {
            key: val
            for key, val in design_hints.items()
            if val is not None
        }

        datasets[dataset_id] = {
            # Dataset contents remain in S3; no local data files required.
            "input_source": "s3",
            "input_object": (
                f"{storage['raw_prefix'].rstrip('/')}/"
                f"{dataset_id}.parquet"
            ),
            "output_source": "s3",
            "output_prefix": storage["processed_prefix"],
            "participant_column": None,
            "timestamp_column": None,
            "response_columns": [],
            "timezone": None,
            "schedule": {
                "type": "unknown",
                "start_date": None,
                "end_date": None,
                "times": [],
                "tolerance_minutes": 30,
            },
            "schedule_verified": False,
            "design_hints": design_hints,
            "verification_status": "needs_review",
            "verification": {
                "source": None,
                "verified_date": None,
                "reviewer": None,
            },
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "defaults": {
            "timezone": None,
            "tolerance_minutes": 30,
            "schedule_verified": False,
        },
        "storage": {
            "provider": "s3",
            "bucket": storage["bucket"],
            "raw_prefix": storage["raw_prefix"],
            "processed_prefix": storage["processed_prefix"],
        },
        "datasets": datasets,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            output,
            file,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )

    print(f"Catalogue datasets: {catalog.height}")
    print(f"Schedule entries created: {len(datasets)}")
    print(f"Verified schedules: 0")
    print(f"Schedules needing review: {len(datasets)}")
    print(f"S3 bucket: {storage['bucket']}")
    print(f"Output: {OUTPUT_PATH}")
    print("No dataset contents were downloaded or modified.")
    print("All schedules remain unverified until reviewed.")


if __name__ == "__main__":
    main()
