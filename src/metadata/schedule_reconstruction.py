
from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
import re
from typing import Any

import boto3
import polars as pl

from src.config import load_config


CATALOG_FILENAME = "catalog.parquet"
OUTPUT_FILENAME = "reconstructed_schedules.parquet"

ID_ALIASES = {
    "dataset_id",
    "datasetid",
    "id",
}

DIMENSION_ALIASES = {
    "n_participants": {
        "n_participants",
        "num_participants",
        "number_of_participants",
        "participants",
    },
    "n_days": {
        "n_days",
        "num_days",
        "number_of_days",
        "study_days",
        "days",
    },
    "n_beeps_per_day": {
        "n_beeps_per_day",
        "beeps_per_day",
        "n_beeps",
        "num_beeps_per_day",
        "prompts_per_day",
    },
}


def normalize_id(value: Any) -> str | None:
    """Normalize IDs for matching catalog records to S3 filenames."""

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    # Handle numeric IDs that were converted to strings such as "26.0".
    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".")[0]

    # OpenESM dataset IDs commonly use four digits, e.g. 0026.
    if value.isdigit() and len(value) < 4:
        value = value.zfill(4)

    return value


def find_column(
    columns: list[str],
    aliases: set[str],
) -> str | None:
    """Find a column by normalized name, ignoring case and punctuation."""

    def normalize_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    normalized_aliases = {
        normalize_name(alias) for alias in aliases
    }

    for column in columns:
        if normalize_name(column) in normalized_aliases:
            return column

    return None


def parse_dimension(value: Any) -> int | None:
    """Parse a nonnegative integer dimension without silently rounding."""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not number.is_integer() or number < 0:
        return None

    return int(number)


def read_s3_parquet(s3, bucket: str, key: str) -> pl.DataFrame:
    """Read a Parquet object without modifying its S3 source."""

    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read()

    if not content:
        raise ValueError("S3 object is empty")

    return pl.read_parquet(BytesIO(content))


def get_catalog_dimensions(
    catalogue: pl.DataFrame,
) -> tuple[dict[str, dict], list[dict]]:
    """Extract dimensions and detect missing or ambiguous catalog metadata."""

    id_column = find_column(catalogue.columns, ID_ALIASES)

    if id_column is None:
        raise ValueError(
            "Cannot identify the dataset ID column in the catalog. "
            f"Available columns: {catalogue.columns}"
        )

    dimension_columns = {
        name: find_column(catalogue.columns, aliases)
        for name, aliases in DIMENSION_ALIASES.items()
    }

    lookup: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    errors: list[dict] = []

    for row in catalogue.iter_rows(named=True):
        dataset_id = normalize_id(row.get(id_column))

        if dataset_id is None:
            errors.append({
                "dataset_id": None,
                "error": "Catalog record has no usable dataset ID",
            })
            continue

        if dataset_id in lookup:
            duplicate_ids.add(dataset_id)
            continue

        dimensions = {}

        for dimension, column in dimension_columns.items():
            dimensions[dimension] = (
                parse_dimension(row.get(column))
                if column is not None
                else None
            )

        lookup[dataset_id] = dimensions

    for dataset_id in sorted(duplicate_ids):
        lookup.pop(dataset_id, None)
        errors.append({
            "dataset_id": dataset_id,
            "error": "Duplicate normalized dataset IDs in catalog",
        })

    return lookup, errors


def calculate_summary(
    dataset_id: str,
    source_key: str,
    actual_rows: int | None,
    dimensions: dict | None,
    status: str,
    error: str | None = None,
) -> dict:
    """Calculate row-count metrics when the required information is valid."""

    dimensions = dimensions or {}

    n_participants = dimensions.get("n_participants")
    n_days = dimensions.get("n_days")
    n_beeps_per_day = dimensions.get("n_beeps_per_day")

    dimension_values = [
        n_participants,
        n_days,
        n_beeps_per_day,
    ]

    dimensions_complete = all(
        value is not None for value in dimension_values
    )

    theoretical_rows = (
        n_participants * n_days * n_beeps_per_day
        if dimensions_complete
        else None
    )

    difference = (
        theoretical_rows - actual_rows
        if theoretical_rows is not None and actual_rows is not None
        else None
    )

    difference_pct = (
        difference / theoretical_rows * 100
        if theoretical_rows not in (None, 0)
        and actual_rows is not None
        else None
    )

    if status == "ok" and not dimensions_complete:
        status = "missing_or_invalid_metadata"
    elif status == "ok" and theoretical_rows == 0:
        status = "zero_theoretical_rows"

    return {
        "dataset_id": dataset_id,
        "source_key": source_key,
        "n_participants": n_participants,
        "n_days": n_days,
        "n_beeps_per_day": n_beeps_per_day,
        "theoretical_rows": theoretical_rows,
        "actual_rows": actual_rows,
        "difference": difference,
        "difference_pct": difference_pct,
        "status": status,
        "error": error,
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

    metadata_prefix = prefixes["metadata"]
    raw_prefix = prefixes["raw"]
    output_prefix = prefixes["processed"]

    catalog_key = f"{metadata_prefix}{CATALOG_FILENAME}"

    # Load catalog metadata.
    catalogue = read_s3_parquet(s3, bucket, catalog_key)
    dimensions_by_id, catalog_errors = get_catalog_dimensions(catalogue)

    results = []
    seen_ids = set()

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=raw_prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.lower().endswith(".parquet"):
                continue

            filename = PurePosixPath(key).name
            dataset_id = normalize_id(
                PurePosixPath(filename).stem
            )

            if dataset_id is None:
                continue

            if dataset_id in seen_ids:
                results.append(calculate_summary(
                    dataset_id=dataset_id,
                    source_key=key,
                    actual_rows=None,
                    dimensions=None,
                    status="duplicate_raw_dataset_id",
                    error="Multiple raw files normalize to the same ID",
                ))
                continue

            seen_ids.add(dataset_id)
            dimensions = dimensions_by_id.get(dataset_id)

            if dimensions is None:
                results.append(calculate_summary(
                    dataset_id=dataset_id,
                    source_key=key,
                    actual_rows=None,
                    dimensions=None,
                    status="missing_or_ambiguous_catalog_record",
                    error="No unique catalog record matched this dataset",
                ))
                continue

            try:
                frame = read_s3_parquet(s3, bucket, key)

                results.append(calculate_summary(
                    dataset_id=dataset_id,
                    source_key=key,
                    actual_rows=frame.height,
                    dimensions=dimensions,
                    status="ok",
                ))

            except Exception as exc:
                results.append(calculate_summary(
                    dataset_id=dataset_id,
                    source_key=key,
                    actual_rows=None,
                    dimensions=dimensions,
                    status="read_error",
                    error=f"{type(exc).__name__}: {exc}",
                ))

    # Include catalog records without a corresponding raw Parquet file.
    for dataset_id, dimensions in dimensions_by_id.items():
        if dataset_id not in seen_ids:
            results.append(calculate_summary(
                dataset_id=dataset_id,
                source_key="",
                actual_rows=None,
                dimensions=dimensions,
                status="missing_raw_file",
                error="No matching Parquet file found under raw/",
            ))

    # Keep the audit trail of catalog problems in the same output.
    for item in catalog_errors:
        results.append(calculate_summary(
            dataset_id=item["dataset_id"] or "",
            source_key="",
            actual_rows=None,
            dimensions=None,
            status="catalog_error",
            error=item["error"],
        ))

    schema = {
        "dataset_id": pl.String,
        "source_key": pl.String,
        "n_participants": pl.Int64,
        "n_days": pl.Int64,
        "n_beeps_per_day": pl.Int64,
        "theoretical_rows": pl.Int64,
        "actual_rows": pl.Int64,
        "difference": pl.Int64,
        "difference_pct": pl.Float64,
        "status": pl.String,
        "error": pl.String,
    }

    summary = pl.DataFrame(results, schema=schema)

    output_buffer = BytesIO()
    summary.write_parquet(output_buffer, compression="zstd")
    output_buffer.seek(0)

    output_key = f"{output_prefix}{OUTPUT_FILENAME}"

    s3.upload_fileobj(
        output_buffer,
        bucket,
        output_key,
    )

    print(
        f"Summary written to s3://{bucket}/{output_key}"
    )


if __name__ == "__main__":
    main()
