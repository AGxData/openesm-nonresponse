from io import BytesIO
import re
import time
import boto3
import openesm
import polars as pl
import requests
from src.config import load_config
from src.metadata.catalog import save_catalogue


ZENODO_API = "https://zenodo.org/api/records"

SPECIAL_DATASETS = {"0026", "0058", "0076"}

DATASET_0026_URL = (
    "https://zenodo.org/records/17347880/files/"
    "0026_fernandez_ts.tsv?download=1"
)


def get_dataset_frame(result) -> pl.DataFrame:
    if isinstance(result, pl.DataFrame):
        return result

    data = getattr(result, "data", None)

    if isinstance(data, pl.DataFrame):
        return data

    raise TypeError(
        "Expected a Polars DataFrame or an object "
        f"with a Polars .data attribute; got {type(result).__name__}"
    )


def find_id_column(catalogue: pl.DataFrame) -> str:
    candidates = {"dataset_id", "datasetid", "id"}

    for column in catalogue.columns:
        if column.lower() in candidates:
            return column

    raise ValueError(
        "Could not identify the dataset ID column. "
        f"Available columns: {catalogue.columns}"
    )


def get_existing_keys(s3, bucket: str, prefix: str) -> set[str]:
    existing = set()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Size"] > 0:
                existing.add(obj["Key"])

    return existing


def find_zenodo_tsv(dataset_id: str) -> str:
    urls = {
        "0026": (
            "https://zenodo.org/records/17347880/files/"
            "0026_fernandez_ts.tsv?download=1"
        ),
        "0058": (
            "https://zenodo.org/records/17348370/files/"
            "0058_gainey_ts.tsv?download=1"
        ),
        "0076": (
            "https://zenodo.org/records/22911022/files/"
            "0076_bayer_ts.tsv?download=1"
        ),
    }

    try:
        return urls[dataset_id]
    except KeyError:
        raise ValueError(
            f"No recovery URL configured for dataset {dataset_id}"
        ) from None



def download_tsv(dataset_id: str) -> bytes:
    url = find_zenodo_tsv(dataset_id)

    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "OpenESM-Research-Ingestion/1.0"},
    )
    response.raise_for_status()

    if not response.content:
        raise ValueError(f"Downloaded an empty TSV for {dataset_id}")

    return response.content


def load_edge_case_dataset(dataset_id: str) -> pl.DataFrame:
    """Recover known parser/download failures from standardized TSV files."""

    content = download_tsv(dataset_id)
    source = BytesIO(content)

    if dataset_id == "0058":
        frame = pl.read_csv(
            source,
            separator="\t",
            infer_schema_length=None,
            schema_overrides={"excited": pl.Float64},
            null_values={"excited": "NA"},
        )

        if "excited" not in frame.columns:
            raise ValueError(
                "Dataset 0058 does not contain the expected 'excited' column"
            )

    elif dataset_id == "0076":
        frame = pl.read_csv(
            source,
            separator="\t",
            infer_schema_length=None,
            quote_char=None,
        )

        if "public_wifi_speed_open" not in frame.columns:
            raise ValueError(
                "Dataset 0076 does not contain the expected "
                "'public_wifi_speed_open' column"
            )

    elif dataset_id == "0026":
        frame = pl.read_csv(
            source,
            separator="\t",
            infer_schema_length=None,
        )

    else:
        raise ValueError(f"No edge-case recovery defined for {dataset_id}")

    if frame.height == 0 or frame.width == 0:
        raise ValueError(f"Recovered dataset {dataset_id} is empty")

    return frame


def load_dataset(dataset_id: str) -> pl.DataFrame:
    """Use OpenESM normally, with explicit fallbacks for known failures."""

    try:
        result = openesm.get_dataset(dataset_id)
        return get_dataset_frame(result)

    except Exception as original_error:
        if dataset_id not in SPECIAL_DATASETS:
            raise

        print(
            f"Standard OpenESM download failed for {dataset_id}: "
            f"{original_error}"
        )
        print(f"Trying the dataset-specific recovery for {dataset_id}...")

        return load_edge_case_dataset(dataset_id)


def upload_dataset(
    s3,
    bucket: str,
    key: str,
    frame: pl.DataFrame,
) -> None:
    buffer = BytesIO()
    frame.write_parquet(buffer)
    buffer.seek(0)

    s3.upload_fileobj(buffer, bucket, key)


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

    # Save the catalogue locally and upload it to S3.
    catalogue, catalogue_path = save_catalogue()
    catalogue_key = f'{prefixes["metadata"]}catalog.parquet'

    s3.upload_file(catalogue_path, bucket, catalogue_key)
    print(f"Catalogue uploaded: s3://{bucket}/{catalogue_key}")

    # Identify dataset IDs.
    id_column = find_id_column(catalogue)

    dataset_ids = (
        catalogue.get_column(id_column)
        .drop_nulls()
        .cast(pl.String)
        .unique(maintain_order=True)
        .to_list()
    )

    # Skip datasets that already have non-empty Parquet objects in S3.
    raw_prefix = prefixes["raw"]
    existing_keys = get_existing_keys(s3, bucket, raw_prefix)

    pending = []

    for dataset_id in dataset_ids:
        dataset_id = str(dataset_id)
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", dataset_id)
        key = f"{raw_prefix}{safe_id}.parquet"

        if key in existing_keys:
            continue

        pending.append((dataset_id, key))

    skipped = len(dataset_ids) - len(pending)

    print(f"Datasets in catalogue: {len(dataset_ids)}")
    print(f"Already ingested: {skipped}")
    print(f"Remaining to ingest: {len(pending)}")

    successful = []
    failed = []

    for dataset_id, key in pending:
        print(f"\nProcessing {dataset_id}...")

        last_error = None

        for attempt in range(1, 4):
            try:
                frame = load_dataset(dataset_id)

                upload_dataset(s3, bucket, key, frame)

                successful.append(dataset_id)

                print(
                    f"Uploaded s3://{bucket}/{key} "
                    f"({frame.height} rows, {frame.width} columns)"
                )

                last_error = None
                break

            except Exception as error:
                last_error = error

                print(
                    f"Attempt {attempt}/3 failed for "
                    f"{dataset_id}: {error}"
                )

                if attempt < 3:
                    time.sleep(2 * attempt)

        if last_error is not None:
            failed.append({
                "dataset_id": dataset_id,
                "error": str(last_error),
            })

    print("\nIngestion summary")
    print("------------------")
    print(f"Catalogue records: {len(dataset_ids)}")
    print(f"Already ingested (skipped): {skipped}")
    print(f"New successful uploads: {len(successful)}")
    print(f"Failed datasets: {len(failed)}")

    if failed:
        print("\nFailures:")
        for item in failed:
            print(f'{item["dataset_id"]}: {item["error"]}')


if __name__ == "__main__":
    main()
