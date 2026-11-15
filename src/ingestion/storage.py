import re
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import yaml


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "storage.yaml"
)


# --------------------------------------------------
# 2. Canonical temporal columns
# --------------------------------------------------

TEMPORAL_COLUMNS = [
    "ema_start_day",
    "ema_start_time",
    "ema_end_day",
    "ema_end_time",
    "eod_start_day",
    "eod_start_time",
    "issued_timestamp",
    "answer_timestamp",
    "created_at",
    "created_date",
    "esm_created_at",
    "esm_ended_at",
    "beep_start_time",
    "beep_end_time",
    "duration_seconds",
]


# --------------------------------------------------
# 3. Configuration
# --------------------------------------------------

def load_storage_config(config_path=None):
    """
    Load storage configuration from storage.yaml.
    """

    config_path = (
        Path(config_path)
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Storage configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Invalid storage configuration."
        )

    for section in ["aws", "s3"]:
        if section not in config:
            raise ValueError(
                f"Missing '{section}' configuration."
            )

    return config


# --------------------------------------------------
# 4. Dataset naming
# --------------------------------------------------

def _normalize_dataset_id(dataset_id):
    """
    Standardize dataset identifiers to four digits.
    """

    return str(dataset_id).strip().zfill(4)


def _get_author_lastname(first_author):
    """
    Extract the first author's surname.

    Supports formats such as:
        Sicorello
        Sicorello et al.
        Sicorello, A.
        John Smith
        Smith, John
    """

    if not first_author:
        raise ValueError(
            "Dataset metadata does not contain first_author."
        )

    author = str(first_author).strip()

    author = re.sub(
        r"\bet\s+al\.?\b",
        "",
        author,
        flags=re.IGNORECASE,
    ).strip()

    if "," in author:
        lastname = author.split(",")[0]
    else:
        lastname = author.split()[-1]

    lastname = re.sub(
        r"[^a-zA-Z0-9-]",
        "",
        lastname,
    )

    if not lastname:
        raise ValueError(
            f"Could not extract surname from: {first_author}"
        )

    return lastname.lower()


def build_dataset_filename(
    dataset_id,
    first_author,
    dataset_type,
):
    """
    Build a standardized processed or curated filename.
    """

    if dataset_type not in {"processed", "curated"}:
        raise ValueError(
            "dataset_type must be 'processed' or 'curated'."
        )

    dataset_id = _normalize_dataset_id(dataset_id)

    lastname = _get_author_lastname(first_author)

    return (
        f"{dataset_id}_{lastname}_{dataset_type}.parquet"
    )


# --------------------------------------------------
# 5. AWS connection
# --------------------------------------------------

def get_s3_client(config=None):
    """
    Create an S3 client using the configured AWS profile.
    """

    if config is None:
        config = load_storage_config()

    aws_config = config["aws"]

    session_kwargs = {}

    if aws_config.get("profile"):
        session_kwargs["profile_name"] = aws_config["profile"]

    if aws_config.get("region"):
        session_kwargs["region_name"] = aws_config["region"]

    session = boto3.Session(**session_kwargs)

    return session.client("s3")


# --------------------------------------------------
# 6. Parquet serialization and upload
# --------------------------------------------------

def upload_parquet(
    df,
    s3_key,
    config=None,
    s3_client=None,
    compression="snappy",
):
    """
    Serialize a DataFrame to Parquet in memory
    and upload it directly to S3.

    No local files are created.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    if config is None:
        config = load_storage_config()

    if s3_client is None:
        s3_client = get_s3_client(config)

    bucket = config["aws"]["bucket"]

    buffer = BytesIO()

    df.to_parquet(
        buffer,
        index=False,
        engine="pyarrow",
        compression=compression,
    )

    buffer.seek(0)

    size_bytes = buffer.getbuffer().nbytes

    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=buffer.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )

    return {
        "bucket": bucket,
        "key": s3_key,
        "s3_uri": f"s3://{bucket}/{s3_key}",
        "size_bytes": size_bytes,
        "rows": len(df),
        "columns": len(df.columns),
    }

# --------------------------------------------------
# 6. Generic file upload
# --------------------------------------------------

def upload_file(
    file_path,
    s3_key,
    config=None,
    s3_client=None,
    content_type=None,
):
    """
    Upload a local file directly to S3.

    Supports images and other file types.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a file: {file_path}"
        )

    if config is None:
        config = load_storage_config()

    if s3_client is None:
        s3_client = get_s3_client(config)

    bucket = config["aws"]["bucket"]

    if content_type is None:
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
        }

        content_type = content_types.get(
            file_path.suffix.lower(),
            "application/octet-stream",
        )

    s3_client.upload_file(
        str(file_path),
        bucket,
        s3_key,
        ExtraArgs={
            "ContentType": content_type,
        },
    )

    size_bytes = file_path.stat().st_size

    return {
        "bucket": bucket,
        "key": s3_key,
        "s3_uri": f"s3://{bucket}/{s3_key}",
        "size_bytes": size_bytes,
        "local_path": file_path,
    }


# 6.5 Generic file upload

def upload_file(
    file_path,
    s3_key,
    config=None,
    s3_client=None,
    content_type=None,
):
    """
    Upload a local file directly to S3.

    Supports images and other file types.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a file: {file_path}"
        )

    if config is None:
        config = load_storage_config()

    if s3_client is None:
        s3_client = get_s3_client(config)

    bucket = config["aws"]["bucket"]

    if content_type is None:
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
        }

        content_type = content_types.get(
            file_path.suffix.lower(),
            "application/octet-stream",
        )

    s3_client.upload_file(
        str(file_path),
        bucket,
        s3_key,
        ExtraArgs={
            "ContentType": content_type,
        },
    )

    size_bytes = file_path.stat().st_size

    return {
        "bucket": bucket,
        "key": s3_key,
        "s3_uri": f"s3://{bucket}/{s3_key}",
        "size_bytes": size_bytes,
        "local_path": file_path,
    }

# --------------------------------------------------
# 7. Curated dataset construction
# --------------------------------------------------

def build_curated_dataset(
    processed_df,
    source_df,
    participant_col="id",
    day_col="day",
    beep_col="beep",
    temporal_columns=None,
    include_columns=None,
):
    """
    Construct a modeling-oriented curated dataset.

    Includes:
        1. Participant, day, and beep identifiers.
        2. Canonical temporal columns.
        3. Response outcome.
        4. All columns generated after feature engineering.
        5. Explicitly requested additional columns.

    Excludes original questionnaire variables and
    reconstruction diagnostics unless explicitly requested.

    Preserves all rows.
    """

    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError(
            "processed_df must be a pandas DataFrame."
        )

    if not isinstance(source_df, pd.DataFrame):
        raise TypeError(
            "source_df must be a pandas DataFrame."
        )

    identifier_columns = [
        participant_col,
        day_col,
        beep_col,
    ]

    missing_identifiers = [
        column for column in identifier_columns
        if column not in processed_df.columns
    ]

    if missing_identifiers:
        raise ValueError(
            f"Missing identifier columns: {missing_identifiers}"
        )

    if temporal_columns is None:
        temporal_columns = TEMPORAL_COLUMNS

    # Canonical temporal fields that actually exist.
    available_temporal = [
        column for column in temporal_columns
        if column in processed_df.columns
    ]

    # Columns created during feature engineering.
    engineered_columns = [
        column for column in processed_df.columns
        if column not in source_df.columns
    ]

    # Retain the binary response outcome.
    outcome_columns = [
        column for column in ["responded"]
        if column in processed_df.columns
    ]

    additional_columns = (
        include_columns
        if include_columns is not None
        else []
    )

    curated_columns = list(dict.fromkeys(
        identifier_columns
        + available_temporal
        + outcome_columns
        + engineered_columns
        + additional_columns
    ))

    missing_columns = [
        column for column in curated_columns
        if column not in processed_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Requested curated columns not found: {missing_columns}"
        )

    return processed_df[curated_columns].copy()


# --------------------------------------------------
# 8. Save processed and curated datasets
# --------------------------------------------------

def save_dataset(
    processed_df,
    source_df,
    inspection,
    participant_col="id",
    day_col="day",
    beep_col="beep",
    config_path=None,
    include_columns=None,
    compression="snappy",
):
    """
    Save processed and curated datasets directly to S3.

    Processed:
        Complete feature-engineered dataset.

    Curated:
        Identifiers, standardized temporal fields,
        response outcome, and engineered features.

    Dataset ID and author are retrieved from
    inspect_dataset() output.

    No local files are created.
    """

    # --------------------------------------------------
    # Retrieve metadata
    # --------------------------------------------------

    if "metadata" not in inspection:
        raise ValueError(
            "inspection must contain a 'metadata' dictionary."
        )

    metadata = inspection["metadata"]

    dataset_id = metadata.get("dataset_id")
    first_author = metadata.get("first_author")

    if dataset_id is None:
        raise ValueError(
            "Dataset ID not found in inspection metadata."
        )

    if not first_author:
        raise ValueError(
            "First author not found in inspection metadata."
        )

    dataset_id = _normalize_dataset_id(dataset_id)

    # --------------------------------------------------
    # Load configuration and connect to S3
    # --------------------------------------------------

    config = load_storage_config(config_path)

    s3_client = get_s3_client(config)

    bucket = config["aws"]["bucket"]

    prefixes = config["s3"]["prefixes"]

    processed_prefix = (
        prefixes["processed"].strip("/") + "/"
    )

    curated_prefix = (
        prefixes["curated"].strip("/") + "/"
    )

    # --------------------------------------------------
    # Build filenames
    # --------------------------------------------------

    processed_filename = build_dataset_filename(
        dataset_id=dataset_id,
        first_author=first_author,
        dataset_type="processed",
    )

    curated_filename = build_dataset_filename(
        dataset_id=dataset_id,
        first_author=first_author,
        dataset_type="curated",
    )

    processed_key = (
        processed_prefix + processed_filename
    )

    curated_key = (
        curated_prefix + curated_filename
    )

    # --------------------------------------------------
    # Construct curated dataset
    # --------------------------------------------------

    curated_df = build_curated_dataset(
        processed_df=processed_df,
        source_df=source_df,
        participant_col=participant_col,
        day_col=day_col,
        beep_col=beep_col,
        include_columns=include_columns,
    )

    # --------------------------------------------------
    # Upload processed dataset
    # --------------------------------------------------

    processed_result = upload_parquet(
        df=processed_df,
        s3_key=processed_key,
        config=config,
        s3_client=s3_client,
        compression=compression,
    )

    # --------------------------------------------------
    # Upload curated dataset
    # --------------------------------------------------

    curated_result = upload_parquet(
        df=curated_df,
        s3_key=curated_key,
        config=config,
        s3_client=s3_client,
        compression=compression,
    )

    # --------------------------------------------------
    # Return upload information
    # --------------------------------------------------

    results = {
        "dataset_id": dataset_id,
        "first_author": first_author,
        "processed": processed_result,
        "curated": curated_result,
        "curated_columns": curated_df.columns.tolist(),
        "processed_df": processed_df,
        "curated_df": curated_df,
    }

    print("=" * 70)
    print("S3 STORAGE RESULTS")
    print("=" * 70)

    print("\nPROCESSED")
    print(f"Location: {processed_result['s3_uri']}")
    print(f"Rows: {processed_result['rows']:,}")
    print(f"Columns: {processed_result['columns']:,}")
    print(f"Size: {processed_result['size_bytes']:,} bytes")

    print("\nCURATED")
    print(f"Location: {curated_result['s3_uri']}")
    print(f"Rows: {curated_result['rows']:,}")
    print(f"Columns: {curated_result['columns']:,}")
    print(f"Size: {curated_result['size_bytes']:,} bytes")

    print("\nCurated columns:")

    for column in curated_df.columns:
        print(f"  - {column}")

    return results
