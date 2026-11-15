import json
from pathlib import Path
import boto3
from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "metadata" / "metadata.json"

config = load_config("storage.yaml")
aws_config = config["aws"]
metadata_prefix = config["s3"]["prefixes"]["metadata"]

s3 = boto3.Session(
    profile_name=aws_config["profile"],
    region_name=aws_config["region"],
).client("s3")

BUCKET = aws_config["bucket"]
S3_KEY = f"{metadata_prefix}metadata.json"

SECTIONS = {"study_design", "features", "derived_features"}


def normalize_id(dataset_id):
    """Normalize numeric dataset IDs, preserving leading zeros."""
    value = str(dataset_id).strip()
    return value.zfill(4) if value.isdigit() else value


def load_metadata(download_if_missing=True):
    """Load the local JSON, optionally downloading it from S3 if missing."""
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    elif download_if_missing:
        response = s3.get_object(Bucket=BUCKET, Key=S3_KEY)
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        _write_local(metadata)
    else:
        metadata = {}

    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object.")

    return metadata


def get_dataset(dataset_id, metadata=None):
    """Retrieve one dataset's metadata by ID."""
    if metadata is None:
        metadata = load_metadata()

    return metadata.get(normalize_id(dataset_id))


def _write_local(metadata):
    """Write JSON locally using UTF-8 and readable formatting."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")


def save_metadata(metadata, upload=True):
    """Save the JSON locally and optionally synchronize it to S3."""
    _write_local(metadata)

    if upload:
        content = json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        s3.put_object(
            Bucket=BUCKET,
            Key=S3_KEY,
            Body=content.encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )


def _merge(existing, incoming):
    """Recursively merge dictionaries; extend lists rather than replace them."""
    if isinstance(existing, dict) and isinstance(incoming, dict):
        result = existing.copy()

        for key, value in incoming.items():
            if key in result:
                result[key] = _merge(result[key], value)
            else:
                result[key] = value

        return result

    if isinstance(existing, list) and isinstance(incoming, list):
        return existing + incoming

    return incoming


def update_dataset(dataset_id, updates, upload=True):
    """
    Add or update fields for one dataset without replacing unrelated metadata.

    Dictionary fields are merged recursively.
    Lists are appended to when both old and new values are lists.
    """
    dataset_id = normalize_id(dataset_id)

    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary.")

    metadata = load_metadata()

    record = metadata.setdefault(
        dataset_id,
        {
            "study_design": {"dataset_id": dataset_id},
            "features": [],
            "derived_features": {},
        },
    )

    metadata[dataset_id] = _merge(record, updates)

    save_metadata(metadata, upload=upload)
    return metadata[dataset_id]


def append_to_list(dataset_id, section, items, upload=True):
    """Append one or more items to a list-valued metadata section."""
    if section not in SECTIONS:
        raise ValueError(f"Invalid metadata section: {section}")

    if not isinstance(items, list):
        items = [items]

    dataset_id = normalize_id(dataset_id)
    metadata = load_metadata()

    record = metadata.setdefault(
        dataset_id,
        {
            "study_design": {"dataset_id": dataset_id},
            "features": [],
            "derived_features": {},
        },
    )

    current = record.setdefault(section, [])

    if not isinstance(current, list):
        raise TypeError(
            f"{dataset_id}.{section} is not a list; refusing to overwrite it."
        )

    current.extend(items)
    save_metadata(metadata, upload=upload)

    return current


def refresh_from_cloud():
    """Replace the local copy with the current JSON stored in S3."""
    response = s3.get_object(Bucket=BUCKET, Key=S3_KEY)
    metadata = json.loads(response["Body"].read().decode("utf-8"))
    save_metadata(metadata, upload=False)
    return metadata
