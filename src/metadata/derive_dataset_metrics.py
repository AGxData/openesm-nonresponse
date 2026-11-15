import json
import math
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
import boto3
import numpy as np
import polars as pl
from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "metadata" / "metadata.json"

config = load_config("storage.yaml")
aws = config["aws"]
prefixes = config["s3"]["prefixes"]

s3 = boto3.Session(
    profile_name=aws["profile"],
    region_name=aws["region"],
).client("s3")

BUCKET = aws["bucket"]
RAW_PREFIX = prefixes["raw"]
METADATA_KEY = f"{prefixes['metadata']}metadata.json"

# Candidate column names. Actual columns vary by dataset.
ID_NAMES = {
    "participant_id", "participant", "subject_id",
    "subject", "person_id", "user_id", "id"
}
TIME_NAMES = {
    "timestamp", "datetime", "date_time", "time_stamp",
    "response_time", "created_at", "start_time"
}
DAY_NAMES = {"study_day", "day", "day_number", "esm_day"}
BEEP_NAMES = {
    "beep", "beep_number", "prompt", "prompt_number",
    "signal", "signal_number", "occasion"
}


def normalize_id(value):
    value = str(value).strip()
    return value.zfill(4) if value.isdigit() else value


def load_metadata():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=METADATA_KEY)
        metadata = json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        if METADATA_PATH.exists():
            metadata = json.loads(
                METADATA_PATH.read_text(encoding="utf-8")
            )
        else:
            metadata = {}

    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object.")

    return metadata


def save_metadata(metadata):
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    METADATA_PATH.write_text(content + "\n", encoding="utf-8")

    s3.put_object(
        Bucket=BUCKET,
        Key=METADATA_KEY,
        Body=content.encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def find_column(columns, candidates):
    lookup = {column.lower(): column for column in columns}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    return None


def parse_timestamps(series):
    """Parse a column into datetimes when possible."""
    try:
        parsed = series.cast(pl.String).str.to_datetime(
            strict=False, time_unit="us"
        )
        if parsed.drop_nulls().len() > 0:
            return parsed
    except Exception:
        pass

    try:
        parsed = series.cast(pl.String).str.to_date(strict=False)
        if parsed.drop_nulls().len() > 0:
            return parsed
    except Exception:
        pass

    return None


def autocorrelation_lag1(values):
    """Pearson lag-1 autocorrelation for one ordered numeric series."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        return None

    left = values[:-1]
    right = values[1:]

    if np.std(left) == 0 or np.std(right) == 0:
        return None

    result = np.corrcoef(left, right)[0, 1]
    return float(result) if np.isfinite(result) else None


def profile_dataset(dataset_id, frame):
    columns = frame.columns
    id_col = find_column(columns, ID_NAMES)
    time_col = find_column(columns, TIME_NAMES)
    day_col = find_column(columns, DAY_NAMES)
    beep_col = find_column(columns, BEEP_NAMES)

    column_profiles = []

    for name, dtype in frame.schema.items():
        series = frame[name]
        column_profiles.append({
            "name": name,
            "dtype": str(dtype),
            "null_count": series.null_count(),
            "null_pct": round(
                series.null_count() / frame.height * 100, 3
            ) if frame.height else None,
            "unique_count": series.n_unique(),
        })

    result = {
        "n_rows": frame.height,
        "n_columns": frame.width,
        "columns": column_profiles,
        "participant_id_column": id_col,
        "timestamp_column": time_col,
        "study_day_column": day_col,
        "beep_column": beep_col,
        "participant_count": None,
        "observed_start": None,
        "observed_end": None,
        "observed_duration_days": None,
        "observed_days": None,
        "observed_prompts_per_day": None,
        "response_rate": {
            "value": None,
            "status": "requires_schedule_or_response_indicator",
        },
        "autocorrelation_lag1": {},
        "study_design": {
            "planned_beeps_per_day": {
                "value": None,
                "status": "requires_design_verification",
            },
            "planned_study_duration_days": {
                "value": None,
                "status": "requires_design_verification",
            },
            "scheduled_prompt_times": {
                "value": None,
                "status": "requires_design_verification",
            },
        },
        "notes": [],
    }

    if frame.height == 0:
        result["notes"].append("Dataset contains no rows.")
        return result

    if id_col:
        result["participant_count"] = frame[id_col].n_unique()
    else:
        result["notes"].append(
            "Participant column not detected; verify column mapping."
        )

    parsed_time = parse_timestamps(frame[time_col]) if time_col else None

    if parsed_time is not None:
        frame = frame.with_columns(parsed_time.alias("__parsed_time"))
        valid_times = frame["__parsed_time"].drop_nulls()

        if valid_times.len():
            start = valid_times.min()
            end = valid_times.max()

            if isinstance(start, (datetime, date)):
                result["observed_start"] = start.isoformat()
                result["observed_end"] = end.isoformat()

                if isinstance(start, datetime) and isinstance(end, datetime):
                    duration = (end - start).total_seconds() / 86400
                else:
                    duration = (end - start).days

                result["observed_duration_days"] = round(duration, 3)

                result["observed_days"] = (
                    valid_times.dt.date().n_unique()
                    if isinstance(start, datetime)
                    else valid_times.n_unique()
                )

                if result["observed_days"]:
                    result["observed_prompts_per_day"] = round(
                        frame.height / result["observed_days"], 3
                    )

    elif time_col:
        result["notes"].append(
            f"Detected {time_col!r}, but could not parse it as timestamps."
        )
    else:
        result["notes"].append(
            "No timestamp column detected; observed duration unavailable."
        )

    # Compute lag-1 autocorrelation for numeric outcomes, separately
    # within each participant where possible.
    excluded = {id_col, time_col, day_col, beep_col, "__parsed_time"}
    numeric_columns = [
        name for name, dtype in frame.schema.items()
        if name not in excluded and dtype.is_numeric()
    ]

    for name in numeric_columns:
        estimates = []

        if id_col:
            groups = frame.partition_by(id_col, maintain_order=True)
        else:
            groups = [frame]

        for group in groups:
            if "__parsed_time" in group.columns:
                group = group.sort("__parsed_time")

            estimate = autocorrelation_lag1(group[name].to_numpy())

            if estimate is not None:
                estimates.append(estimate)

        result["autocorrelation_lag1"][name] = {
            "median_across_participants": (
                round(float(np.median(estimates)), 4)
                if estimates else None
            ),
            "participants_with_estimate": len(estimates),
            "method": "lag1_pearson_within_participant",
        }

    # This is a row-level completeness proxy, not a true prompt response rate.
    excluded_lower = {str(c).lower() for c in excluded if c}
    outcome_columns = [
        name for name in columns
        if name.lower() not in excluded_lower
    ]

    if outcome_columns:
        row_has_response = pl.any_horizontal([
            pl.col(name).is_not_null() for name in outcome_columns
        ])
        result["row_completeness_proxy"] = round(
            frame.select(row_has_response.mean()).item() * 100, 3
        )
        result["notes"].append(
            "Row completeness is only a proxy; it is not the percentage "
            "of scheduled prompts answered."
        )

    result["notes"].append(
        "Observed duration and prompts per day do not establish the "
        "intended study schedule."
    )

    return result


def main():
    metadata = load_metadata()

    objects = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=RAW_PREFIX,
    )

    keys = []
    while True:
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)

        if not objects.get("IsTruncated"):
            break

        objects = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=RAW_PREFIX,
            ContinuationToken=objects["NextContinuationToken"],
        )

    processed = 0
    failed = []

    for key in sorted(keys):
        dataset_id = normalize_id(Path(key).stem)

        try:
            response = s3.get_object(Bucket=BUCKET, Key=key)
            frame = pl.read_parquet(BytesIO(response["Body"].read()))

            metrics = profile_dataset(dataset_id, frame)

            record = metadata.setdefault(
                dataset_id,
                {
                    "study_design": {"dataset_id": dataset_id},
                    "features": [],
                    "derived_features": {},
                },
            )

            # Preserve existing hand-entered values unless the metric
            # explicitly represents a computed value.
            record["derived_features"] = {
                **record.get("derived_features", {}),
                **metrics,
            }

            print(f"Processed {dataset_id}: {frame.height:,} rows")
            processed += 1

        except Exception as exc:
            failed.append({"dataset_id": dataset_id, "error": str(exc)})
            print(f"FAILED {dataset_id}: {exc}")

    save_metadata(metadata)

    print(f"\nProcessed: {processed}")
    print(f"Failed: {len(failed)}")
    print(f"Metadata records: {len(metadata)}")

    if failed:
        print(json.dumps(failed, indent=2))


if __name__ == "__main__":
    main()
