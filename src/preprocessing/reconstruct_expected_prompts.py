
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml
import boto3

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_PATH = PROJECT_ROOT / "config" / "prompt_schedules.yaml"
METADATA_PATH = PROJECT_ROOT / "metadata" / "metadata.json"

config = load_config("storage.yaml")
aws = config["aws"]
prefixes = config["s3"]["prefixes"]

s3 = boto3.Session(
    profile_name=aws["profile"],
    region_name=aws["region"],
).client("s3")

BUCKET = aws["bucket"]
METADATA_KEY = f"{prefixes['metadata']}metadata.json"
PROCESSED_PREFIX = prefixes["processed"]


def normalize_id(value):
    value = str(value).strip()
    return value.zfill(4) if value.isdigit() else value


def load_metadata():
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        response = s3.get_object(Bucket=BUCKET, Key=METADATA_KEY)
        metadata = json.loads(
            response["Body"].read().decode("utf-8")
        )

    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object.")

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


def validate_schedule(dataset_id, settings):
    schedule = settings.get("schedule", {})

    if not settings.get("schedule_verified", False):
        return "schedule is not verified"

    if schedule.get("type") != "fixed_times":
        return "only fixed_times schedules are supported"

    start_value = schedule.get("start_date")
    end_value = schedule.get("end_date")
    times = schedule.get("times")

    if not start_value or not end_value:
        return "start_date or end_date is missing"

    if not isinstance(times, list) or not times:
        return "no scheduled prompt times configured"

    if not settings.get("participant_column"):
        return "participant_column is not configured"

    if not settings.get("timestamp_column"):
        return "timestamp_column is not configured"

    response_columns = settings.get("response_columns")
    if not isinstance(response_columns, list) or not response_columns:
        return "response_columns is empty"

    input_file = settings.get("input_file")
    if not input_file:
        return "input_file is not configured"

    try:
        start = date.fromisoformat(start_value)
        end = date.fromisoformat(end_value)

        if end < start:
            return "end_date precedes start_date"

        for item in times:
            time.fromisoformat(item)

        tolerance = float(
            schedule.get(
                "tolerance_minutes",
                settings.get("tolerance_minutes", 30),
            )
        )

        if tolerance < 0:
            return "tolerance_minutes cannot be negative"

        ZoneInfo(settings.get("timezone") or "UTC")

    except (TypeError, ValueError, KeyError):
        return "invalid date, time, tolerance, or timezone configuration"

    return None


def parse_datetime(value, timezone):
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time.min)

    if not isinstance(value, datetime):
        try:
            value = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone)

    return value.astimezone(timezone)


def generate_schedule(schedule, timezone):
    start = date.fromisoformat(schedule["start_date"])
    end = date.fromisoformat(schedule["end_date"])
    times = sorted(
        time.fromisoformat(value)
        for value in schedule["times"]
    )

    current = start

    while current <= end:
        for prompt_time in times:
            yield datetime.combine(
                current,
                prompt_time,
                tzinfo=timezone,
            )

        current += timedelta(days=1)


def reconstruct(dataset_id, settings):
    input_path = (PROJECT_ROOT / settings["input_file"]).resolve()

    if not input_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("input_file must be inside the project directory")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {input_path}"
        )

    frame = pl.read_parquet(input_path)

    participant_col = settings["participant_column"]
    timestamp_col = settings["timestamp_column"]
    response_cols = settings["response_columns"]
    schedule = settings["schedule"]

    required = [participant_col, timestamp_col, *response_cols]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError(f"Configured columns not found: {missing}")

    if frame.height == 0:
        raise ValueError("Processed dataset has no rows")

    timezone = ZoneInfo(settings.get("timezone") or "UTC")
    tolerance_minutes = float(
        schedule.get(
            "tolerance_minutes",
            settings.get("tolerance_minutes", 30),
        )
    )
    tolerance_seconds = tolerance_minutes * 60

    # Preserve source row numbers for auditing.
    frame = frame.with_row_index("__source_row")
    records = frame.to_dicts()

    participants = sorted({
        str(row[participant_col])
        for row in records
        if row[participant_col] is not None
    })

    if not participants:
        raise ValueError("No non-null participant IDs found")

    observed_by_participant = {}
    invalid_timestamp_rows = []

    for row in records:
        participant_value = row[participant_col]

        if participant_value is None:
            invalid_timestamp_rows.append(row)
            continue

        participant = str(participant_value)
        observed_time = parse_datetime(row[timestamp_col], timezone)

        if observed_time is None:
            invalid_timestamp_rows.append(row)
            continue

        observed_by_participant.setdefault(participant, []).append(
            (observed_time, row)
        )

    for participant in observed_by_participant:
        observed_by_participant[participant].sort(
            key=lambda item: item[0]
        )

    expected_times = list(generate_schedule(schedule, timezone))

    expected_rows = []
    matched_source_rows = set()

    for participant in participants:
        observed = observed_by_participant.get(participant, [])

        # Each observed row can match at most one expected prompt.
        used_indices = set()

        for expected_time in expected_times:
            best_index = None
            best_distance = None

            for index, (observed_time, _) in enumerate(observed):
                if index in used_indices:
                    continue

                distance = abs(
                    (observed_time - expected_time).total_seconds()
                )

                if distance > tolerance_seconds:
                    continue

                if best_distance is None or distance < best_distance:
                    best_index = index
                    best_distance = distance

            observed_row = None
            observed_time = None

            if best_index is not None:
                used_indices.add(best_index)
                observed_time, observed_row = observed[best_index]
                matched_source_rows.add(
                    observed_row["__source_row"]
                )

            if observed_row is None:
                status = "missing_prompt"
                minutes_offset = None
            else:
                has_response = any(
                    observed_row[column] is not None
                    for column in response_cols
                )

                status = (
                    "answered"
                    if has_response
                    else "observed_without_response"
                )

                minutes_offset = round(
                    (observed_time - expected_time).total_seconds() / 60,
                    2,
                )

            expected_rows.append({
                "dataset_id": dataset_id,
                "participant_id": participant,
                "expected_timestamp": expected_time,
                "observed_timestamp": observed_time,
                "source_row": (
                    observed_row["__source_row"]
                    if observed_row is not None else None
                ),
                "minutes_from_schedule": minutes_offset,
                "status": status,
            })

    expected_frame = pl.DataFrame(expected_rows)

    unmatched_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "__source_row"
        }
        for row in records
        if row["__source_row"] not in matched_source_rows
    ]

    # Include invalid timestamps in the unmatched audit output too.
    invalid_source_ids = {
        row["__source_row"] for row in invalid_timestamp_rows
    }
    already_unmatched_ids = {
        row["__source_row"]
        for row in records
        if row["__source_row"] not in matched_source_rows
    }
    assert invalid_source_ids.issubset(already_unmatched_ids)

    unmatched_frame = (
        pl.DataFrame(unmatched_rows)
        if unmatched_rows
        else pl.DataFrame({
            "dataset_id": pl.Series([], dtype=pl.String),
            "participant_id": pl.Series([], dtype=pl.String),
            "unmatched_reason": pl.Series([], dtype=pl.String),
        })
    )

    # Add a reason to each unmatched observation.
    if unmatched_rows:
        reasons = []
        for row in records:
            if row["__source_row"] not in matched_source_rows:
                if row["participant_col"] is None:
                    reasons.append("missing_participant_id")
                elif parse_datetime(row[timestamp_col], timezone) is None:
                    reasons.append("invalid_timestamp")
                else:
                    reasons.append("no_expected_prompt_within_tolerance")

        unmatched_frame = unmatched_frame.with_columns(
            pl.Series("unmatched_reason", reasons)
        )

    output_dir = (
        PROJECT_ROOT / "data" / "processed" / "expected_prompts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_path = output_dir / f"{dataset_id}_expected_prompts.parquet"
    unmatched_path = (
        output_dir / f"{dataset_id}_unmatched_observations.parquet"
    )

    expected_frame.write_parquet(expected_path)
    unmatched_frame.write_parquet(unmatched_path)

    for path in (expected_path, unmatched_path):
        s3.upload_file(
            str(path),
            BUCKET,
            f"{PROCESSED_PREFIX}expected_prompts/{path.name}",
        )

    status_counts = {
        row["status"]: row["len"]
        for row in expected_frame.group_by("status").len().to_dicts()
    }

    total_expected = expected_frame.height
    answered = status_counts.get("answered", 0)
    missing_prompts = status_counts.get("missing_prompt", 0)
    no_response = status_counts.get("observed_without_response", 0)

    return {
        "method": "fixed_daily_schedule_nearest_timestamp",
        "schedule_type": schedule["type"],
        "schedule_verified": settings["schedule_verified"],
        "schedule_start_date": schedule["start_date"],
        "schedule_end_date": schedule["end_date"],
        "scheduled_times": schedule["times"],
        "timezone": str(timezone),
        "tolerance_minutes": tolerance_minutes,
        "participants_in_input": len(participants),
        "expected_prompts": total_expected,
        "answered_prompts": answered,
        "missing_prompts": missing_prompts,
        "observed_without_response": no_response,
        "unmatched_observations": unmatched_frame.height,
        "response_rate": (
            round(answered / total_expected, 6)
            if total_expected else None
        ),
        "response_rate_definition": (
            "answered expected prompts / all reconstructed expected prompts"
        ),
        "output_file": str(expected_path.relative_to(PROJECT_ROOT)),
        "unmatched_file": str(unmatched_path.relative_to(PROJECT_ROOT)),
        "limitations": [
            "Only participants present in the processed dataset are included.",
            "Fixed daily schedules only; random schedules are unsupported.",
            "Any non-null configured response field counts as answered.",
            "Unmatched observations require review.",
        ],
    }


def main():
    with SCHEDULE_PATH.open("r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file) or {}

    schedules = yaml_config.get("datasets", {})

    if not schedules:
        raise ValueError(
            "No dataset schedules found in config/prompt_schedules.yaml"
        )

    metadata = load_metadata()
    processed = 0
    skipped = 0
    failures = []

    for raw_id, settings in schedules.items():
        dataset_id = normalize_id(raw_id)

        problem = validate_schedule(dataset_id, settings)

        if problem:
            print(f"SKIPPED {dataset_id}: {problem}")
            skipped += 1
            continue

        try:
            summary = reconstruct(dataset_id, settings)

            record = metadata.setdefault(
                dataset_id,
                {
                    "study_design": {"dataset_id": dataset_id},
                    "features": [],
                    "derived_features": {},
                },
            )

            record.setdefault("derived_features", {})[
                "prompt_reconstruction"
            ] = summary

            print(
                f"{dataset_id}: "
                f"{summary['answered_prompts']}/"
                f"{summary['expected_prompts']} answered; "
                f"{summary['missing_prompts']} missing"
            )

            processed += 1

        except Exception as exc:
            failures.append({
                "dataset_id": dataset_id,
                "error": str(exc),
            })
            print(f"FAILED {dataset_id}: {exc}")

    save_metadata(metadata)

    print("\nSummary")
    print(f"Configured: {len(schedules)}")
    print(f"Processed:  {processed}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {len(failures)}")

    if failures:
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
