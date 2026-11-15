
import re
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MISSING_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
}


IDENTIFIER_TERMS = {
    "id",
    "participant",
    "subject",
    "respondent",
    "user",
}


TEMPORAL_COLUMN_MAPPING = {
    "begin_day_ema": "ema_start_day",
    "begin_time_ema": "ema_start_time",
    "finish_day_ema": "ema_end_day",
    "finish_time_ema": "ema_end_time",
    "begin_day_eod": "eod_start_day",
    "begin_time_eod": "eod_start_time",
    "timestamp_issued": "issued_timestamp",
    "timestamp_answer": "answer_timestamp",
    "created": "created_at",
    "creation_time": "created_at",
    "creation_date": "created_date",
    "created_esm": "esm_created_at",
    "ended_esm": "esm_ended_at",
    "beep_start": "beep_start_time",
    "beep_end": "beep_end_time",
    "duration_in_seconds": "duration_seconds",
}


# These columns describe the participant, schedule, prompt,
# collection process, or administrative metadata.
# They are not survey response variables.
NON_RESPONSE_COLUMNS = {
    # Identifiers and schedule
    "id",
    "participant_id",
    "subject_id",
    "respondent_id",
    "user_id",
    "day",
    "beep",
    "counter",
    "record_id",
    "response_id",
    "event_id",
    "session_id",
    "study_id",

    # Temporal fields
    "ema_start_day",
    "ema_start_time",
    "ema_end_day",
    "ema_end_time",
    "eod_start_day",
    "eod_start_time",
    "timestamp",
    "start_time",
    "end_date",
    "created_at",
    "esm_created_at",
    "created_date",
    "duration_seconds",
    "ema_time",
    "esm_ended_at",
    "finished",
    "beep_start_time",
    "beep_end_time",
    "completed",
    "completion_date",
    "duration",
    "duration_eod",
    "export_timezone",
    "hour_block",
    "notification_time",
    "prompt_duration",
    "recorded_date",
    "response_duration",
    "run_timestamp",
    "study_period",
    "submission_timestamp",
    "survey_duration",
    "time_elapsed",
    "time_prompt",
    "answer_timestamp",
    "issued_timestamp",

    # Administrative and collection metadata
    "status",
    "response",
    "response_status",
    "schedule_status",
    "source",
    "source_file",
    "file_name",
    "filename",
    "batch",
    "batch_id",
    "version",
    "export_id",
    "device",
    "device_id",
    "platform",
    "timezone",
    "language",
}


# Include original temporal names so response-column inference
# also works when called before temporal standardization.
NON_RESPONSE_COLUMNS.update(TEMPORAL_COLUMN_MAPPING.keys())


# ---------------------------------------------------------------------
# Column name handling
# ---------------------------------------------------------------------

def _normalize_column_name(name: str) -> str:
    """Normalize a column name into lowercase snake_case."""

    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)

    return name.strip("_")


def _resolve_duplicate_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Give duplicate and blank column names unique identifiers."""

    renamed_columns = []
    seen_columns = {}

    for position, column in enumerate(df.columns):
        column = str(column).strip()

        if not column:
            column = f"unnamed_column_{position}"

        count = seen_columns.get(column, 0)

        if count:
            column = f"{column}_{count + 1}"

        seen_columns[column] = count + 1
        renamed_columns.append(column)

    df.columns = renamed_columns

    return df


def _apply_temporal_column_mapping(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Rename recognized temporal columns and detect collisions."""

    rename_mapping = {
        column: TEMPORAL_COLUMN_MAPPING[column]
        for column in df.columns
        if column in TEMPORAL_COLUMN_MAPPING
    }

    resulting_columns = [
        rename_mapping.get(column, column)
        for column in df.columns
    ]

    if len(resulting_columns) != len(set(resulting_columns)):
        duplicates = sorted({
            column
            for column in resulting_columns
            if resulting_columns.count(column) > 1
        })

        raise ValueError(
            "Temporal column standardization would create "
            f"duplicate columns: {duplicates}"
        )

    return df.rename(columns=rename_mapping)


def _drop_redundant_response_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Drop a redundant generic response column, if present."""

    response_cols = [
        column
        for column in df.columns
        if _normalize_column_name(column) == "response"
    ]

    if response_cols:
        df = df.drop(columns=response_cols)

    return df


# ---------------------------------------------------------------------
# Response column identification
# ---------------------------------------------------------------------

def get_response_columns(
    df: pd.DataFrame,
    exclude_cols: list | set | tuple | None = None,
) -> list[str]:
    """
    Identify survey response columns.

    Returns columns that are not identified as participant,
    schedule, temporal, or administrative metadata.

    Additional exclusions can be supplied through exclude_cols.

    This is an inference function, not a guarantee that every
    returned column is a valid response variable. Dataset-specific
    exclusions should be supplied when necessary.
    """

    excluded = {
        _normalize_column_name(column)
        for column in NON_RESPONSE_COLUMNS
    }

    if exclude_cols is not None:
        excluded.update(
            _normalize_column_name(column)
            for column in exclude_cols
        )

    return [
        column
        for column in df.columns
        if _normalize_column_name(column) not in excluded
    ]


# ---------------------------------------------------------------------
# Data type standardization
# ---------------------------------------------------------------------

def _is_identifier_column(column_name: str) -> bool:
    """Determine whether a column should be protected as an identifier."""

    terms = set(column_name.split("_"))

    return bool(terms.intersection(IDENTIFIER_TERMS))


def _convert_numeric(series: pd.Series) -> pd.Series:
    """Convert numeric columns to nullable integer or float types."""

    numeric = pd.to_numeric(series, errors="coerce")

    original_nonmissing = series.notna().sum()
    converted_nonmissing = numeric.notna().sum()

    if original_nonmissing == 0:
        return series.astype("string")

    if converted_nonmissing / original_nonmissing < 0.95:
        return series.astype("string")

    observed = numeric.dropna()

    if not observed.empty and (observed % 1 == 0).all():
        return numeric.astype("Int64")

    return numeric.astype("Float64")


def _standardize_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Infer and standardize an individual column."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    if pd.api.types.is_integer_dtype(series):
        return series.astype("Int64")

    if pd.api.types.is_float_dtype(series):
        return series.astype("Float64")

    if (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.StringDtype)
    ):
        series = series.astype("string").str.strip()

        series = series.mask(
            series.str.lower().isin(MISSING_VALUES),
            pd.NA,
        )

        if _is_identifier_column(column_name):
            return series

        return _convert_numeric(series)

    return series


# ---------------------------------------------------------------------
# Main standardization function
# ---------------------------------------------------------------------

def standardize_dataset(
    df: pd.DataFrame,
    column_types: dict | None = None,
    normalize_column_names: bool = True,
) -> pd.DataFrame:
    """
    Standardize column names, temporal fields, and data types.

    Processing order:
        1. Normalize column names.
        2. Resolve duplicate column names.
        3. Apply temporal column mapping.
        4. Drop redundant response column.
        5. Infer and standardize data types.
        6. Apply explicit dtype overrides.
    """

    if not isinstance(df, pd.DataFrame):
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        else:
            raise TypeError(
                "Input must be a pandas DataFrame or an object "
                "with a to_pandas() method."
            )

    result = df.copy()

    # 1. Normalize column names.
    if normalize_column_names:
        result.columns = [
            _normalize_column_name(column)
            for column in result.columns
        ]
    
    # 2. Resolve duplicate column names.
    result = _resolve_duplicate_columns(result)

    # 3. Apply temporal column mappings.
    result = _apply_temporal_column_mapping(result)

    # 4. Drop redundant response column.
    result = _drop_redundant_response_column(result)

    # 5. Automatically standardize each column.
    for column in result.columns:
        result[column] = _standardize_series(
            result[column],
            column,
        )

    # 6. Apply explicit dtype overrides.
    if column_types is not None:
        for column, dtype in column_types.items():

            if column not in result.columns:
                raise ValueError(
                    f"Column '{column}' was not found. "
                    f"Available columns: {result.columns.tolist()}"
                )

            if dtype == "datetime64[ns]":
                result[column] = pd.to_datetime(
                    result[column],
                    errors="coerce",
                )
            else:
                result[column] = result[column].astype(dtype)

    return result


# ---------------------------------------------------------------------
# Standalone temporal standardization
# ---------------------------------------------------------------------

def standardize_temporal_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply temporal column mapping and remove the redundant response
    column without performing full data type standardization.
    """

    result = df.copy()

    result.columns = [
        _normalize_column_name(column)
        for column in result.columns
    ]

    result = _resolve_duplicate_columns(result)
    result = _apply_temporal_column_mapping(result)
    result = _drop_redundant_response_column(result)

    return result








# ---------------------------------------------------------------------
# Temporal feature preparation
# ---------------------------------------------------------------------

def prepare_temporal_features(
    df: pd.DataFrame,
    start_col: str,
    end_col: str | None = None,
    timezone: str | None = None,
    utc: bool = True,
    timestamp_col: str = "timestamp_analysis",
    date_col: str = "analysis_date",
    minute_col: str = "minute_of_day",
    hour_col: str = "hour_of_day",
    duration_col: str = "duration_minutes",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Parse timestamps and generate temporal analysis features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    start_col : str
        Column containing the start timestamp.

    end_col : str, optional
        Column containing the end timestamp.
        If None, duration features are skipped.

    timezone : str, optional
        IANA timezone identifier for local-time conversion.
        Example: 'America/New_York'.

    utc : bool, default=True
        Whether to parse timestamps as UTC.
        Recommended for ISO 8601 timestamps containing timezone
        information, such as '2016-11-09T16:57:35Z'.

    timestamp_col : str
        Output column containing the parsed analysis timestamp.

    date_col : str
        Output column containing the calendar date.

    minute_col : str
        Output column containing minutes elapsed since midnight.

    hour_col : str
        Output column containing the hour of day.

    duration_col : str
        Output column containing response duration in minutes.

    overwrite : bool, default=False
        Whether to allow existing output columns to be overwritten.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataset with temporal features.

    Notes
    -----
    Original timestamp columns are preserved.

    When timezone is supplied, calendar dates and time-of-day
    features are calculated using that timezone.

    Duration is calculated using the parsed start and end
    timestamps, independently of the analysis timezone.
    """

    if not isinstance(df, pd.DataFrame):
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        else:
            raise TypeError(
                "Input must be a pandas DataFrame or an object "
                "with a to_pandas() method."
            )

    # -------------------------------------------------------------
    # Validate input columns
    # -------------------------------------------------------------

    if start_col not in df.columns:
        raise ValueError(
            f"Start timestamp column '{start_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    if end_col is not None and end_col not in df.columns:
        raise ValueError(
            f"End timestamp column '{end_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # -------------------------------------------------------------
    # Validate output columns
    # -------------------------------------------------------------

    output_columns = {
        timestamp_col,
        date_col,
        minute_col,
        hour_col,
    }

    if end_col is not None:
        output_columns.add(duration_col)

    existing_columns = output_columns.intersection(df.columns)

    if existing_columns and not overwrite:
        raise ValueError(
            "The following output columns already exist: "
            f"{sorted(existing_columns)}. "
            "Set overwrite=True to replace them."
        )

    result = df.copy()

    # -------------------------------------------------------------
    # Parse start timestamp
    # -------------------------------------------------------------

    start_timestamp = pd.to_datetime(
        result[start_col],
        errors="coerce",
        utc=utc,
    )

    # -------------------------------------------------------------
    # Parse end timestamp, if available
    # -------------------------------------------------------------

    if end_col is not None:

        end_timestamp = pd.to_datetime(
            result[end_col],
            errors="coerce",
            utc=utc,
        )

        result[duration_col] = (
            end_timestamp - start_timestamp
        ).dt.total_seconds() / 60

    # -------------------------------------------------------------
    # Apply analysis timezone
    # -------------------------------------------------------------

    analysis_timestamp = start_timestamp.copy()

    if timezone is not None:

        if analysis_timestamp.dt.tz is None:
            analysis_timestamp = (
                analysis_timestamp.dt.tz_localize(timezone)
            )
        else:
            analysis_timestamp = (
                analysis_timestamp.dt.tz_convert(timezone)
            )

    result[timestamp_col] = analysis_timestamp

    # -------------------------------------------------------------
    # Generate calendar date
    # -------------------------------------------------------------

    result[date_col] = (
        result[timestamp_col].dt.date
    )

    # -------------------------------------------------------------
    # Generate time-of-day features
    # -------------------------------------------------------------

    result[minute_col] = (
        result[timestamp_col].dt.hour * 60
        + result[timestamp_col].dt.minute
    ).astype("Int64")

    result[hour_col] = (
        result[timestamp_col].dt.hour
    ).astype("Int64")

    return result
