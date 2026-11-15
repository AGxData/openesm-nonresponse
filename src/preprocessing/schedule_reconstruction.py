
import numpy as np
import pandas as pd

from preprocessing.standardization import get_response_columns


# ---------------------------------------------------------------------
# Expected schedule utilities
# ---------------------------------------------------------------------

def _get_expected_values(
    df,
    column,
    expected_values=None,
):
    """
    Determine expected values for a schedule dimension.

    If explicit values are provided, use those.

    Otherwise:
        - Infer observed values.
        - For integer-valued dimensions, infer the continuous
          range between the minimum and maximum.
        - For other dimensions, return sorted observed values.

    Explicit values are recommended when the schedule is known,
    particularly when observed values may not be contiguous.
    """

    if expected_values is not None:
        return list(expected_values)

    observed = df[column].dropna().unique()

    if len(observed) == 0:
        return []

    numeric = pd.to_numeric(
        pd.Series(observed),
        errors="coerce",
    )

    if numeric.notna().all() and (numeric % 1 == 0).all():
        return list(
            range(
                int(numeric.min()),
                int(numeric.max()) + 1,
            )
        )

    try:
        return sorted(observed.tolist())

    except TypeError:
        # Mixed incomparable types: preserve first-observed order.
        return list(observed)


# ---------------------------------------------------------------------
# Schedule validation
# ---------------------------------------------------------------------

def validate_schedule(
    df,
    participant_col="id",
    day_col="day",
    beep_col="beep",
    expected_participants=None,
    expected_days=None,
    expected_beeps=None,
):
    """
    Validate participant-day-beep schedule integrity.

    Checks:
        1. Duplicate combinations.
        2. Missing expected combinations.
        3. Unexpected combinations.
        4. Records with missing schedule identifiers.

    Expected participants:
        If not supplied, observed participant identifiers are used.

        To identify entirely absent participants, provide an
        authoritative participant roster through expected_participants.
    """

    key_cols = [
        participant_col,
        day_col,
        beep_col,
    ]

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if len(set(key_cols)) != len(key_cols):
        raise ValueError(
            "Schedule key columns must be distinct."
        )

    missing_cols = [
        column for column in key_cols
        if column not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required schedule columns: {missing_cols}"
        )

    valid_df = df.dropna(
        subset=key_cols
    ).copy()

    # --------------------------------------------------
    # 2. Determine expected schedule dimensions
    # --------------------------------------------------

    if expected_participants is not None:
        participants = list(expected_participants)

    else:
        participants = (
            valid_df[participant_col]
            .drop_duplicates()
            .tolist()
        )

    days = _get_expected_values(
        valid_df,
        day_col,
        expected_days,
    )

    beeps = _get_expected_values(
        valid_df,
        beep_col,
        expected_beeps,
    )

    # Remove duplicate expected values while preserving order.
    participants = list(dict.fromkeys(participants))
    days = list(dict.fromkeys(days))
    beeps = list(dict.fromkeys(beeps))

    # --------------------------------------------------
    # 3. Generate expected combinations
    # --------------------------------------------------

    expected_combinations = pd.MultiIndex.from_product(
        [participants, days, beeps],
        names=key_cols,
    ).to_frame(index=False)

    # --------------------------------------------------
    # 4. Count observed combinations
    # --------------------------------------------------

    observed_combinations = (
        valid_df
        .groupby(
            key_cols,
            dropna=False,
            sort=False,
        )
        .size()
        .reset_index(name="records")
    )

    # --------------------------------------------------
    # 5. Identify duplicate combinations
    # --------------------------------------------------

    duplicate_combinations = (
        observed_combinations[
            observed_combinations["records"] > 1
        ]
        .sort_values(
            "records",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    duplicate_groups = len(duplicate_combinations)

    duplicate_excess_rows = (
        int(
            duplicate_combinations["records"].sum()
            - duplicate_groups
        )
        if duplicate_groups
        else 0
    )

    # --------------------------------------------------
    # 6. Identify missing combinations
    # --------------------------------------------------

    missing_combinations = (
        expected_combinations
        .merge(
            observed_combinations[key_cols],
            on=key_cols,
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 7. Identify unexpected combinations
    # --------------------------------------------------

    unexpected_combinations = (
        observed_combinations
        .merge(
            expected_combinations,
            on=key_cols,
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 8. Identify invalid records
    # --------------------------------------------------

    invalid_records = df[
        df[key_cols].isna().any(axis=1)
    ].copy()

    # --------------------------------------------------
    # 9. Validation summary
    # --------------------------------------------------

    summary = pd.DataFrame({
        "Metric": [
            "Expected combinations",
            "Actual source rows",
            "Unique observed combinations",
            "Missing combinations",
            "Duplicate combinations",
            "Excess duplicate records",
            "Unexpected combinations",
            "Records with missing identifiers",
        ],
        "Value": [
            len(expected_combinations),
            len(df),
            len(observed_combinations),
            len(missing_combinations),
            duplicate_groups,
            duplicate_excess_rows,
            len(unexpected_combinations),
            len(invalid_records),
        ],
    })

    return {
        "summary": summary,
        "expected_combinations": expected_combinations,
        "observed_combinations": observed_combinations,
        "duplicate_combinations": duplicate_combinations,
        "missing_combinations": missing_combinations,
        "unexpected_combinations": unexpected_combinations,
        "invalid_records": invalid_records,
    }


# ---------------------------------------------------------------------
# Schedule reconstruction
# ---------------------------------------------------------------------

def reconstruct_schedule(
    df,
    participant_col="id",
    day_col="day",
    beep_col="beep",
    response_cols=None,
    expected_participants=None,
    expected_days=None,
    expected_beeps=None,
):
    """
    Reconstruct and validate an expected participant-day-beep schedule.

    Missing expected combinations are inserted as placeholder rows.
    Existing observations are preserved.

    Response columns:
        If response_cols is not supplied, get_response_columns()
        from standardization.py identifies likely survey response fields.

        Dataset-specific response_cols can be supplied when necessary.

    Response statuses:
        answered:
            All response fields are populated.

        partial:
            Some response fields are populated.

        nonresponse:
            All response fields are missing in an existing source row.

        not_observed:
            Expected schedule combination was absent from the source.

    Schedule statuses:
        valid:
            Expected, nonduplicate schedule record.

        duplicate:
            Record belongs to a duplicated schedule combination.

        unexpected:
            Record falls outside the expected schedule.

        invalid:
            Record has missing schedule identifiers.

        reconstructed:
            Placeholder inserted to complete the expected schedule.

    Boolean flags are independent and can overlap.

    Returns
    -------
    dict
        reconstructed_df:
            Original observations and inserted schedule placeholders.

        schedule_check:
            Detailed schedule validation results.

        summary:
            Reconstruction and response summary.
    """

    key_cols = [
        participant_col,
        day_col,
        beep_col,
    ]

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if len(set(key_cols)) != len(key_cols):
        raise ValueError(
            "Schedule key columns must be distinct."
        )

    missing_cols = [
        column for column in key_cols
        if column not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required schedule columns: {missing_cols}"
        )

    # --------------------------------------------------
    # 2. Identify response columns
    # --------------------------------------------------

    if response_cols is None:
        response_cols = get_response_columns(df)

    else:
        response_cols = list(response_cols)

    if not response_cols:
        raise ValueError(
            "No response columns were identified. "
            "Check NON_RESPONSE_COLUMNS in standardization.py "
            "or provide response_cols explicitly."
        )

    if len(response_cols) != len(set(response_cols)):
        raise ValueError(
            "response_cols contains duplicate column names."
        )

    missing_response_cols = [
        column for column in response_cols
        if column not in df.columns
    ]

    if missing_response_cols:
        raise ValueError(
            f"Response columns not found: {missing_response_cols}"
        )

    overlapping_cols = [
        column for column in response_cols
        if column in key_cols
    ]

    if overlapping_cols:
        raise ValueError(
            "Schedule key columns cannot be response columns: "
            f"{overlapping_cols}"
        )

    # --------------------------------------------------
    # 3. Validate schedule
    # --------------------------------------------------

    schedule_check = validate_schedule(
        df=df,
        participant_col=participant_col,
        day_col=day_col,
        beep_col=beep_col,
        expected_participants=expected_participants,
        expected_days=expected_days,
        expected_beeps=expected_beeps,
    )

    expected = schedule_check["expected_combinations"]
    missing = schedule_check["missing_combinations"]

    # --------------------------------------------------
    # 4. Prepare original observations
    # --------------------------------------------------

    reconstructed = df.copy().reset_index(drop=True)

    valid_keys = reconstructed[key_cols].notna().all(axis=1)

    reconstructed["is_duplicate"] = False
    reconstructed["is_unexpected"] = False
    reconstructed["is_invalid"] = ~valid_keys
    reconstructed["is_reconstructed"] = False

    # --------------------------------------------------
    # 5. Identify duplicate combinations
    # --------------------------------------------------

    duplicate_mask = (
        reconstructed.duplicated(
            subset=key_cols,
            keep=False,
        )
        & valid_keys
    )

    reconstructed.loc[
        duplicate_mask,
        "is_duplicate",
    ] = True

    # --------------------------------------------------
    # 6. Identify unexpected combinations
    # --------------------------------------------------

    expected_index = pd.MultiIndex.from_frame(
        expected[key_cols]
    )

    valid_indices = reconstructed.index[valid_keys]

    observed_index = pd.MultiIndex.from_frame(
        reconstructed.loc[valid_keys, key_cols]
    )

    unexpected_mask = ~observed_index.isin(expected_index)

    unexpected_indices = valid_indices[unexpected_mask]

    reconstructed.loc[
        unexpected_indices,
        "is_unexpected",
    ] = True

    # --------------------------------------------------
    # 7. Classify response completeness
    # --------------------------------------------------

    response_data = reconstructed[response_cols]

    all_missing = response_data.isna().all(axis=1)
    any_missing = response_data.isna().any(axis=1)

    reconstructed["is_nonresponse"] = all_missing

    reconstructed["is_partial_response"] = (
        any_missing & ~all_missing
    )

    reconstructed["response_status"] = np.select(
        [
            all_missing,
            any_missing,
        ],
        [
            "nonresponse",
            "partial",
        ],
        default="answered",
    )

    # --------------------------------------------------
    # 8. Insert missing schedule combinations
    # --------------------------------------------------

    missing_rows = pd.DataFrame(
        np.nan,
        index=range(len(missing)),
        columns=reconstructed.columns,
    )

    # Populate schedule identifiers.
    for column in key_cols:
        missing_rows[column] = missing[column].values

    # Schedule flags.
    missing_rows["is_duplicate"] = False
    missing_rows["is_unexpected"] = False
    missing_rows["is_invalid"] = False
    missing_rows["is_reconstructed"] = True

    # Response flags.
    # A reconstructed prompt is not an observed nonresponse.
    missing_rows["is_nonresponse"] = False
    missing_rows["is_partial_response"] = False
    missing_rows["response_status"] = "not_observed"

    # Append reconstructed placeholders.
    reconstructed = pd.concat(
        [
            reconstructed,
            missing_rows,
        ],
        ignore_index=True,
    )

    # Ensure flag columns have consistent boolean types.
    boolean_cols = [
        "is_duplicate",
        "is_unexpected",
        "is_invalid",
        "is_reconstructed",
        "is_nonresponse",
        "is_partial_response",
    ]

    for column in boolean_cols:
        reconstructed[column] = (
            reconstructed[column]
            .fillna(False)
            .astype(bool)
        )

    # --------------------------------------------------
    # 9. Assign schedule status
    # --------------------------------------------------

    reconstructed["schedule_status"] = "valid"

    reconstructed.loc[
        reconstructed["is_duplicate"],
        "schedule_status",
    ] = "duplicate"

    reconstructed.loc[
        reconstructed["is_unexpected"],
        "schedule_status",
    ] = "unexpected"

    reconstructed.loc[
        reconstructed["is_invalid"],
        "schedule_status",
    ] = "invalid"

    reconstructed.loc[
        reconstructed["is_reconstructed"],
        "schedule_status",
    ] = "reconstructed"

    # --------------------------------------------------
    # 10. Reconstruction summary
    # --------------------------------------------------

    original_observations = reconstructed[
        ~reconstructed["is_reconstructed"]
    ]

    summary = pd.DataFrame({
        "Metric": [
            "Original rows",
            "Expected combinations",
            "Expected combinations minus original rows",
            "Missing combinations inserted",
            "Duplicate records",
            "Unexpected records",
            "Invalid records",
            "Answered observations",
            "Entirely unanswered observations",
            "Partially answered observations",
            "Reconstructed rows",
        ],
        "Value": [
            len(df),
            len(expected),
            len(expected) - len(df),
            len(missing),
            int(reconstructed["is_duplicate"].sum()),
            int(reconstructed["is_unexpected"].sum()),
            int(reconstructed["is_invalid"].sum()),
            int(
                (
                    original_observations["response_status"]
                    == "answered"
                ).sum()
            ),
            int(
                (
                    original_observations["response_status"]
                    == "nonresponse"
                ).sum()
            ),
            int(
                (
                    original_observations["response_status"]
                    == "partial"
                ).sum()
            ),
            int(reconstructed["is_reconstructed"].sum()),
        ],
    })

    # --------------------------------------------------
    # 11. Return results
    # --------------------------------------------------

    return {
        "reconstructed_df": reconstructed,
        "schedule_check": schedule_check,
        "summary": summary,
    }
