import numpy as np
import pandas as pd
import openesm


def inspect_dataset(
    df,
    dataset_id,
    participant_col="id",
    response_status_col="response_status",
):
    """
    Inspect an OpenESM dataset using catalog metadata
    and observed dataset contents.

    Returns:
        metadata:
            Dataset metadata retrieved from the OpenESM catalog.

        size_summary:
            Theoretical vs. actual dataset row counts.

        rows_per_participant:
            Number of rows per participant.

        row_distribution:
            Distribution of participant row counts.

        response_summary:
            Counts and percentages of answered, partial,
            and unanswered records, when response status exists.
    """

    # --------------------------------------------------
    # 1. Retrieve metadata from OpenESM catalog
    # --------------------------------------------------

    catalog = openesm.list_datasets()

    dataset_metadata = catalog.filter(
        catalog["dataset_id"].cast(str)
        == str(dataset_id).zfill(4)
    )

    if len(dataset_metadata) == 0:
        raise ValueError(
            f"Dataset {dataset_id} not found in catalog."
        )

    metadata = dataset_metadata.to_dicts()[0]

    metadata_fields = [
        "dataset_id",
        "first_author",
        "year",
        "n_participants",
        "n_time_points",
        "n_days",
        "n_beeps_per_day",
        "passive_data_available",
        "cross_sectional_available",
        "sampling_scheme",
        "implicit_missingness",
        "raw_time_stamp",
        "topics",
        "features",
    ]

    features = {
        field: metadata.get(field)
        for field in metadata_fields
    }

    # --------------------------------------------------
    # 2. Normalize numeric metadata
    # --------------------------------------------------

    numeric_fields = [
        "n_participants",
        "n_time_points",
        "n_days",
        "n_beeps_per_day",
    ]

    for field in numeric_fields:
        try:
            features[field] = int(features[field])
        except (ValueError, TypeError):
            features[field] = None

    # --------------------------------------------------
    # 3. Theoretical vs. actual dataset size
    # --------------------------------------------------

    participants = features["n_participants"]
    days = features["n_days"]
    beeps_per_day = features["n_beeps_per_day"]

    actual_rows = len(df)

    if all(
        value is not None
        for value in [participants, days, beeps_per_day]
    ):
        theoretical_rows = (
            participants * days * beeps_per_day
        )

        difference = theoretical_rows - actual_rows

        difference_pct = (
            difference / theoretical_rows * 100
            if theoretical_rows > 0
            else np.nan
        )

        size_summary = pd.DataFrame({
            "Metric": [
                "Theoretical rows",
                "Actual rows",
                "Difference",
                "Difference (%)",
            ],
            "Value": [
                theoretical_rows,
                actual_rows,
                difference,
                round(difference_pct, 2)
                if not np.isnan(difference_pct)
                else np.nan,
            ],
        })

    else:
        theoretical_rows = None
        difference = None
        difference_pct = None
        size_summary = None

    # --------------------------------------------------
    # 4. Rows per participant
    # --------------------------------------------------

    if participant_col not in df.columns:
        raise ValueError(
            f"Participant column '{participant_col}' not found."
        )

    rows_per_participant = (
        df.dropna(subset=[participant_col])
        .groupby(participant_col)
        .size()
    )

    row_distribution = (
        rows_per_participant
        .value_counts()
        .sort_index()
        .rename_axis("rows")
        .reset_index(name="participants")
    )

    total_participants = len(rows_per_participant)

    if total_participants > 0:
        row_distribution["percentage"] = (
            row_distribution["participants"]
            / total_participants
            * 100
        ).round(2)
    else:
        row_distribution["percentage"] = pd.Series(
            dtype="float64"
        )

    # --------------------------------------------------
    # 5. Response summary
    # --------------------------------------------------

    response_summary = None

    if response_status_col in df.columns:
        response_counts = (
            df[response_status_col]
            .value_counts(dropna=False)
        )

        response_summary = (
            response_counts
            .rename_axis("response_status")
            .reset_index(name="records")
        )

        total_records = len(df)

        response_summary["percentage"] = (
            response_summary["records"]
            / total_records
            * 100
            if total_records > 0
            else 0
        ).round(2)

    # --------------------------------------------------
    # 6. Return results
    # --------------------------------------------------

    return {
        "metadata": features,
        "size_summary": size_summary,
        "rows_per_participant": rows_per_participant,
        "row_distribution": row_distribution,
        "response_summary": response_summary,
        "actual_rows": actual_rows,
        "theoretical_rows": theoretical_rows,
        "row_difference": difference,
        "row_difference_pct": difference_pct,
    }
