
import numpy as np
import pandas as pd


def identify_nonresponse_rows(
    df,
    missingness_rules,
    combine_rules="any",
):
    """
    Identify rows that should be classified as nonresponses.
    """
    if combine_rules not in {"any", "all"}:
        raise ValueError("combine_rules must be 'any' or 'all'.")

    if not missingness_rules:
        raise ValueError("At least one missingness rule is required.")

    rule_results = []

    for rule in missingness_rules:
        missing_columns = [
            column for column in rule
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns required by rule: {missing_columns}"
            )

        rule_mask = df[rule].isna().all(axis=1)
        rule_results.append(rule_mask)

    results = pd.concat(rule_results, axis=1)

    if combine_rules == "any":
        return results.any(axis=1)

    return results.all(axis=1)


# ---------------------------------------------------------------------
# Exploratory missingness analysis
# ---------------------------------------------------------------------

DEFAULT_NON_RESPONSE_COLUMNS = {
    "id",
    "participant_id",
    "subject_id",
    "day",
    "study_day",
    "beep",
    "prompt",
    "prompt_id",
    "counter",
    "time",
    "timestamp",
    "datetime",
    "date",
    "created_at",
    "updated_at",
    "response_status",
    "schedule_status",
}


def _get_candidate_response_columns(
    df,
    response_cols=None,
    exclude_cols=None,
):
    """
    Identify columns to include in exploratory missingness summaries.

    If response_cols is provided, those columns are used explicitly.
    Otherwise, common structural fields are excluded.
    """

    if response_cols is not None:
        columns = list(response_cols)

    else:
        excluded = DEFAULT_NON_RESPONSE_COLUMNS.copy()

        if exclude_cols is not None:
            excluded.update(
                str(column).lower()
                for column in exclude_cols
            )

        columns = [
            column
            for column in df.columns
            if str(column).lower() not in excluded
            and not str(column).lower().startswith("is_")
        ]

    if not columns:
        raise ValueError(
            "No candidate response columns were identified."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "response_cols contains duplicate column names."
        )

    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Response columns not found in dataframe: {missing_columns}"
        )

    return columns


def _build_missingness_column_summary(response_data):
    """Calculate missingness statistics for each response variable."""

    n_rows = len(response_data)

    summary = pd.DataFrame({
        "column": response_data.columns,
        "missing_count": response_data.isna().sum().values,
        "observed_count": response_data.notna().sum().values,
    })

    summary["missing_rate"] = (
        summary["missing_count"] / n_rows
        if n_rows
        else np.nan
    )

    summary["observed_rate"] = (
        summary["observed_count"] / n_rows
        if n_rows
        else np.nan
    )

    summary["all_missing"] = summary["missing_count"].eq(n_rows)
    summary["no_missing"] = summary["missing_count"].eq(0)

    return summary.sort_values(
        ["missing_rate", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _build_missingness_row_summary(response_data):
    """Calculate response completeness for each source row."""

    missing_count = response_data.isna().sum(axis=1)
    observed_count = response_data.notna().sum(axis=1)

    n_columns = len(response_data.columns)

    result = pd.DataFrame({
        "source_row_number": np.arange(len(response_data)),
        "missing_count": missing_count.values,
        "observed_count": observed_count.values,
        "missing_rate": (
            missing_count / n_columns
        ).values,
        "observed_rate": (
            observed_count / n_columns
        ).values,
    })

    result["completeness_category"] = np.select(
        [
            result["observed_count"].eq(0),
            result["missing_count"].eq(0),
        ],
        [
            "entirely_missing",
            "complete",
        ],
        default="partial",
    )

    return result


def _build_missingness_pattern_summary(
    response_data,
    max_patterns=None,
):
    """Summarize observed combinations of missing variables."""

    if max_patterns is not None:
        if (
            isinstance(max_patterns, bool)
            or not isinstance(max_patterns, (int, np.integer))
            or max_patterns < 1
        ):
            raise ValueError(
                "max_patterns must be a positive integer or None."
            )

    missing = response_data.isna()

    pattern_data = pd.DataFrame({
        "missing_pattern": missing.apply(
            lambda row: "".join(
                "1" if value else "0"
                for value in row
            ),
            axis=1,
        ),
        "missing_count": missing.sum(axis=1).values,
    })

    pattern_summary = (
        pattern_data
        .groupby(
            ["missing_pattern", "missing_count"],
            dropna=False,
        )
        .size()
        .reset_index(name="row_count")
        .sort_values(
            ["row_count", "missing_count"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    pattern_summary["row_rate"] = (
        pattern_summary["row_count"] / len(response_data)
        if len(response_data)
        else np.nan
    )

    if max_patterns is not None:
        pattern_summary = pattern_summary.head(
            max_patterns
        ).copy()

    column_names = list(response_data.columns)

    pattern_summary["missing_columns"] = (
        pattern_summary["missing_pattern"].apply(
            lambda pattern: ", ".join(
                str(column)
                for column, indicator in zip(
                    column_names,
                    pattern,
                )
                if indicator == "1"
            )
        )
    )

    return pattern_summary[
        [
            "missing_pattern",
            "missing_columns",
            "missing_count",
            "row_count",
            "row_rate",
        ]
    ]


def _build_pairwise_missingness_similarity(response_data):
    """
    Calculate pairwise Jaccard similarity between missingness patterns.
    """

    missing = response_data.isna().astype(bool)
    columns = list(missing.columns)

    records = []

    for i, column_a in enumerate(columns):
        missing_a = missing[column_a]

        for column_b in columns[i + 1:]:
            missing_b = missing[column_b]

            intersection = int(
                (missing_a & missing_b).sum()
            )

            union = int(
                (missing_a | missing_b).sum()
            )

            similarity = (
                intersection / union
                if union
                else 1.0
            )

            records.append({
                "column_a": column_a,
                "column_b": column_b,
                "shared_missing_rows": intersection,
                "union_missing_rows": union,
                "jaccard_similarity": similarity,
                "jaccard_distance": 1 - similarity,
            })

    return pd.DataFrame(
        records,
        columns=[
            "column_a",
            "column_b",
            "shared_missing_rows",
            "union_missing_rows",
            "jaccard_similarity",
            "jaccard_distance",
        ],
    ).sort_values(
        "jaccard_similarity",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Missingness group comparison
# ---------------------------------------------------------------------

def compare_missingness_groups(
    df,
    groups,
):
    """
    Compare missingness across manually defined response groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Standardized dataset.

    groups : dict
        Dictionary mapping group names to lists of response columns.

    Returns
    -------
    dict
        group_summary:
            Group-level missingness and completeness statistics.

        column_summary:
            Individual variable missingness statistics, including
            the group to which each variable belongs.

    Notes
    -----
    Group missingness rate is calculated across all cells in the group.

    Complete-row rate measures the proportion of source rows where
    every variable in that group is observed.

    The input dataframe is not modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot compare missingness in an empty dataframe."
        )

    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            "groups must be a non-empty dictionary."
        )

    group_records = []
    column_records = []

    assigned_columns = set()

    for group_name, columns in groups.items():

        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError(
                "Every group must have a non-empty string name."
            )

        if not isinstance(columns, (list, tuple)):
            raise TypeError(
                f"Columns for group '{group_name}' must be a list or tuple."
            )

        if not columns:
            raise ValueError(
                f"Group '{group_name}' does not contain any columns."
            )

        if len(columns) != len(set(columns)):
            raise ValueError(
                f"Group '{group_name}' contains duplicate columns."
            )

        missing_columns = [
            column for column in columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Columns for group '{group_name}' not found in dataframe: "
                f"{missing_columns}"
            )

        overlapping_columns = [
            column for column in columns
            if column in assigned_columns
        ]

        if overlapping_columns:
            raise ValueError(
                f"Columns assigned to multiple groups: "
                f"{overlapping_columns}"
            )

        assigned_columns.update(columns)

        group_data = df[columns]

        n_rows = len(group_data)
        n_columns = len(columns)

        total_values = group_data.size

        missing_counts = group_data.isna().sum()
        observed_counts = group_data.notna().sum()

        total_missing = int(missing_counts.sum())
        total_observed = int(observed_counts.sum())

        rows_with_any_missing = int(
            group_data.isna().any(axis=1).sum()
        )

        rows_with_all_missing = int(
            group_data.isna().all(axis=1).sum()
        )

        rows_with_all_present = int(
            group_data.notna().all(axis=1).sum()
        )

        group_records.append({
            "group": group_name,
            "n_variables": n_columns,
            "n_rows": n_rows,
            "total_values": total_values,
            "missing_values": total_missing,
            "observed_values": total_observed,
            "missingness_rate": total_missing / total_values,
            "observed_rate": total_observed / total_values,
            "rows_with_any_missing": rows_with_any_missing,
            "rows_with_all_missing": rows_with_all_missing,
            "rows_with_all_present": rows_with_all_present,
            "complete_row_rate": rows_with_all_present / n_rows,
        })

        for column in columns:

            missing_count = int(missing_counts[column])
            observed_count = int(observed_counts[column])

            column_records.append({
                "group": group_name,
                "column": column,
                "missing_count": missing_count,
                "observed_count": observed_count,
                "missingness_rate": missing_count / n_rows,
                "observed_rate": observed_count / n_rows,
                "all_missing": missing_count == n_rows,
                "no_missing": missing_count == 0,
            })

    group_summary = (
        pd.DataFrame(group_records)
        .sort_values(
            ["missingness_rate", "complete_row_rate"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    column_summary = (
        pd.DataFrame(column_records)
        .sort_values(
            ["missingness_rate", "column"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return {
        "group_summary": group_summary,
        "column_summary": column_summary,
    }


# ---------------------------------------------------------------------
# Complete exploratory missingness analysis
# ---------------------------------------------------------------------

def analyze_missingness(
    df,
    response_cols=None,
    exclude_cols=None,
    max_patterns=None,
    max_clusters=None,
):
    """
    Perform exploratory missingness analysis on a standardized dataset.

    The Missingno dendrogram is generated from the entire dataframe.
    The remaining exploratory summaries use response_cols when supplied,
    or automatically detected response columns otherwise.

    The input dataframe is not modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot analyze missingness in an empty dataframe."
        )

    # --------------------------------------------------
    # Select response columns for summary analysis
    # --------------------------------------------------

    columns = _get_candidate_response_columns(
        df=df,
        response_cols=response_cols,
        exclude_cols=exclude_cols,
    )

    response_data = df[columns].copy()

    # --------------------------------------------------
    # Missing-value counts across ALL original columns
    # --------------------------------------------------

    missing_value_counts = (
        df.isna()
        .sum()
        .rename("missing_count")
        .rename_axis("column")
        .reset_index()
    )

    # --------------------------------------------------
    # Missingness summaries
    # --------------------------------------------------

    column_summary = _build_missingness_column_summary(
        response_data
    )

    row_summary = _build_missingness_row_summary(
        response_data
    )

    pattern_summary = _build_missingness_pattern_summary(
        response_data,
        max_patterns=max_patterns,
    )

    pairwise_similarity = _build_pairwise_missingness_similarity(
        response_data
    )

    # --------------------------------------------------
    # Completeness distribution
    # --------------------------------------------------

    completeness_distribution = (
        row_summary
        .groupby(
            "completeness_category",
            as_index=False,
        )
        .agg(
            row_count=("source_row_number", "count"),
            mean_observed_rate=("observed_rate", "mean"),
        )
    )

    # --------------------------------------------------
    # Missingno dendrogram
    # --------------------------------------------------

    import matplotlib.pyplot as plt
    import missingno as msno

    msno.dendrogram(df=df)

    dendrogram_figure = plt.gcf()

    dendrogram_figure.tight_layout()

    # --------------------------------------------------
    # Return results
    # --------------------------------------------------

    return {
        "response_columns": columns,
        "missing_value_counts": missing_value_counts,
        "column_summary": column_summary,
        "row_summary": row_summary,
        "completeness_distribution": completeness_distribution,
        "pattern_summary": pattern_summary,
        "pairwise_similarity": pairwise_similarity,
        "dendrogram_figure": dendrogram_figure,
    }
