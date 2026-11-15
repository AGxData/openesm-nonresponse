import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Response indicator utilities
# ---------------------------------------------------------------------

def _to_boolean_indicator(series):
    """
    Convert common boolean representations into a boolean Series.

    Missing and unrecognized values are treated as False.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return normalized.isin({"true", "1", "yes", "y"})


def _get_response_indicator(
    df,
    response_col="responded",
    response_status_col="response_status",
    reconstructed_col="is_reconstructed",
    nonresponse_col="is_nonresponse",
):
    """
    Obtain a binary response indicator.

    A partial response counts as a response.

    Reconstructed prompts and explicitly identified nonresponses
    are forced to zero.

    Response status takes precedence over an existing indicator.

    Missing or unrecognized response statuses are treated as
    nonresponses. Input data should distinguish genuine
    nonresponses from unavailable or invalid observations.
    """

    if response_status_col in df.columns:

        response_status = (
            df[response_status_col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        response = (
            response_status
            .isin({"answered", "partial"})
            .astype("int8")
        )

    elif response_col in df.columns:

        response = pd.to_numeric(
            df[response_col],
            errors="coerce",
        )

        invalid_values = response.dropna()

        invalid_values = invalid_values[
            ~invalid_values.isin([0, 1])
        ]

        if not invalid_values.empty:
            raise ValueError(
                f"'{response_col}' must contain only 0 and 1. "
                f"Invalid values: {invalid_values.unique().tolist()}"
            )

        response = response.fillna(0).astype("int8")

    else:
        return None

    if reconstructed_col in df.columns:

        reconstructed = _to_boolean_indicator(
            df[reconstructed_col]
        )

        response = response.mask(reconstructed, 0)

    if nonresponse_col in df.columns:

        nonresponse = _to_boolean_indicator(
            df[nonresponse_col]
        )

        response = response.mask(nonresponse, 0)

    return response.astype("int8")


# ---------------------------------------------------------------------
# Response streak utilities
# ---------------------------------------------------------------------

def _prior_streak(series, value):
    """
    Calculate the length of the consecutive prior run of `value`.

    The current observation is excluded.

    Parameters
    ----------
    series : pandas.Series
        Chronologically ordered binary response indicator.

    value : int
        Value defining the streak (1 = responses, 0 = nonresponses).

    Returns
    -------
    pandas.Series
        Number of consecutive previous observations equal to value.
    """

    previous = series.shift(1)

    run_id = previous.ne(previous.shift()).cumsum()

    streak = (
        previous
        .groupby(run_id)
        .cumcount()
        + 1
    )

    return streak.where(
        previous.eq(value),
        0,
    ).astype("int64")


# ---------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------

def engineer_features(
    df,
    participant_col="id",
    day_col="day",
    beep_col="beep",
    response_col="responded",
    response_status_col="response_status",
    rolling_windows=(3, 6, 12),
    daily_windows=(3, 7),
    ewm_spans=(3, 6),
    beta_prior_alpha=1.0,
    beta_prior_beta=1.0,
):
    """
    Generate temporal and behavioral features from prompt-level data.

    Each row should represent one scheduled prompt, including
    reconstructed nonresponses.

    Features use participant ID, study day, beep number, and
    response information.

    All historical behavioral features exclude the current prompt.

    Parameters
    ----------
    df : pandas.DataFrame or object with to_pandas()
        Prompt-level dataset.

    participant_col : str, default="id"
        Participant identifier column.

    day_col : str, default="day"
        Study day column.

    beep_col : str, default="beep"
        Beep number within the study day.

    response_col : str, default="responded"
        Binary response indicator column.

    response_status_col : str, default="response_status"
        Optional categorical response status column.
        "answered" and "partial" count as responses.

    rolling_windows : tuple, default=(3, 6, 12)
        Number of previous prompts used for rolling response rates.

    daily_windows : tuple, default=(3, 7)
        Number of previous observed days used for daily rolling rates.

    ewm_spans : tuple, default=(3, 6)
        Exponential weighting spans.

    beta_prior_alpha : float, default=1.0
        Alpha parameter for Beta smoothing.

    beta_prior_beta : float, default=1.0
        Beta parameter for Beta smoothing.

    Returns
    -------
    dict
        model_df:
            Prompt-level dataset with engineered features.

        daily_cumulative:
            Retrospective daily snapshots of cumulative response history.

        daily_participation:
            Daily participation summaries and historical daily features.

        participant_status:
            Retrospective participant-level summaries.

    Important
    ---------
    Features prefixed with "prior_" and "prev_" use historical
    information only.

    Current-day summaries such as responded_any_today,
    daily_response_rate, and missed_day are retrospective.

    Participant-level summaries are also retrospective.

    Schedule features such as prompts_in_day and last_beep_of_day
    are safe predictors only when the complete schedule is known
    before the target prompt.
    """

    # --------------------------------------------------
    # 1. Validate and prepare input
    # --------------------------------------------------

    if not isinstance(df, pd.DataFrame):

        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        else:
            raise TypeError(
                "Input must be a pandas DataFrame or an object "
                "with a to_pandas() method."
            )

    required_cols = [
        participant_col,
        day_col,
        beep_col,
    ]

    missing_cols = [
        column for column in required_cols
        if column not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    for name, windows in [
        ("rolling_windows", rolling_windows),
        ("daily_windows", daily_windows),
        ("ewm_spans", ewm_spans),
    ]:

        if not windows or any(
            not isinstance(window, (int, np.integer))
            or isinstance(window, bool)
            or window < 1
            for window in windows
        ):

            raise ValueError(
                f"{name} must contain positive integers."
            )

        if len(set(windows)) != len(windows):

            raise ValueError(
                f"{name} must not contain duplicate values."
            )

    if beta_prior_alpha <= 0 or beta_prior_beta <= 0:

        raise ValueError(
            "Beta prior parameters must both be greater than zero."
        )

    model_df = df.copy()

    internal_cols = [
        "_feature_row_order",
        "_feature_day_sort",
        "_feature_beep_sort",
    ]

    collisions = [
        column for column in internal_cols
        if column in model_df.columns
    ]

    if collisions:
        raise ValueError(
            f"Input contains reserved internal columns: {collisions}"
        )

    model_df["_feature_row_order"] = np.arange(len(model_df))

    model_df["_feature_day_sort"] = pd.to_numeric(
        model_df[day_col],
        errors="coerce",
    )

    model_df["_feature_beep_sort"] = pd.to_numeric(
        model_df[beep_col],
        errors="coerce",
    )

    if model_df["_feature_day_sort"].isna().any():

        raise ValueError(
            f"'{day_col}' must be numeric or numerically convertible."
        )

    if model_df["_feature_beep_sort"].isna().any():

        raise ValueError(
            f"'{beep_col}' must be numeric or numerically convertible."
        )

    model_df = (
        model_df
        .sort_values(
            [
                participant_col,
                "_feature_day_sort",
                "_feature_beep_sort",
                "_feature_row_order",
            ],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    participant_groups = model_df.groupby(
        participant_col,
        sort=False,
        dropna=False,
    )

    participant_day_groups = model_df.groupby(
        [participant_col, day_col],
        sort=False,
        dropna=False,
    )

    model_df["day_index"] = model_df["_feature_day_sort"]

    model_df["beep_index"] = model_df["_feature_beep_sort"]

    # --------------------------------------------------
    # 2. Basic temporal features
    # --------------------------------------------------

    model_df["prompt_index"] = (
        participant_groups.cumcount() + 1
    )

    model_df["day_squared"] = (
        model_df["day_index"] ** 2
    )

    model_df["beep_squared"] = (
        model_df["beep_index"] ** 2
    )

    model_df["day_beep_interaction"] = (
        model_df["day_index"] * model_df["beep_index"]
    )

    first_observed_day = (
        participant_groups["day_index"]
        .transform("min")
    )

    last_observed_day = (
        participant_groups["day_index"]
        .transform("max")
    )

    model_df["days_since_first_observed"] = (
        model_df["day_index"] - first_observed_day
    )

    model_df["first_observed_day"] = (
        model_df["day_index"]
        .eq(first_observed_day)
        .astype("int8")
    )

    model_df["last_observed_day"] = (
        model_df["day_index"]
        .eq(last_observed_day)
        .astype("int8")
    )

    # --------------------------------------------------
    # 3. Schedule structure features
    # --------------------------------------------------

    model_df["prompts_in_day"] = (
        participant_day_groups[beep_col]
        .transform("size")
    )

    first_beep = (
        participant_day_groups["beep_index"]
        .transform("min")
    )

    last_beep = (
        participant_day_groups["beep_index"]
        .transform("max")
    )

    model_df["first_beep_of_day"] = (
        model_df["beep_index"]
        .eq(first_beep)
        .astype("int8")
    )

    model_df["last_beep_of_day"] = (
        model_df["beep_index"]
        .eq(last_beep)
        .astype("int8")
    )

    beep_rank = (
        participant_day_groups["beep_index"]
        .rank(method="first")
    )

    model_df["beep_position"] = (
        beep_rank / model_df["prompts_in_day"]
    )

    previous_day = (
        participant_groups["day_index"]
        .shift(1)
    )

    previous_beep = (
        participant_groups["beep_index"]
        .shift(1)
    )

    model_df["days_since_previous_prompt"] = (
        model_df["day_index"] - previous_day
    )

    model_df["beep_difference_from_previous"] = (
        model_df["beep_index"] - previous_beep
    )

    model_df["beep_gap_from_previous_same_day"] = (
        participant_day_groups["beep_index"].diff()
    )

    model_df["scheduled_beeps_in_day"] = (
        model_df["prompts_in_day"]
    )

    model_df["scheduled_beep_position"] = (
        participant_day_groups.cumcount() + 1
    )

    model_df["scheduled_beep_position_rate"] = (
        model_df["scheduled_beep_position"]
        / model_df["scheduled_beeps_in_day"]
    )

    # --------------------------------------------------
    # 4. Response-dependent features
    # --------------------------------------------------

    responded = _get_response_indicator(
        model_df,
        response_col=response_col,
        response_status_col=response_status_col,
    )

    daily_cumulative = None
    daily_participation = None
    participant_status = None

    if responded is not None:

        model_df[response_col] = responded

        participant_response = model_df.groupby(
            participant_col,
            sort=False,
            dropna=False,
        )[response_col]

        participant_day_response = model_df.groupby(
            [participant_col, day_col],
            sort=False,
            dropna=False,
        )[response_col]

        # ----------------------------------------------
        # 4A. Cumulative response history
        # ----------------------------------------------

        model_df["cumulative_responses"] = (
            participant_response.cumsum()
        )

        model_df["cumulative_prompts"] = (
            participant_groups.cumcount() + 1
        )

        model_df["cumulative_response_rate"] = (
            model_df["cumulative_responses"]
            / model_df["cumulative_prompts"]
            * 100
        )

        model_df["prior_cumulative_responses"] = (
            model_df["cumulative_responses"]
            - model_df[response_col]
        )

        model_df["prior_cumulative_prompts"] = (
            model_df["cumulative_prompts"] - 1
        )

        model_df["prior_cumulative_response_rate"] = (
            model_df["prior_cumulative_responses"]
            .div(
                model_df["prior_cumulative_prompts"].replace(
                    0,
                    np.nan,
                )
            )
            * 100
        )

        model_df["prior_cumulative_nonresponses"] = (
            model_df["prior_cumulative_prompts"]
            - model_df["prior_cumulative_responses"]
        )

        model_df["prior_missed_beep_count"] = (
            model_df["prior_cumulative_nonresponses"]
        )

        model_df["prior_smoothed_response_rate"] = (
            (
                model_df["prior_cumulative_responses"]
                + beta_prior_alpha
            )
            / (
                model_df["prior_cumulative_prompts"]
                + beta_prior_alpha
                + beta_prior_beta
            )
            * 100
        )

        model_df["prior_response_history_count"] = (
            model_df["prior_cumulative_prompts"]
        )

        # ----------------------------------------------
        # 4B. Previous prompt lags
        # ----------------------------------------------

        for lag in range(1, 7):

            model_df[f"prev_{lag}_response"] = (
                participant_response.shift(lag)
            )

        model_df["prev_response"] = (
            model_df["prev_1_response"]
        )

        model_df["prev_response_same_day"] = (
            participant_day_response.shift(1)
        )

        # ----------------------------------------------
        # 4C. Rolling prior prompt response rates
        # ----------------------------------------------

        for window in rolling_windows:

            model_df[
                f"prior_{window}_prompt_response_rate"
            ] = participant_response.transform(
                lambda series, w=window: (
                    series.shift(1)
                    .rolling(w, min_periods=1)
                    .mean()
                    * 100
                )
            )

        # ----------------------------------------------
        # 4D. Exponentially weighted response rates
        # ----------------------------------------------

        for span in ewm_spans:

            model_df[
                f"prior_ewm_{span}_prompt_response_rate"
            ] = participant_response.transform(
                lambda series, s=span: (
                    series.shift(1)
                    .ewm(
                        span=s,
                        adjust=False,
                        min_periods=1,
                    )
                    .mean()
                    * 100
                )
            )

        # ----------------------------------------------
        # 4E. Prior response streaks across prompts
        # ----------------------------------------------

        model_df["prior_response_streak"] = (
            participant_response.transform(
                lambda series: _prior_streak(series, 1)
            )
        )

        model_df["prior_nonresponse_streak"] = (
            participant_response.transform(
                lambda series: _prior_streak(series, 0)
            )
        )

        # ----------------------------------------------
        # 4F. Time since previous response
        # ----------------------------------------------

        response_day = model_df["day_index"].where(
            model_df[response_col].eq(1)
        )

        previous_response_day = (
            response_day
            .groupby(
                model_df[participant_col],
                sort=False,
                dropna=False,
            )
            .ffill()
            .groupby(
                model_df[participant_col],
                sort=False,
                dropna=False,
            )
            .shift(1)
        )

        model_df["days_since_previous_response"] = (
            model_df["day_index"] - previous_response_day
        )

        # ----------------------------------------------
        # 4G. Within-day prior response history
        # ----------------------------------------------

        model_df["prior_same_day_responses"] = (
            participant_day_response.cumsum()
            - model_df[response_col]
        )

        # Calculate the number of previous prompts first.
        model_df["prior_same_day_prompts"] = (
            participant_day_groups.cumcount()
        )

        # Previous prompts minus previous responses.
        model_df["prior_same_day_nonresponses"] = (
            model_df["prior_same_day_prompts"]
            - model_df["prior_same_day_responses"]
        )

        model_df["prior_same_day_response_rate"] = (
            model_df["prior_same_day_responses"]
            .div(
                model_df["prior_same_day_prompts"].replace(
                    0,
                    np.nan,
                )
            )
            * 100
        )

        model_df["prior_same_day_smoothed_response_rate"] = (
            (
                model_df["prior_same_day_responses"]
                + beta_prior_alpha
            )
            / (
                model_df["prior_same_day_prompts"]
                + beta_prior_alpha
                + beta_prior_beta
            )
            * 100
        )

        model_df["prior_same_day_response_streak"] = (
            participant_day_response.transform(
                lambda series: _prior_streak(series, 1)
            )
        )

        model_df["prior_same_day_nonresponse_streak"] = (
            participant_day_response.transform(
                lambda series: _prior_streak(series, 0)
            )
        )

        for window in (2, 3):

            model_df[
                f"prior_same_day_{window}_prompt_response_rate"
            ] = participant_day_response.transform(
                lambda series, w=window: (
                    series.shift(1)
                    .rolling(w, min_periods=1)
                    .mean()
                    * 100
                )
            )

        # Same-day exponentially weighted history.
        for span in ewm_spans:

            model_df[
                f"prior_same_day_ewm_{span}_response_rate"
            ] = participant_day_response.transform(
                lambda series, s=span: (
                    series.shift(1)
                    .ewm(
                        span=s,
                        adjust=False,
                        min_periods=1,
                    )
                    .mean()
                    * 100
                )
            )

        # ----------------------------------------------
        # 4H. Participant × beep historical behavior
        # ----------------------------------------------

        participant_beep_groups = model_df.groupby(
            [participant_col, beep_col],
            sort=False,
            dropna=False,
        )

        model_df["prior_same_beep_responses"] = (
            participant_beep_groups[response_col]
            .transform("cumsum")
            - model_df[response_col]
        )

        model_df["prior_same_beep_observations"] = (
            participant_beep_groups.cumcount()
        )

        model_df["prior_same_beep_nonresponses"] = (
            model_df["prior_same_beep_observations"]
            - model_df["prior_same_beep_responses"]
        )

        model_df["prior_same_beep_response_rate"] = (
            model_df["prior_same_beep_responses"]
            .div(
                model_df["prior_same_beep_observations"].replace(
                    0,
                    np.nan,
                )
            )
            * 100
        )

        model_df["prior_same_beep_smoothed_rate"] = (
            (
                model_df["prior_same_beep_responses"]
                + beta_prior_alpha
            )
            / (
                model_df["prior_same_beep_observations"]
                + beta_prior_alpha
                + beta_prior_beta
            )
            * 100
        )

        model_df["prior_same_beep_response_streak"] = (
            participant_beep_groups[response_col].transform(
                lambda series: _prior_streak(series, 1)
            )
        )

        model_df["prior_same_beep_nonresponse_streak"] = (
            participant_beep_groups[response_col].transform(
                lambda series: _prior_streak(series, 0)
            )
        )

        model_df["prev_same_beep_response"] = (
            participant_beep_groups[response_col].shift(1)
        )

        model_df["days_since_previous_same_beep"] = (
            participant_beep_groups["day_index"].diff()
        )

        for span in ewm_spans:

            model_df[
                f"prior_same_beep_ewm_{span}_response_rate"
            ] = participant_beep_groups[response_col].transform(
                lambda series, s=span: (
                    series.shift(1)
                    .ewm(
                        span=s,
                        adjust=False,
                        min_periods=1,
                    )
                    .mean()
                    * 100
                )
            )

        # ----------------------------------------------
        # 4I. Previous calendar day's same-beep response
        # ----------------------------------------------

        same_beep_source = model_df[
            [
                participant_col,
                "day_index",
                "beep_index",
                response_col,
            ]
        ].copy()

        duplicate_schedule_keys = same_beep_source.duplicated(
            [
                participant_col,
                "day_index",
                "beep_index",
            ],
            keep=False,
        )

        if duplicate_schedule_keys.any():

            examples = (
                same_beep_source.loc[
                    duplicate_schedule_keys,
                    [
                        participant_col,
                        "day_index",
                        "beep_index",
                    ],
                ]
                .drop_duplicates()
                .head(10)
                .to_dict("records")
            )

            raise ValueError(
                "Duplicate participant/day/beep rows prevent "
                "unambiguous same-beep lag matching. Examples: "
                f"{examples}"
            )

        same_beep_source = same_beep_source.rename(
            columns={
                "day_index": "_previous_day_index",
                response_col: "prev_day_same_beep_response",
            }
        )

        same_beep_source["_current_day_index"] = (
            same_beep_source["_previous_day_index"] + 1
        )

        same_beep_source["prev_day_same_beep_available"] = 1

        same_beep_features = same_beep_source[
            [
                participant_col,
                "_current_day_index",
                "beep_index",
                "prev_day_same_beep_response",
                "prev_day_same_beep_available",
            ]
        ]

        model_df = model_df.merge(
            same_beep_features,
            left_on=[
                participant_col,
                "day_index",
                "beep_index",
            ],
            right_on=[
                participant_col,
                "_current_day_index",
                "beep_index",
            ],
            how="left",
            sort=False,
            validate="one_to_one",
        )

        model_df["prev_day_same_beep_available"] = (
            model_df["prev_day_same_beep_available"]
            .fillna(0)
            .astype("int8")
        )

        model_df["prev_day_same_beep_response"] = (
            model_df["prev_day_same_beep_response"].astype("float")
        )

        model_df = model_df.drop(
            columns=["_current_day_index"]
        )

        # ----------------------------------------------
        # 4J. Daily participation summaries
        # ----------------------------------------------

        daily_participation = (
            model_df
            .groupby(
                [participant_col, day_col],
                sort=False,
                dropna=False,
            )
            .agg(
                responded_any=(response_col, "max"),
                daily_responses=(response_col, "sum"),
                daily_prompts=(response_col, "size"),
                daily_response_rate=(response_col, "mean"),
            )
            .reset_index()
        )

        daily_participation["daily_response_rate"] *= 100

        daily_participation["_day_sort"] = pd.to_numeric(
            daily_participation[day_col],
            errors="coerce",
        )

        daily_participation = (
            daily_participation
            .sort_values(
                [participant_col, "_day_sort"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        daily_groups = daily_participation.groupby(
            participant_col,
            sort=False,
            dropna=False,
        )

        # Historical average daily response rate.
        daily_participation["prior_day_response_rate"] = (
            daily_groups["daily_response_rate"].transform(
                lambda series: (
                    series.shift(1)
                    .expanding(min_periods=1)
                    .mean()
                )
            )
        )

        daily_participation["prior_days_with_response"] = (
            daily_groups["responded_any"].transform(
                lambda series: (
                    series.shift(1)
                    .fillna(0)
                    .cumsum()
                )
            )
        )

        daily_participation["prior_days_observed"] = (
            daily_groups.cumcount()
        )

        daily_participation["prior_day_participation_rate"] = (
            daily_participation["prior_days_with_response"]
            .div(
                daily_participation["prior_days_observed"].replace(
                    0,
                    np.nan,
                )
            )
            * 100
        )

        daily_participation["prev_day_responded"] = (
            daily_groups["responded_any"].shift(1)
        )

        daily_participation["prev_day_response_rate"] = (
            daily_groups["daily_response_rate"].shift(1)
        )

        daily_participation["days_since_previous_observed_day"] = (
            daily_groups["_day_sort"].diff()
        )

        # Consecutive observed-day participation streaks.
        daily_participation["prior_consecutive_active_days"] = (
            daily_groups["responded_any"].transform(
                lambda series: _prior_streak(series, 1)
            )
        )

        daily_participation["prior_consecutive_inactive_days"] = (
            daily_groups["responded_any"].transform(
                lambda series: _prior_streak(series, 0)
            )
        )

        # Recent daily response history.
        for window in daily_windows:

            daily_participation[
                f"prior_{window}_day_response_rate"
            ] = daily_groups["daily_response_rate"].transform(
                lambda series, w=window: (
                    series.shift(1)
                    .rolling(w, min_periods=1)
                    .mean()
                )
            )

            daily_participation[
                f"prior_{window}_day_participation_rate"
            ] = daily_groups["responded_any"].transform(
                lambda series, w=window: (
                    series.shift(1)
                    .rolling(w, min_periods=1)
                    .mean()
                    * 100
                )
            )

        # Exponentially weighted daily response history.
        for span in ewm_spans:

            daily_participation[
                f"prior_ewm_{span}_day_response_rate"
            ] = daily_groups["daily_response_rate"].transform(
                lambda series, s=span: (
                    series.shift(1)
                    .ewm(
                        span=s,
                        adjust=False,
                        min_periods=1,
                    )
                    .mean()
                )
            )

        # Compare recent daily behavior with longer-term history.
        if 3 in daily_windows:

            daily_participation["prior_response_rate_change"] = (
                daily_participation["prior_3_day_response_rate"]
                - daily_participation["prior_day_response_rate"]
            )

        else:
            daily_participation["prior_response_rate_change"] = np.nan

        # ----------------------------------------------
        # 4K. Map prior daily features to prompt-level data
        # ----------------------------------------------

        daily_feature_cols = [
            participant_col,
            day_col,
            "responded_any",
            "daily_response_rate",
            "prior_day_response_rate",
            "prior_days_with_response",
            "prior_days_observed",
            "prior_day_participation_rate",
            "prev_day_responded",
            "prev_day_response_rate",
            "days_since_previous_observed_day",
            "prior_consecutive_active_days",
            "prior_consecutive_inactive_days",
            "prior_response_rate_change",
        ]

        for window in daily_windows:

            daily_feature_cols.extend([
                f"prior_{window}_day_response_rate",
                f"prior_{window}_day_participation_rate",
            ])

        for span in ewm_spans:

            daily_feature_cols.append(
                f"prior_ewm_{span}_day_response_rate"
            )

        daily_features = daily_participation[
            daily_feature_cols
        ].rename(
            columns={
                "responded_any": "responded_any_today",
            }
        )

        model_df = model_df.merge(
            daily_features,
            on=[
                participant_col,
                day_col,
            ],
            how="left",
            sort=False,
            validate="many_to_one",
        )

        # Retrospective current-day summaries.
        model_df["missed_day"] = (
            model_df["responded_any_today"]
            .eq(0)
            .astype("int8")
        )

        # ----------------------------------------------
        # 4L. Daily cumulative response summaries
        # ----------------------------------------------

        daily_cumulative = (
            model_df
            .groupby(
                [participant_col, day_col],
                sort=False,
                dropna=False,
            )
            .agg(
                cumulative_responses=(
                    "cumulative_responses",
                    "last",
                ),
                cumulative_prompts=(
                    "cumulative_prompts",
                    "last",
                ),
                cumulative_response_rate=(
                    "cumulative_response_rate",
                    "last",
                ),
                responded_any=(response_col, "max"),
                daily_response_rate=(response_col, "mean"),
            )
            .reset_index()
        )

        daily_cumulative["daily_response_rate"] *= 100

        # ----------------------------------------------
        # 4M. Participant-level retrospective summaries
        # ----------------------------------------------

        daily_responded = daily_participation[
            daily_participation["responded_any"].eq(1)
        ]

        last_response = (
            daily_responded
            .groupby(
                participant_col,
                sort=False,
                dropna=False,
            )[day_col]
            .max()
            .rename("last_response_day")
        )

        first_response = (
            daily_responded
            .groupby(
                participant_col,
                sort=False,
                dropna=False,
            )[day_col]
            .min()
            .rename("first_response_day")
        )

        participant_status = (
            model_df[[participant_col]]
            .drop_duplicates()
            .set_index(participant_col)
            .join(last_response)
            .join(first_response)
            .reset_index()
        )

        participant_totals = (
            model_df
            .groupby(
                participant_col,
                sort=False,
                dropna=False,
            )
            .agg(
                total_observed_prompts=(response_col, "size"),
                total_responses=(response_col, "sum"),
                overall_response_rate=(response_col, "mean"),
            )
            .reset_index()
        )

        participant_totals["overall_response_rate"] *= 100

        participant_status = participant_status.merge(
            participant_totals,
            on=participant_col,
            how="left",
            validate="one_to_one",
        )

        participant_status["never_responded"] = (
            participant_status["last_response_day"].isna()
        )

        daily_participation = daily_participation.drop(
            columns=["_day_sort"]
        )

    # --------------------------------------------------
    # 5. Restore original row order and remove internals
    # --------------------------------------------------

    model_df = (
        model_df
        .sort_values(
            "_feature_row_order",
            kind="stable",
        )
        .drop(
            columns=[
                "_feature_row_order",
                "_feature_day_sort",
                "_feature_beep_sort",
            ],
            errors="ignore",
        )
        .reset_index(drop=True)
    )

    return {
        "model_df": model_df,
        "daily_cumulative": daily_cumulative,
        "daily_participation": daily_participation,
        "participant_status": participant_status,
    }