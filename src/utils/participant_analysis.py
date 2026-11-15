import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_participation(
    participant_status,
    model_df,
    expected_days,
    expected_beeps,
    participant_col="id",
    day_col="day",
    beep_col="beep",
    response_col="responded",
):
    """
    Generate participant dropout and response-rate analyses.

    Returns:
        participant_status_viz
        dropout_counts
        response_data
        overall_rates
        retained_rates
        dropout_figure
        response_rate_figure
    """

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    required_status_columns = [
        participant_col,
        "last_response_day",
    ]

    required_model_columns = [
        participant_col,
        day_col,
        beep_col,
        response_col,
    ]

    missing_status = [
        column for column in required_status_columns
        if column not in participant_status.columns
    ]

    missing_model = [
        column for column in required_model_columns
        if column not in model_df.columns
    ]

    if missing_status:
        raise ValueError(
            f"Missing participant status columns: {missing_status}"
        )

    if missing_model:
        raise ValueError(
            f"Missing model columns: {missing_model}"
        )

    expected_days = list(expected_days)
    expected_beeps = list(expected_beeps)

    if not expected_days or not expected_beeps:
        raise ValueError(
            "Expected days and beeps cannot be empty."
        )

    final_study_day = max(expected_days)

    # --------------------------------------------------
    # Participant Status
    # --------------------------------------------------

    participant_status_viz = participant_status.copy()

    last_response_day = pd.to_numeric(
        participant_status_viz["last_response_day"],
        errors="coerce",
    )

    participant_status_viz["apparent_quit_day"] = (
        last_response_day + 1
    ).where(
        last_response_day < final_study_day
    )

    participant_status_viz["never_responded"] = (
        last_response_day.isna()
    )

    participant_status_viz["apparent_dropout"] = (
        last_response_day.notna()
        & (last_response_day < final_study_day)
    )

    participant_status_viz["retained"] = (
        last_response_day == final_study_day
    )

    # --------------------------------------------------
    # Dropout Distribution
    # --------------------------------------------------

    dropout_counts = (
        participant_status_viz[
            participant_status_viz["apparent_dropout"]
        ]
        .groupby("apparent_quit_day")
        .size()
        .reindex(
            range(
                min(expected_days) + 1,
                final_study_day + 1,
            ),
            fill_value=0,
        )
        .rename("participants")
        .reset_index()
    )

    dropout_counts.columns = [
        "apparent_quit_day",
        "participants",
    ]

    # --------------------------------------------------
    # Participant Groups
    # --------------------------------------------------

    retained_ids = (
        participant_status_viz.loc[
            participant_status_viz["retained"],
            participant_col,
        ]
        .dropna()
        .unique()
    )

    # --------------------------------------------------
    # Aggregate Duplicate Responses
    # --------------------------------------------------

    response_data = (
        model_df
        .groupby(
            [participant_col, day_col, beep_col],
            as_index=False,
        )[response_col]
        .max()
    )

    response_data[response_col] = (
        pd.to_numeric(
            response_data[response_col],
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
    )

    # --------------------------------------------------
    # Complete Expected Schedule
    # --------------------------------------------------

    participant_ids = (
        participant_status_viz[participant_col]
        .dropna()
        .unique()
    )

    expected_schedule = pd.MultiIndex.from_product(
        [
            participant_ids,
            expected_days,
            expected_beeps,
        ],
        names=[
            participant_col,
            day_col,
            beep_col,
        ],
    ).to_frame(index=False)

    response_data = expected_schedule.merge(
        response_data,
        on=[
            participant_col,
            day_col,
            beep_col,
        ],
        how="left",
        validate="one_to_one",
    )

    response_data[response_col] = (
        response_data[response_col]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------
    # Overall Response Rates
    # --------------------------------------------------

    overall_rates = (
        response_data
        .groupby([day_col, beep_col])[response_col]
        .mean()
        .mul(100)
        .unstack(beep_col)
        .reindex(
            index=expected_days,
            columns=expected_beeps,
        )
    )

    # --------------------------------------------------
    # Retained Participant Response Rates
    # --------------------------------------------------

    retained_rates = (
        response_data[
            response_data[participant_col].isin(retained_ids)
        ]
        .groupby([day_col, beep_col])[response_col]
        .mean()
        .mul(100)
        .unstack(beep_col)
        .reindex(
            index=expected_days,
            columns=expected_beeps,
        )
    )

    # --------------------------------------------------
    # Dropout Visualization
    # --------------------------------------------------

    dropout_figure, dropout_ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=dropout_counts,
        x="apparent_quit_day",
        y="participants",
        ax=dropout_ax,
    )

    dropout_ax.set_xlabel(
        "First day after the final observed response"
    )

    dropout_ax.set_ylabel(
        "Number of participants"
    )

    dropout_ax.set_title(
        "Apparent Disengagement by Study Day"
    )

    dropout_figure.tight_layout()

    # --------------------------------------------------
    # Response Rate Heatmap
    # --------------------------------------------------

    response_rate_figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            16,
            max(6, len(expected_days) * 0.25),
        ),
        sharey=True,
    )

    annotate = len(expected_days) <= 20

    sns.heatmap(
        overall_rates,
        ax=axes[0],
        annot=annotate,
        fmt=".0f",
        cmap="coolwarm_r",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar=False,
    )

    sns.heatmap(
        retained_rates,
        ax=axes[1],
        annot=annotate,
        fmt=".0f",
        cmap="coolwarm_r",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar=True,
        cbar_kws={
            "label": "Response rate (%)"
        },
    )

    axes[0].set_title(
        f"All Participants (n={len(participant_ids)})"
    )

    axes[1].set_title(
        f"Retained Participants (n={len(retained_ids)})"
    )

    for ax in axes:
        ax.set_xlabel("Beep Number")
        ax.set_xticklabels(expected_beeps)

        ax.set_yticklabels(
            expected_days,
            rotation=0,
        )

    axes[0].set_ylabel("Study Day")
    axes[1].set_ylabel("")

    response_rate_figure.suptitle(
        "Response Rates Across Study Days and Beeps",
        fontsize=15,
        y=1.01,
    )

    response_rate_figure.tight_layout()

    return {
        "participant_status_viz": participant_status_viz,
        "dropout_counts": dropout_counts,
        "response_data": response_data,
        "overall_rates": overall_rates,
        "retained_rates": retained_rates,
        "dropout_figure": dropout_figure,
        "response_rate_figure": response_rate_figure,
        "retained_ids": retained_ids,
    }




# ---------------------------------------------------------------------
# Temporal Participation Analysis
# ---------------------------------------------------------------------

def summarize_daily_activity(
    df,
    date_col="analysis_date",
    participant_col="id",
    timestamp_col="timestamp_analysis",
    duration_col="duration_minutes",
):
    """
    Summarize observed participation by calendar date.

    Returns one row per date, including:
        - Total observed responses
        - Number of unique participants
        - First and last observed response
        - Average and median response duration
    """

    required_columns = [date_col, participant_col]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = df.dropna(subset=[date_col]).copy()

    grouped = data.groupby(date_col)

    summary = grouped.size().rename(
        "total_responses"
    ).to_frame()

    summary["unique_participants"] = (
        grouped[participant_col].nunique()
    )

    if timestamp_col in data.columns:
        summary["first_response"] = (
            grouped[timestamp_col].min()
        )

        summary["last_response"] = (
            grouped[timestamp_col].max()
        )

    if duration_col in data.columns:
        summary["mean_duration_minutes"] = (
            grouped[duration_col].mean()
        )

        summary["median_duration_minutes"] = (
            grouped[duration_col].median()
        )

    return summary.reset_index()


def summarize_participant_daily_activity(
    df,
    participant_col="id",
    date_col="analysis_date",
    timestamp_col="timestamp_analysis",
    duration_col="duration_minutes",
):
    """
    Summarize observed participation for each participant
    on each calendar date.

    Returns one row per participant-date combination, including:
        - Total observed responses
        - First and last observed response
        - Average and median response duration
    """

    required_columns = [
        participant_col,
        date_col,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = df.dropna(
        subset=required_columns
    ).copy()

    grouped = data.groupby(
        [participant_col, date_col]
    )

    summary = grouped.size().rename(
        "responses"
    ).to_frame()

    if timestamp_col in data.columns:
        summary["first_response"] = (
            grouped[timestamp_col].min()
        )

        summary["last_response"] = (
            grouped[timestamp_col].max()
        )

    if duration_col in data.columns:
        summary["mean_duration_minutes"] = (
            grouped[duration_col].mean()
        )

        summary["median_duration_minutes"] = (
            grouped[duration_col].median()
        )

    return summary.reset_index()





# ---------------------------------------------------------------------
# Temporal Daily Activity Analysis
# ---------------------------------------------------------------------

def analyze_daily_activity(
    df,
    dataset_id=None,
    date_col="analysis_date",
    participant_col="id",
    timestamp_col="timestamp_analysis",
    duration_col="duration_minutes",
    save=False,
    topic="temporal_activity",
    config=None,
):
    """
    Analyze observed participation across calendar dates.

    Generates:
        - Daily activity summary
        - Daily response count visualization
        - Daily unique participant visualization

    Optionally saves:
        - Daily summary as a local Parquet file
        - Visualization locally and to S3

    This function summarizes observed activity only. It does not
    infer expected prompts or classify missing activity as nonresponse.
    """

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if save and dataset_id is None:
        raise ValueError(
            "dataset_id is required when save=True."
        )

    # --------------------------------------------------
    # Daily Activity Summary
    # --------------------------------------------------

    daily_summary = summarize_daily_activity(
        df=df,
        date_col=date_col,
        participant_col=participant_col,
        timestamp_col=timestamp_col,
        duration_col=duration_col,
    )

    if daily_summary.empty:
        raise ValueError(
            "No daily activity records were available."
        )

    daily_summary = daily_summary.sort_values(
        date_col
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Daily Activity Visualization
    # --------------------------------------------------

    daily_activity_figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
    )

    # Daily response counts

    sns.lineplot(
        data=daily_summary,
        x=date_col,
        y="total_responses",
        marker="o",
        ax=axes[0],
    )

    axes[0].set_title(
        "Daily Observed Response Counts"
    )

    axes[0].set_ylabel(
        "Number of Responses"
    )

    axes[0].set_xlabel("")

    # Daily unique participants

    sns.lineplot(
        data=daily_summary,
        x=date_col,
        y="unique_participants",
        marker="o",
        ax=axes[1],
    )

    axes[1].set_title(
        "Daily Active Participants"
    )

    axes[1].set_ylabel(
        "Unique Participants"
    )

    axes[1].set_xlabel(
        "Calendar Date"
    )

    for ax in axes:
        ax.tick_params(
            axis="x",
            rotation=45,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

    daily_activity_figure.suptitle(
        "Daily Participation Analysis",
        fontsize=15,
    )

    daily_activity_figure.tight_layout()

    # --------------------------------------------------
    # Optional Saving
    # --------------------------------------------------

    saved_outputs = {}

    if save:
        from utils.analysis_storage import (
            save_analysis_table,
            save_visualization,
        )

        saved_outputs["daily_summary"] = save_analysis_table(
            df=daily_summary,
            dataset_id=dataset_id,
            filename="daily_activity_summary.parquet",
            topic=topic,
        )

        saved_outputs["daily_activity_figure"] = (
            save_visualization(
                figure=daily_activity_figure,
                dataset_id=dataset_id,
                filename="daily_activity.png",
                topic=topic,
                config=config,
            )
        )

    # --------------------------------------------------
    # Return Results
    # --------------------------------------------------

    return {
        "daily_summary": daily_summary,
        "daily_activity_figure": daily_activity_figure,
        "saved_outputs": saved_outputs,
    }


# ---------------------------------------------------------------------
# Temporal Participant-Level Activity Analysis
# ---------------------------------------------------------------------

def analyze_participant_daily_activity(
    df,
    dataset_id=None,
    participant_col="id",
    date_col="analysis_date",
    timestamp_col="timestamp_analysis",
    duration_col="duration_minutes",
    max_participants=50,
    save=False,
    topic="temporal_activity",
    config=None,
):
    """
    Analyze observed participation at the participant-date level.

    Generates:
        - Participant-date activity summary
        - Total observed responses per participant
        - Participant-by-date activity heatmap
        - Distribution of total responses per participant

    The heatmap displays the participants with the highest observed
    response counts when max_participants is specified.

    Optionally saves:
        - Participant-date summary as a local Parquet file
        - Participant response totals as a local Parquet file
        - Visualizations locally and to S3

    Missing participant-date combinations in the heatmap represent
    no observed records in the supplied dataframe. They are not
    automatically classified as confirmed nonresponses.
    """

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if save and dataset_id is None:
        raise ValueError(
            "dataset_id is required when save=True."
        )

    if max_participants is not None:
        if (
            isinstance(max_participants, bool)
            or not isinstance(max_participants, int)
            or max_participants < 1
        ):
            raise ValueError(
                "max_participants must be a positive integer or None."
            )

    # --------------------------------------------------
    # Participant-Date Summary
    # --------------------------------------------------

    participant_daily_summary = (
        summarize_participant_daily_activity(
            df=df,
            participant_col=participant_col,
            date_col=date_col,
            timestamp_col=timestamp_col,
            duration_col=duration_col,
        )
    )

    if participant_daily_summary.empty:
        raise ValueError(
            "No participant-date activity records were available."
        )

    # --------------------------------------------------
    # Total Responses Per Participant
    # --------------------------------------------------

    participant_totals = (
        participant_daily_summary
        .groupby(
            participant_col,
            as_index=False,
        )
        .agg(
            total_responses=("responses", "sum"),
            active_days=(date_col, "nunique"),
        )
        .sort_values(
            "total_responses",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Participant-Date Activity Matrix
    # --------------------------------------------------

    activity_matrix = (
        participant_daily_summary
        .pivot(
            index=participant_col,
            columns=date_col,
            values="responses",
        )
        .fillna(0)
    )

    activity_matrix = activity_matrix.reindex(
        index=participant_totals[participant_col],
    )

    # Select participants for visualization

    if max_participants is not None:
        displayed_participants = (
            participant_totals
            .head(max_participants)[participant_col]
        )

        visualization_matrix = activity_matrix.loc[
            displayed_participants
        ]

    else:
        visualization_matrix = activity_matrix

    # --------------------------------------------------
    # Participant Activity Heatmap
    # --------------------------------------------------

    heatmap_height = max(
        6,
        len(visualization_matrix) * 0.25,
    )

    participant_activity_figure, ax = plt.subplots(
        figsize=(
            16,
            heatmap_height,
        )
    )

    sns.heatmap(
        visualization_matrix,
        ax=ax,
        cmap="YlGnBu",
        linewidths=0.2,
        cbar=True,
        cbar_kws={
            "label": "Observed Responses"
        },
    )

    ax.set_title(
        "Participant Activity Across Calendar Dates"
    )

    ax.set_xlabel(
        "Calendar Date"
    )

    ax.set_ylabel(
        "Participant"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    participant_activity_figure.tight_layout()

    # --------------------------------------------------
    # Participant Response Distribution
    # --------------------------------------------------

    response_distribution_figure, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.histplot(
        data=participant_totals,
        x="total_responses",
        bins="auto",
        kde=False,
        ax=ax,
    )

    ax.set_title(
        "Distribution of Observed Responses per Participant"
    )

    ax.set_xlabel(
        "Total Observed Responses"
    )

    ax.set_ylabel(
        "Number of Participants"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    response_distribution_figure.tight_layout()

    # --------------------------------------------------
    # Optional Saving
    # --------------------------------------------------

    saved_outputs = {}

    if save:
        from utils.analysis_storage import (
            save_analysis_table,
            save_visualization,
        )

        saved_outputs["participant_daily_summary"] = (
            save_analysis_table(
                df=participant_daily_summary,
                dataset_id=dataset_id,
                filename="participant_daily_activity.parquet",
                topic=topic,
            )
        )

        saved_outputs["participant_totals"] = (
            save_analysis_table(
                df=participant_totals,
                dataset_id=dataset_id,
                filename="participant_activity_totals.parquet",
                topic=topic,
            )
        )

        saved_outputs["activity_matrix"] = save_analysis_table(
            df=activity_matrix.reset_index(),
            dataset_id=dataset_id,
            filename="participant_activity_matrix.parquet",
            topic=topic,
        )

        saved_outputs["participant_activity_figure"] = (
            save_visualization(
                figure=participant_activity_figure,
                dataset_id=dataset_id,
                filename="participant_activity_heatmap.png",
                topic=topic,
                config=config,
            )
        )

        saved_outputs["response_distribution_figure"] = (
            save_visualization(
                figure=response_distribution_figure,
                dataset_id=dataset_id,
                filename="participant_response_distribution.png",
                topic=topic,
                config=config,
            )
        )

    # --------------------------------------------------
    # Return Results
    # --------------------------------------------------

    return {
        "participant_daily_summary": participant_daily_summary,
        "participant_totals": participant_totals,
        "activity_matrix": activity_matrix,
        "visualization_matrix": visualization_matrix,
        "participant_activity_figure": participant_activity_figure,
        "response_distribution_figure": response_distribution_figure,
        "saved_outputs": saved_outputs,
    }
