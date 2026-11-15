
# Imports
import matplotlib.pyplot as plt
import seaborn as sns

from ingestion import load_dataset, storage
from preprocessing.standardization import standardize_dataset
from preprocessing.schedule_reconstruction import reconstruct_schedule
from preprocessing.feature_engineering import engineer_features
from preprocessing.missingness import identify_nonresponse_rows, analyze_missingness, compare_missingness_groups
from utils.participant_analysis import analyze_participation
from utils.dataset_inspection import inspect_dataset
from utils.analysis_storage import save_analysis_table, save_visualization

sns.set_style("whitegrid")


# Configuration
DATASET_ID = "0053"


# Missingness Configuration
CORE_RESPONSE_COLUMNS = [
    "guilty",
    "anxious",
    "inspired",
    "happy",
    "depressed",
    "confident",
]


MISSINGNESS_GROUPS = {
    "emotional_affect": [
        "guilty",
        "anxious",
        "inspired",
        "happy",
        "depressed",
        "confident",
    ],

    "body_image_satisfaction": [
        "weight_satisfaction",
        "appearance_satisfaction",
        "shape_satisfaction",
    ],

    "exposure_variables": [
        "fits_exposure",
        "thins_exposure",
        "bo_po_exposure",
        "dual_thin_fits",
        "dual_thin_bo_po",
        "dual_fits_bo_po",
        "triple_exposure",
    ],

    "response_timing": [
        "response_time",
        "response_lag",
    ],

    "temporal_and_schedule_variables": [
        "counter",
        "day",
        "beep",
        "hour",
        "weekend",
    ],

    "response_status": [
        "responded_fixed",
    ],
}

MISSINGNESS_RULES = [
    CORE_RESPONSE_COLUMNS,
]

MISSINGNESS_ANALYSIS_COLUMNS = list(
    dict.fromkeys(
        col
        for columns in MISSINGNESS_GROUPS.values()
        for col in columns
    )
)
MAX_MISSINGNESS_PATTERNS = None
MAX_MISSINGNESS_CLUSTERS = None


# Load Dataset
df = load_dataset(DATASET_ID)

# ID 54474 has no responses at all. Rows where counter is empty are excesive. Day and Beep columns need reconstruction.
def reconstruct_day_beep(df):
    df = df[df["counter"].notna() & (df["id"] != 54474)].copy().reset_index(drop=True)
    df = df.drop(columns=["day", "beep"], errors="ignore")
    df["day"] = (((df["counter"] - 1) // 6) + 1).astype(int)
    df["beep"] = (((df["counter"] - 1) % 6) + 1).astype(int)
    return df

df = reconstruct_day_beep(df)
df = standardize_dataset(df)

# Removing participant with duplicate response that was null
df = df.loc[
    ~(
        (df["id"] == 57453)
        & (df["counter"] == 7)
        & (df["response_time"].isna())
    )
].copy().reset_index(drop=True)

# Exploratory Missingness Analysis
missingness_analysis = analyze_missingness(
    df=df,
    response_cols=MISSINGNESS_ANALYSIS_COLUMNS,
    max_patterns=MAX_MISSINGNESS_PATTERNS,
    max_clusters=MAX_MISSINGNESS_CLUSTERS,
)

group_analysis = compare_missingness_groups(
    df=df,
    groups=MISSINGNESS_GROUPS,
)


# Dataset Inspection
inspection = inspect_dataset(
    df=df,
    dataset_id=DATASET_ID,
)

response_cols = CORE_RESPONSE_COLUMNS

metadata = inspection["metadata"]

# Accounting for removed participant with no responses
metadata["n_participants"] -= 1

EXPECTED_DAYS = list(
    range(1, int(metadata["n_days"]) + 1)
)

EXPECTED_BEEPS = list(
    range(1, int(metadata["n_beeps_per_day"]) + 1)
)

FINAL_STUDY_DAY = max(EXPECTED_DAYS)


# Schedule Reconstruction
reconstruction = reconstruct_schedule(
    df=df,
    response_cols=response_cols,
    expected_days=EXPECTED_DAYS,
    expected_beeps=EXPECTED_BEEPS,
)

df_reconstructed = reconstruction["reconstructed_df"]
schedule_check = reconstruction["schedule_check"]
summary = reconstruction["summary"]


# Feature Engineering
features = engineer_features(
    df=df_reconstructed,
)

model_df = features["model_df"]
daily_cumulative = features["daily_cumulative"]
daily_participation = features["daily_participation"]
participant_status = features["participant_status"]


# S3 Dataset Storage
storage_results = storage.save_dataset(
    processed_df=model_df,
    source_df=df_reconstructed,
    inspection=inspection,
)


# Participant Analysis
analysis = analyze_participation(
    participant_status=participant_status,
    model_df=model_df,
    expected_days=EXPECTED_DAYS,
    expected_beeps=EXPECTED_BEEPS,
)

participant_status_viz = analysis["participant_status_viz"]
dropout_counts = analysis["dropout_counts"]
response_data = analysis["response_data"]
overall_rates = analysis["overall_rates"]
retained_rates = analysis["retained_rates"]

dropout_figure = analysis["dropout_figure"]
response_rate_figure = analysis["response_rate_figure"]

retained_ids = analysis["retained_ids"]


# Analysis Storage
config = storage.load_storage_config()


# Save Analysis Tables Locally

# Participant Analysis
save_analysis_table(
    participant_status_viz,
    DATASET_ID,
    "participant_status.parquet",
    "participant_dropout",
)

save_analysis_table(
    dropout_counts,
    DATASET_ID,
    "dropout_distribution.parquet",
    "participant_dropout",
)

save_analysis_table(
    overall_rates,
    DATASET_ID,
    "overall_response_rates.parquet",
    "response_rates",
)

save_analysis_table(
    retained_rates,
    DATASET_ID,
    "retained_response_rates.parquet",
    "response_rates",
)


# Missingness Analysis
save_analysis_table(
    missingness_analysis["column_summary"],
    DATASET_ID,
    "column_missingness.parquet",
    "missingness",
)

save_analysis_table(
    missingness_analysis["row_summary"],
    DATASET_ID,
    "row_missingness.parquet",
    "missingness",
)

save_analysis_table(
    missingness_analysis["completeness_distribution"],
    DATASET_ID,
    "completeness_distribution.parquet",
    "missingness",
)

save_analysis_table(
    missingness_analysis["pattern_summary"],
    DATASET_ID,
    "missingness_patterns.parquet",
    "missingness",
)

save_analysis_table(
    missingness_analysis["pairwise_similarity"],
    DATASET_ID,
    "pairwise_missingness_similarity.parquet",
    "missingness",
)

save_analysis_table(
    missingness_analysis["missing_value_counts"],
    DATASET_ID,
    "missing_value_counts.parquet",
    "missingness",
)

save_analysis_table(
    group_analysis["group_summary"],
    DATASET_ID,
    "missingness_group_comparison.parquet",
    "missingness",
)

save_analysis_table(
    group_analysis["column_summary"],
    DATASET_ID,
    "missingness_group_variables.parquet",
    "missingness",
)


# Save Visualizations Locally and to S3
try:
    save_visualization(
        figure=dropout_figure,
        dataset_id=DATASET_ID,
        filename="participant_dropout.png",
        topic="visualizations",
        config=config,
    )

    save_visualization(
        figure=response_rate_figure,
        dataset_id=DATASET_ID,
        filename="response_rate_heatmap.png",
        topic="visualizations",
        config=config,
    )

    save_visualization(
        figure=missingness_analysis["dendrogram_figure"],
        dataset_id=DATASET_ID,
        filename="missingness_dendrogram.png",
        topic="missingness",
        config=config,
    )

finally:
    plt.close(dropout_figure)
    plt.close(response_rate_figure)
    plt.close(missingness_analysis["dendrogram_figure"])
