from pathlib import Path
from ingestion import storage


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_dataset_id(dataset_id):
    return str(dataset_id).strip().zfill(4)


def _build_analysis_path(dataset_id, topic, filename):
    dataset_id = _normalize_dataset_id(dataset_id)
    topic_path = Path(topic)
    filename_path = Path(filename)

    if (
        not dataset_id
        or Path(dataset_id).name != dataset_id
        or dataset_id in {".", ".."}
    ):
        raise ValueError("Invalid dataset_id.")

    if (
        topic_path.is_absolute()
        or ".." in topic_path.parts
        or not topic_path.parts
    ):
        raise ValueError("Invalid analysis topic.")

    if filename_path.name != filename or filename in {"", ".", ".."}:
        raise ValueError("Filename must be a filename, not a path.")

    local_dir = PROJECT_ROOT / "analysis" / dataset_id / topic_path
    local_dir.mkdir(parents=True, exist_ok=True)

    return local_dir / filename


def save_analysis_table(
    df,
    dataset_id,
    filename,
    topic,
):
    """
    Save an analysis DataFrame locally as a Parquet file.
    Analysis tables are not uploaded to S3.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if Path(filename).suffix.lower() != ".parquet":
        raise ValueError("Analysis tables must use the .parquet extension.")

    local_path = _build_analysis_path(
        dataset_id=dataset_id,
        topic=topic,
        filename=filename,
    )

    df.to_parquet(
        local_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    print(f"Analysis table saved locally: {local_path}")

    return {
        "local_path": local_path,
    }


def save_visualization(
    figure,
    dataset_id,
    filename,
    topic="visualizations",
    config=None,
    dpi=300,
    bbox_inches="tight",
):
    """
    Save a visualization locally and upload the image to S3.
    """
    if config is None:
        config = storage.load_storage_config()

    allowed_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}

    if Path(filename).suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Unsupported visualization format: {Path(filename).suffix}"
        )

    local_path = _build_analysis_path(
        dataset_id=dataset_id,
        topic=topic,
        filename=filename,
    )

    figure.savefig(
        local_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
    )

    dataset_id = _normalize_dataset_id(dataset_id)
    topic_path = Path(topic)

    s3_prefix = config["s3"]["prefixes"]["analysis"].strip("/")
    s3_topic = topic_path.as_posix()

    s3_key = (
        f"{s3_prefix}/{dataset_id}/{s3_topic}/{filename}"
    )

    upload_result = storage.upload_file(
        file_path=local_path,
        s3_key=s3_key,
        config=config,
    )

    print(f"Visualization saved locally: {local_path}")
    print(f"Visualization uploaded: {upload_result['s3_uri']}")

    return {
        "local_path": local_path,
        "s3_uri": upload_result["s3_uri"],
        "size_bytes": upload_result["size_bytes"],
    }
