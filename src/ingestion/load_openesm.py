import openesm
import pandas as pd
import boto3
import yaml
from io import BytesIO
from pathlib import Path


def load_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Load an OpenESM dataset by its ID and return its contents as a
    pandas DataFrame.

    Parameters
    ----------
    dataset_id : str
        OpenESM dataset ID, e.g. "0006" or "73".

    Returns
    -------
    pd.DataFrame
        Dataset contents.

    Raises
    ------
    ValueError
        If the dataset ID is invalid or not found in the OpenESM catalog.
    """

    # Normalize dataset ID
    dataset_id = str(dataset_id).strip().zfill(4)

    if not dataset_id.isdigit():
        raise ValueError(
            f"Invalid dataset ID: {dataset_id!r}. "
            "Expected a numeric dataset ID."
        )

    # Retrieve OpenESM dataset catalog
    datasets = openesm.list_datasets()

    if "dataset_id" not in datasets.columns:
        raise RuntimeError(
            "Unexpected OpenESM catalog structure. "
            f"Available columns: {datasets.columns}"
        )

    # Check whether the requested dataset exists
    available_ids = (
        datasets["dataset_id"]
        .cast(str)
        .to_list()
    )

    if dataset_id not in available_ids:
        raise ValueError(
            f"Dataset {dataset_id} was not found in the OpenESM catalog."
        )

    # Retrieve and convert dataset
    dataset = openesm.get_dataset(dataset_id)

    return dataset.data.to_pandas()








def load_curated_dataset(
    dataset_id: str,
    config_path=None,
) -> pd.DataFrame:
    """
    Load a curated OpenESM dataset directly from S3.

    Automatically locates storage.yaml relative to the
    current working directory or project structure.

    Parameters
    ----------
    dataset_id : str
        OpenESM dataset ID, e.g. "0006" or "73".

    config_path : str or Path, optional
        Explicit path to storage.yaml. If omitted, searches
        the current directory and its parent directories.

    Returns
    -------
    pd.DataFrame
        Curated dataset contents.
    """

    # --------------------------------------------------
    # 1. Normalize dataset ID
    # --------------------------------------------------

    dataset_id = str(dataset_id).strip()

    if not dataset_id.isdigit():
        raise ValueError(
            f"Invalid dataset ID: {dataset_id!r}. "
            "Expected a numeric dataset ID."
        )

    dataset_id = dataset_id.zfill(4)

    # --------------------------------------------------
    # 2. Locate storage configuration
    # --------------------------------------------------

    if config_path is not None:

        config_path = Path(config_path)

    else:

        current_path = Path.cwd()

        config_path = next(
            (
                parent / "config" / "storage.yaml"
                for parent in [current_path, *current_path.parents]
                if (parent / "config" / "storage.yaml").exists()
            ),
            None,
        )

    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            "Could not locate config/storage.yaml. "
            "Pass config_path explicitly or run from "
            "within the project directory."
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # --------------------------------------------------
    # 3. Retrieve configuration
    # --------------------------------------------------

    aws_config = config["aws"]

    bucket = aws_config["bucket"]

    prefixes = config["s3"]["prefixes"]

    curated_prefix = (
        prefixes["curated"].strip("/") + "/"
    )

    # --------------------------------------------------
    # 4. Establish S3 connection
    # --------------------------------------------------

    session_kwargs = {}

    if aws_config.get("profile"):
        session_kwargs["profile_name"] = aws_config["profile"]

    if aws_config.get("region"):
        session_kwargs["region_name"] = aws_config["region"]

    session = boto3.Session(**session_kwargs)

    s3_client = session.client("s3")

    # --------------------------------------------------
    # 5. Locate curated dataset
    # --------------------------------------------------

    paginator = s3_client.get_paginator("list_objects_v2")

    matching_keys = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=curated_prefix,
    ):
        for obj in page.get("Contents", []):

            filename = obj["Key"].rsplit("/", 1)[-1]

            if (
                filename.startswith(f"{dataset_id}_")
                and filename.endswith("_curated.parquet")
            ):
                matching_keys.append(obj["Key"])

    if not matching_keys:
        raise FileNotFoundError(
            f"Curated dataset {dataset_id} was not found "
            f"in s3://{bucket}/{curated_prefix}"
        )

    if len(matching_keys) > 1:
        raise ValueError(
            f"Multiple curated datasets found for ID "
            f"{dataset_id}: {matching_keys}"
        )

    # --------------------------------------------------
    # 6. Load Parquet from S3
    # --------------------------------------------------

    response = s3_client.get_object(
        Bucket=bucket,
        Key=matching_keys[0],
    )

    return pd.read_parquet(
        BytesIO(response["Body"].read()),
        engine="pyarrow",
    )
