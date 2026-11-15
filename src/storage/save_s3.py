import boto3

from src.config import load_config


def get_s3_client():
    config = load_config("storage.yaml")
    aws = config["aws"]

    session = boto3.Session(
        profile_name=aws["profile"],
        region_name=aws["region"],
    )

    return session.client("s3")


def upload_file(local_path: str, key: str) -> None:
    config = load_config("storage.yaml")
    bucket = config["aws"]["bucket"]

    get_s3_client().upload_file(
        local_path,
        bucket,
        key,
    )
