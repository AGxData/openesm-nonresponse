import openesm
import polars as pl
from src.metadata.save_local import save_parquet


def get_catalogue() -> pl.DataFrame:
    catalogue = openesm.list_datasets()

    if not isinstance(catalogue, pl.DataFrame):
        raise TypeError(
            "Expected openesm.list_datasets() to return "
            f"a Polars DataFrame, got {type(catalogue).__name__}"
        )

    return catalogue


def save_catalogue() -> tuple[pl.DataFrame, str]:
    catalogue = get_catalogue()

    path = save_parquet(
        catalogue,
        "metadata/catalog.parquet",
    )

    print(f"Catalogue records: {catalogue.height}")
    print(f"Saved locally: {path}")

    return catalogue, str(path)
