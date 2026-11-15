from pathlib import Path
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_parquet(
    data: pl.DataFrame,
    relative_path: str | Path,
) -> Path:
    path = Path(relative_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(path)

    return path
