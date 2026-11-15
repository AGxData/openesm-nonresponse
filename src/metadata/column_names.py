
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INVENTORY_PATH = PROJECT_ROOT / "metadata" / "column_inventory.json"
OUTPUT_PATH = PROJECT_ROOT / "metadata" / "unique_column_names.parquet"


def load_column_inventory():
    with INVENTORY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_column_frequency(inventory):
    total_datasets = inventory["summary"]["datasets_scanned"]
    variables = inventory["all_variables"]

    records = []

    for normalized_name, details in variables.items():
        datasets = details.get("datasets", {})

        records.append({
            "normalized_name": normalized_name,
            "name_variants": details.get("name_variants", []),
            "dataset_count": details.get("dataset_count", 0),
            "dataset_percentage": round(
                details.get("dataset_count", 0) / total_datasets * 100,
                2,
            ),
            "datasets": list(datasets.keys()),
            "column_names": sorted({
                entry["column_name"]
                for entry in datasets.values()
            }),
            "dtypes": details.get("dtypes", []),
            "shared_across_scanned_datasets": details.get(
                "shared_across_scanned_datasets", False
            ),
        })

    df = pd.DataFrame(records)

    return df.sort_values(
        ["dataset_count", "normalized_name"],
        ascending=[False, True],
    ).reset_index(drop=True)


def export_column_frequency():
    inventory = load_column_inventory()
    df = build_column_frequency(inventory)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Unique variables: {len(df):,}")
    print(f"Datasets scanned: {inventory['summary']['datasets_scanned']:,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nMost common variables:")
    print(
        df[[
            "normalized_name",
            "dataset_count",
            "dataset_percentage",
            "dtypes",
        ]].head(20).to_string(index=False)
    )

    return df


if __name__ == "__main__":
    df = export_column_frequency()
