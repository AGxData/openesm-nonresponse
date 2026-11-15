
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "metadata" / "column_inventory.json"
OUTPUT_PATH = PROJECT_ROOT / "metadata" / "variable_frequency_summary.json"


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    variables = data["all_variables"]

    frequencies = Counter(
        details["dataset_count"]
        for details in variables.values()
    )

    summary = {
        "source": "metadata/column_inventory.json",
        "datasets_scanned": data["summary"]["datasets_scanned"],
        "complete_inventory": data["summary"]["complete_inventory"],
        "unique_variables": len(variables),
        "shared_by_dataset_count": {
            str(count): sum(
                details["dataset_count"] == count
                for details in variables.values()
            )
            for count in sorted(frequencies, reverse=True)
        },
        "thresholds": {},
        "most_common_variables": [],
    }

    for threshold in [63, 50, 40, 20, 10]:
        summary["thresholds"][str(threshold)] = [
            {
                "variable": name,
                "datasets": details["dataset_count"],
                "dtypes": details["dtypes"],
            }
            for name, details in sorted(
                variables.items(),
                key=lambda item: (
                    -item[1]["dataset_count"],
                    item[0],
                ),
            )
            if details["dataset_count"] >= threshold
        ]

    summary["most_common_variables"] = [
        {
            "variable": name,
            "datasets": details["dataset_count"],
            "dtypes": details["dtypes"],
            "name_variants": details["name_variants"],
        }
        for name, details in sorted(
            variables.items(),
            key=lambda item: (
                -item[1]["dataset_count"],
                item[0],
            ),
        )[:30]
    ]

    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Unique variables: {len(variables)}")
    for threshold, matches in summary["thresholds"].items():
        print(f"Present in at least {threshold} datasets: {len(matches)}")


if __name__ == "__main__":
    main()
