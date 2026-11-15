
import json
from pathlib import Path

path = Path("metadata/column_inventory.json")
data = json.loads(path.read_text(encoding="utf-8"))

print("SUMMARY")
print(json.dumps(data["summary"], indent=2))

print("\nSHARED VARIABLES")
print(list(data["shared_variables"].keys()))

print("\n30 MOST COMMON VARIABLES")
variables = sorted(
    data["all_variables"].items(),
    key=lambda item: item[1]["dataset_count"],
    reverse=True,
)

for name, details in variables[:30]:
    print(
        f"{name}: {details['dataset_count']} datasets; "
        f"types={details['dtypes']}; "
        f"names={details['name_variants']}"
    )
