
from pathlib import Path
import subprocess
import sys
import time

from tqdm import tqdm


# Configuration
PIPELINE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIRECTORY.parent
ORCHESTRATOR_NAME = "preprocess_all.py"


# Discover Preprocessing Scripts
pipeline_files = sorted(
    path
    for path in PIPELINE_DIRECTORY.glob("preprocess_*.py")
    if path.is_file()
    and path.name != ORCHESTRATOR_NAME
)

if not pipeline_files:
    print(f"No preprocessing scripts found in: {PIPELINE_DIRECTORY}")
    sys.exit(1)


# Execute Preprocessing Pipelines
results = []

print("=" * 70)
print("RUNNING ALL DATASET PREPROCESSING PIPELINES")
print("=" * 70)
print(f"Scripts discovered: {len(pipeline_files)}")


with tqdm(
    pipeline_files,
    desc="Preprocessing",
    unit="dataset",
    dynamic_ncols=True,
) as progress:

    for script in progress:

        progress.set_postfix_str(script.stem)

        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=PROJECT_ROOT,
                check=False,
            )

            elapsed = time.perf_counter() - start_time

            results.append({
                "script": script.name,
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "elapsed": elapsed,
            })

        except Exception as error:

            elapsed = time.perf_counter() - start_time

            results.append({
                "script": script.name,
                "success": False,
                "return_code": None,
                "elapsed": elapsed,
            })

            tqdm.write(f"Execution error in {script.name}: {error}")

        status = "SUCCESS" if results[-1]["success"] else "FAILED"

        tqdm.write(
            f"{status:<10} "
            f"{script.name:<35} "
            f"{elapsed:.2f}s"
        )


# Execution Summary
print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

for result in results:

    status = "SUCCESS" if result["success"] else "FAILED"

    print(
        f'{status:<10} '
        f'{result["script"]:<35} '
        f'{result["elapsed"]:.2f}s'
    )


# Final Statistics
successful = sum(result["success"] for result in results)
failed = len(results) - successful
total_time = sum(result["elapsed"] for result in results)

print("-" * 70)
print(f"Total scripts: {len(results)}")
print(f"Successful:    {successful}")
print(f"Failed:        {failed}")
print(f"Total runtime: {total_time:.2f}s")


# Failed Pipelines
if failed:

    print("\nFailed pipelines:")

    for result in results:
        if not result["success"]:
            print(
                f'  {result["script"]} '
                f'(exit code: {result["return_code"]})'
            )

    sys.exit(1)


print("\nAll preprocessing pipelines completed successfully.")
