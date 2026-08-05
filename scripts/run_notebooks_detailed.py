#!/usr/bin/env python3
"""Execute notebooks with detailed error reporting."""

import sys
import json
import time
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = Path("notebooks")
TIMEOUT = 600  # 10 min per notebook


def execute_notebook(nb_path: Path) -> dict:
    """Execute a notebook and capture detailed error info."""
    print(f"\nExecuting: {nb_path.name}", file=sys.stderr)

    start_time = time.time()
    try:
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Create executor
        ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3")

        # Execute
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

        elapsed = time.time() - start_time

        # Save executed notebook
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        return {
            "notebook": nb_path.name,
            "success": True,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        return {
            "notebook": nb_path.name,
            "success": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": error_msg[:200],
            "error_type": type(e).__name__,
        }


def main():
    """Execute all notebooks."""
    import os

    os.chdir(Path(__file__).parent)

    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    notebooks = [nb for nb in notebooks if not nb.name.startswith("_")]

    if not notebooks:
        print("No notebooks found")
        return 1

    print(f"Found {len(notebooks)} notebooks", file=sys.stderr)

    results = []
    for nb_path in notebooks:
        result = execute_notebook(nb_path)
        results.append(result)
        status = "✓" if result["success"] else "✗"
        print(
            f"{status} {result['notebook']:40s} {result['elapsed_seconds']:6.1f}s", file=sys.stderr
        )

    # Summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"✓ {len(successful)}/{len(results)} successful", file=sys.stderr)
    if failed:
        print(f"✗ {len(failed)}/{len(results)} failed", file=sys.stderr)
        for r in failed:
            print(f"  {r['notebook']:40s} {r.get('error_type', 'UNKNOWN')}", file=sys.stderr)

    # Output JSON
    with open("notebook_execution_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
