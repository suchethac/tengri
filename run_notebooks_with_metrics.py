#!/usr/bin/env python3
"""Execute notebooks with timing and memory metrics."""
import subprocess
import os
import json
import time
from pathlib import Path
import psutil
import sys

NOTEBOOKS_DIR = Path("notebooks")
METRICS_FILE = Path("notebook_metrics.json")

def get_memory_usage():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def execute_notebook(nb_path: Path) -> dict:
    """Execute a single notebook and capture metrics."""
    print(f"\n{'='*60}")
    print(f"Executing: {nb_path.name}")
    print(f"{'='*60}")

    mem_before = get_memory_usage()
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                str(nb_path)
            ],
            capture_output=True,
            text=True,
            timeout=900,  # 15 min timeout
        )

        elapsed = time.time() - start_time
        mem_after = get_memory_usage()
        mem_delta = mem_after - mem_before

        success = result.returncode == 0

        metrics = {
            "notebook": nb_path.name,
            "success": success,
            "elapsed_seconds": round(elapsed, 2),
            "memory_before_mb": round(mem_before, 1),
            "memory_after_mb": round(mem_after, 1),
            "memory_delta_mb": round(mem_delta, 1),
            "returncode": result.returncode,
        }

        if not success:
            metrics["stderr"] = result.stderr[-500:] if result.stderr else ""
            metrics["stdout"] = result.stdout[-500:] if result.stdout else ""

        print(f"✓ {nb_path.name}")
        print(f"  Time: {elapsed:.1f}s | Memory: {mem_before:.0f}MB → {mem_after:.0f}MB (Δ {mem_delta:+.0f}MB)")

        return metrics

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT after {elapsed:.1f}s: {nb_path.name}")
        return {
            "notebook": nb_path.name,
            "success": False,
            "error": "TIMEOUT",
            "elapsed_seconds": round(elapsed, 2),
        }
    except Exception as e:
        print(f"✗ ERROR: {nb_path.name} - {e}")
        return {
            "notebook": nb_path.name,
            "success": False,
            "error": str(e),
        }

def main():
    """Run all notebooks with metrics."""
    os.chdir(Path(__file__).parent)

    if not NOTEBOOKS_DIR.exists():
        print(f"Error: {NOTEBOOKS_DIR} not found")
        sys.exit(1)

    # Find all .ipynb notebooks
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    # Skip helper notebooks
    notebooks = [nb for nb in notebooks if not nb.name.startswith("_")]

    if not notebooks:
        print("No notebooks found")
        sys.exit(1)

    print(f"\nFound {len(notebooks)} notebooks")
    print(f"Total memory available: {psutil.virtual_memory().available / 1024**3:.1f} GB")

    metrics = []
    for nb_path in notebooks:
        metric = execute_notebook(nb_path)
        metrics.append(metric)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [m for m in metrics if m.get("success", False)]
    failed = [m for m in metrics if not m.get("success", False)]

    print(f"\n✓ Successful: {len(successful)}/{len(metrics)}")
    for m in successful:
        print(f"  {m['notebook']:40s} {m['elapsed_seconds']:6.1f}s {m['memory_delta_mb']:+7.0f}MB")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(metrics)}")
        for m in failed:
            error = m.get("error", "UNKNOWN")
            print(f"  {m['notebook']:40s} {error}")

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_FILE}")

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
