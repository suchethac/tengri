#!/bin/bash
#SBATCH --job-name=tengri-figs
#SBATCH --partition=kipac
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=/scratch/users/cooray/tengri_bench/logs/%x-%j.out
#SBATCH --error=/scratch/users/cooray/tengri_bench/logs/%x-%j.err

# Render the figures. CPU-only and JAX-free by design: the measurement happens
# inside a GPU allocation, rendering must not need one.

set -uo pipefail
ml purge
ml devel python/3.12.1

HARNESS=/oak/stanford/projects/c4u/u/cooray/tengri-bench-harness
PY=/oak/stanford/projects/c4u/u/cooray/tengri/.venv/bin/python

JAX_PLATFORMS=cpu MPLCONFIGDIR=/scratch/users/cooray/tengri_bench/mplconfig \
  "$PY" "$HARNESS/plot_device_matrix.py" \
    --results-dir /scratch/users/cooray/tengri_bench/results \
    --out-dir "$HARNESS/figures"

echo "=== figures ==="
ls -la "$HARNESS/figures"
