#!/bin/bash
# Catalog throughput — PROCESS-PER-GPU (the primary path; scales across nodes).
#
# An array job: each task takes one GPU and fits its slice of the catalog with
# vectorized per-galaxy sampling. SLURM spreads the tasks over whatever GPU
# nodes are free, so this scales past a single node. Each task writes one
# shard_NNNNN.npz; merge them afterwards with merge_shards.py.
#
# Set the array range to the number of slices you want (= number of parallel
# GPU tasks). `%8` caps concurrent tasks at 8 — raise/lower for your allocation.
#
#   CATALOG=cat.npz MODEL_BUILDER=myfit:build_model OUTDIR=out \
#       sbatch --array=0-15%8 fit_catalog_array.sbatch
#   python merge_shards.py --shards out --out out/catalog_posteriors.npz
#
#SBATCH --job-name=tengri-catalog
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4     # pick your GPU; `node_feat -p gpu | grep GPU_`
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-15%8                      # 16 slices, <=8 concurrent (override on the CLI)
#SBATCH --output=logs/catalog_%A_%a.out

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

# n_slices = size of the array. SLURM sets SLURM_ARRAY_TASK_COUNT; the runner
# reads it (and SLURM_ARRAY_TASK_ID) automatically, so no --n-slices needed.
python "$HERE/run_catalog_slice.py" \
    --catalog "$CATALOG" --model-builder "$MODEL_BUILDER" --out "$OUTDIR" \
    --method "$METHOD" --chunk "$CHUNK" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES"
