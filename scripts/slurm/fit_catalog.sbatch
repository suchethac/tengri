#!/bin/bash
# Fit a catalog of galaxies in parallel on ONE GPU with NUTS.
#
# The simple case: one GPU, one process, the whole catalog. CatalogFitter
# vmaps `forward_chunk_size` galaxies per step so they sample simultaneously on
# the GPU. Bump CHUNK until you fill the card. Writes one shard_00000.npz of
# per-galaxy posterior summaries.
#
#   CATALOG=cat.npz MODEL_BUILDER=myfit:build_model OUTDIR=out \
#       CHUNK=256 sbatch fit_catalog.sbatch
#
# For a catalog too big for one GPU / one time-limit, use fit_catalog_array.sbatch.
#
#SBATCH --job-name=tengri-catalog
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4     # pick your GPU; `node_feat -p gpu | grep GPU_`
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/catalog_%j.out

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

python "$HERE/run_catalog_slice.py" \
    --catalog "$CATALOG" --model-builder "$MODEL_BUILDER" --out "$OUTDIR" \
    --slice 0 --n-slices 1 \
    --method "$METHOD" --chunk "$CHUNK" --n-warmup "$N_WARMUP" --n-samples "$N_SAMPLES"
