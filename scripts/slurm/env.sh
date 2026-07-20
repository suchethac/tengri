#!/usr/bin/env bash
# Shared environment for tengri SLURM jobs — sourced by every *.sbatch here.
#
# Edit TENGRI_REPO / TENGRI_VENV for your checkout, then export the job inputs
# (CATALOG, MODEL_BUILDER, OUTDIR, ...) before `sbatch`. Everything is
# overridable from the environment so the same scripts serve every catalog.

set -euo pipefail

# --- your checkout (edit these, or export before submitting) ---------------
export TENGRI_REPO="${TENGRI_REPO:-$HOME/tengri}"
export TENGRI_VENV="${TENGRI_VENV:-$TENGRI_REPO/.venv}"

# --- persistent caches on scratch (shared across all array tasks) ----------
# The JAX compile cache is the big win on a cluster: the first task compiles
# the forward + NUTS kernels, every later task loads them in milliseconds.
export TENGRI_JAX_CACHE_DIR="${TENGRI_JAX_CACHE_DIR:-$SCRATCH/tengri_jax_cache}"
export TENGRI_PRECOMP_CACHE_DIR="${TENGRI_PRECOMP_CACHE_DIR:-$SCRATCH/tengri_precomp}"

# --- GPU memory: allocate on demand, leave headroom for cuDNN --------------
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

# --- job inputs (export these before sbatch; shown here as the contract) ----
: "${CATALOG:?set CATALOG=/path/to/catalog.npz (flux_obs (N,B), noise (N,B))}"
: "${MODEL_BUILDER:?set MODEL_BUILDER=package.module:function returning an SEDModel}"
: "${OUTDIR:?set OUTDIR=/path/for/output/shards}"
export METHOD="${METHOD:-mcmc_nuts}"
export CHUNK="${CHUNK:-64}"
export N_WARMUP="${N_WARMUP:-300}"
export N_SAMPLES="${N_SAMPLES:-1000}"

mkdir -p "$TENGRI_JAX_CACHE_DIR" "$TENGRI_PRECOMP_CACHE_DIR" "$OUTDIR" logs

# shellcheck disable=SC1091
source "$TENGRI_VENV/bin/activate"
cd "$TENGRI_REPO"

echo "env: repo=$TENGRI_REPO jax_cache=$TENGRI_JAX_CACHE_DIR"
echo "env: catalog=$CATALOG builder=$MODEL_BUILDER out=$OUTDIR"
echo "env: method=$METHOD chunk=$CHUNK n_warmup=$N_WARMUP n_samples=$N_SAMPLES"
nvidia-smi -L 2>/dev/null || echo "env: nvidia-smi unavailable (CPU node?)"
