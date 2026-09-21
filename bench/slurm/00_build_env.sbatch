#!/bin/bash
#SBATCH --job-name=tengri-env
#SBATCH --partition=kipac
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --output=/scratch/users/cooray/tengri_bench/logs/%x-%j.out
#SBATCH --error=/scratch/users/cooray/tengri_bench/logs/%x-%j.err

# Build the tengri benchmark venv at <repo>/.venv (the layout the repo's own
# docs and scripts/slurm/env.sh assume; `.venv/` is gitignored, so the
# benchmarked tree stays clean for commit attribution).
#
# One env serves both arms: jax[cuda12] ships the CUDA libs as pip wheels, and
# JAX_PLATFORMS=cpu makes the same interpreter run CPU-only -- which is what
# lets the CPU control run on the *same node* as the GPU, as the brief requires.
#
# Extras are `[gpu,optax]`, NOT `[all]`. `[all]` pulls blackjax>=1.6, which
# requires jax>=0.9.0, and no jax >0.7.0 has a manylinux2014 wheel -- see
# constraints-sherlock.txt. That is reported upstream rather than worked around
# by pinning an older blackjax. It costs shapes E and H; A/B/C/D/G/I are
# unaffected, and C (the mandatory one) touches no sampler.

set -euo pipefail

REPO=/oak/stanford/projects/c4u/u/cooray/tengri
HARNESS=/oak/stanford/projects/c4u/u/cooray/tengri-bench-harness
VENV="$REPO/.venv"
CONSTRAINTS="$HARNESS/slurm/constraints-sherlock.txt"

export PIP_CACHE_DIR=/scratch/users/cooray/pip_cache
export XDG_CACHE_HOME=/scratch/users/cooray/tengri_bench/xdg_cache
export TENGRI_JAX_CACHE_DIR=/scratch/users/cooray/tengri_bench/jax_cache

ml purge
ml devel python/3.12.1

echo "== host $(hostname) ; $(ldd --version | head -1) =="
python3 --version

echo "== benchmarked commit (recorded for attribution) =="
cd "$REPO"
git rev-parse HEAD
git status --porcelain && echo "(tree clean)"

echo "== creating venv at $VENV =="
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

echo "== installing tengri [optax] (NOT [gpu] -- see below) =="
"$VENV/bin/pip" install --prefer-binary -c "$CONSTRAINTS" -e ".[optax]"

# pyproject's [gpu] extra is `jax[cuda12]`, which pulls
# jax-cuda12-plugin[with-cuda] -> nvidia-cudnn-cu12>=9.8. The newest cuDNN
# wheel with a manylinux2014 build is 9.5.0.50, so pip falls to a stub sdist
# and dies in metadata generation. `cuda12-local` is the same plugin WITHOUT
# the bundled nvidia-* wheels: CUDA and cuDNN come from Sherlock's module
# stack instead (cuda/12.8.0 + cudnn/9.14.0.64, which clears the >=9.8 floor).
# The GPU job must therefore `ml load` both before running.
echo "== installing jax[cuda12-local]==0.7.0 (system CUDA, not bundled) =="
"$VENV/bin/pip" install --prefer-binary -c "$CONSTRAINTS" "jax[cuda12-local]==0.7.0"

echo "== versions =="
"$VENV/bin/python" - <<'PY'
import sys
print("python  ", sys.version.split()[0])
for m in ("jax", "jaxlib", "numpy", "scipy", "h5py", "dsps", "optax", "matplotlib", "blackjax"):
    try:
        print(f"{m:11s}", getattr(__import__(m), "__version__", "?"))
    except Exception as e:
        print(f"{m:11s} ABSENT ({type(e).__name__})")
PY

echo "== CUDA plugin wheels actually installed =="
"$VENV/bin/pip" list --format=freeze | grep -Ei "^(jax|nvidia)" || true

echo "== import tengri on CPU =="
JAX_PLATFORMS=cpu "$VENV/bin/python" -c "import jax, tengri; print('OK; devices:', jax.devices())"

echo "== SSP grid: already tracked in the repo, nothing to download =="
ls -l "$REPO/data/fsps_prsc_miles_chabrier.h5"

echo "== DONE =="
