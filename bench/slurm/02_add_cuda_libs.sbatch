#!/bin/bash
#SBATCH --job-name=tengri-cudalibs
#SBATCH --partition=kipac
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=/scratch/users/cooray/tengri_bench/logs/%x-%j.out
#SBATCH --error=/scratch/users/cooray/tengri_bench/logs/%x-%j.err

# Make the venv self-contained in CUDA, so the GPU run does not depend on the
# node's module tree.
#
# Why: probe 44335358 landed on an RTX 2080 Ti node where `ml av cuda` offers
# only up to cuda/12.4.0 and NO cudnn at all -- Sherlock's module trees are
# architecture-specific, so "cuda/12.8.0 + cudnn/9.14.0.64 exist" (true via
# `ml spider` on the login node) does not mean they exist on the node the job
# lands on. Depending on that is a coin flip resolved only after an 8-hour
# queue wait.
#
# So install the nvidia-* wheels directly. `--no-deps` is what makes it
# possible: it bypasses jax-cuda12-plugin's declared `nvidia-cudnn-cu12>=9.8`,
# which is unsatisfiable here (9.5.0.50 is the last manylinux2014 cuDNN).
# `--only-binary=:all:` is what makes it safe: without it pip happily picks a
# stub sdist for cuDNN and dies in metadata generation, which is how the
# earlier build failed.
#
# The cuDNN shortfall (9.5.0.50 installed vs 9.8 declared) is recorded, not
# hidden. tengri's forward model is elementwise + matmul + interpolation and
# invokes no convolution kernels, so cuDNN is expected to be loaded but never
# called. The GPU job verifies a real matmul and a real forward pass rather
# than trusting that reasoning.

set -euo pipefail

REPO=/oak/stanford/projects/c4u/u/cooray/tengri
VENV="$REPO/.venv"
export PIP_CACHE_DIR=/scratch/users/cooray/pip_cache

ml purge
ml devel python/3.12.1

LIBS=(
  nvidia-cublas-cu12
  nvidia-cuda-cupti-cu12
  nvidia-cuda-nvcc-cu12      # ptxas -- JAX cannot compile without it
  nvidia-cuda-nvrtc-cu12
  nvidia-cuda-runtime-cu12
  nvidia-cudnn-cu12
  nvidia-cufft-cu12
  nvidia-cusolver-cu12
  nvidia-cusparse-cu12
  nvidia-nccl-cu12
  nvidia-nvjitlink-cu12
)

echo "== installing CUDA runtime wheels (newest manylinux2014 of each) =="
"$VENV/bin/pip" install --no-deps --only-binary=:all: "${LIBS[@]}"

echo ""
echo "== what landed =="
"$VENV/bin/pip" list --format=freeze | grep -i "^nvidia-" | sort

echo ""
echo "== ptxas present? (JAX compiles PTX; without this every GPU cell dies) =="
find "$VENV" -name "ptxas" -type f 2>/dev/null | head -3

echo ""
echo "== libcudnn / libcublas in the venv =="
find "$VENV" -name "libcudnn.so*" -o -name "libcublas.so*" 2>/dev/null | head -4

echo ""
echo "== CPU import still healthy? =="
JAX_PLATFORMS=cpu "$VENV/bin/python" -c "import jax, tengri; print('OK', jax.devices())"

echo "== DONE =="
