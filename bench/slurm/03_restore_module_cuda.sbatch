#!/bin/bash
#SBATCH --job-name=tengri-restore
#SBATCH --partition=kipac
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --output=/scratch/users/cooray/tengri_bench/logs/%x-%j.out
#SBATCH --error=/scratch/users/cooray/tengri_bench/logs/%x-%j.err

# Restore the venv to the exact stack that produced the measured dataset.
#
# Job 44335875 ran BEFORE 02_add_cuda_libs.sbatch, so it used the H100 node's
# module CUDA (cuda/12.8.0 + cudnn/9.14.0.64). The pip nvidia-* wheels
# installed afterwards sit in site-packages and take precedence, which would
# silently run any follow-up on cuDNN 9.5.0.50 instead -- a different stack
# from the one the published numbers were taken on.
#
# A follow-up that extends the same curve has to be on the same stack, or the
# extension is not an extension. So the wheels come back out; the H100 node
# supplies CUDA, as it did for the measurement.

set -euo pipefail
VENV=/oak/stanford/projects/c4u/u/cooray/tengri/.venv

ml purge
ml devel python/3.12.1

echo "== before =="
"$VENV/bin/pip" list --format=freeze | grep -i "^nvidia-" | sort || echo "(none)"

echo ""
echo "== uninstalling pip CUDA wheels =="
"$VENV/bin/pip" uninstall -y \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 \
  nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 2>&1 | tail -20

echo ""
echo "== after (should be empty) =="
"$VENV/bin/pip" list --format=freeze | grep -i "^nvidia-" | sort || echo "(none -- correct)"

echo ""
echo "== jax intact? =="
"$VENV/bin/pip" list --format=freeze | grep -iE "^jax"
JAX_PLATFORMS=cpu "$VENV/bin/python" -c "import jax, tengri; print('OK', jax.devices())"
echo "== DONE =="
