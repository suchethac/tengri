#!/bin/bash
#SBATCH --job-name=tengri-probe
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=/scratch/users/cooray/tengri_bench/logs/%x-%j.out
#SBATCH --error=/scratch/users/cooray/tengri_bench/logs/%x-%j.err

# Does the module stack actually provide CUDA + cuDNN on a GPU node?
#
# This decides the install strategy. glibc 2.17 caps the pip `nvidia-cudnn-cu12`
# wheel at 9.5.0.50, but jax-cuda12-plugin 0.7.0's `with-cuda` extra requires
# >=9.8 -- so the bundled-CUDA route is dead. The `cuda12-local` extra instead
# takes CUDA from the system, which is only viable if these modules exist here.

echo "=== node: $(hostname) ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

echo ""
echo "=== available cuda / cudnn modules on THIS node ==="
ml purge 2>/dev/null
ml devel 2>/dev/null
ml av cuda 2>&1 | tail -12
ml math 2>/dev/null
ml av cudnn 2>&1 | tail -8

echo ""
echo "=== loading cuda/12.8.0 + cudnn/9.14.0.64 ==="
ml load cuda/12.8.0 2>&1 | tail -3
ml load cudnn/9.14.0.64 2>&1 | tail -3
ml list 2>&1 | tail -10

echo ""
echo "CUDA_HOME=[${CUDA_HOME:-unset}]"
ls "${CUDA_HOME:-/nonexistent}/lib64/" 2>/dev/null | grep -E "libcudart|libcublas|libcusolver" | head -5

echo ""
echo "=== libcudnn.so anywhere on LD_LIBRARY_PATH? ==="
for d in $(tr ':' '\n' <<< "$LD_LIBRARY_PATH"); do
  ls "$d"/libcudnn.so* 2>/dev/null | head -1
done | head -3

echo ""
echo "=== nvcc ==="
which nvcc 2>/dev/null && nvcc --version | tail -2

echo "=== PROBE DONE ==="
