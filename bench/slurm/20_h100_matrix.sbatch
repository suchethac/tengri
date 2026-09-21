#!/bin/bash
#SBATCH --job-name=tengri-h100
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/users/cooray/tengri_bench/logs/%x-%j.out
#SBATCH --error=/scratch/users/cooray/tengri_bench/logs/%x-%j.err

# The whole device matrix for issue #2444, in ONE allocation.
#
# Both arms run here on purpose. The brief is explicit: "a crossover measured
# against a different machine's CPU is not a crossover. Run it on the same
# node." So the CPU control is this H100 node's host CPU, selected per cell via
# JAX_PLATFORMS rather than by a second job on a CPU partition.
#
# Cells are spawned one process each, directly. The script's own --all driver
# is NOT used: it loops over both devices (which one allocation can express,
# but only by holding a jax import at driver scope) and the brief's §5 trap is
# exactly that a driver left on the GPU preallocates 9-10 GiB and then competes
# with its own children for VRAM. Spawning per-cell sidesteps it entirely.
#
# Override the card for a second datacentre point, e.g.:
#   sbatch -C GPU_SKU:L40S --job-name=tengri-l40s \
#          --export=ALL,SKU_TAG=l40s slurm/20_h100_matrix.sbatch

set -uo pipefail
ml purge
ml devel python/3.12.1

# NO cuda/cudnn module loads, deliberately. Probe 44335358 landed on an RTX
# 2080 Ti node whose module tree offers only cuda<=12.4.0 and no cudnn at all:
# Sherlock's trees are architecture-specific, so what `ml spider` shows on the
# login node is not what the allocated node has. The CUDA runtime is installed
# into the venv instead (02_add_cuda_libs.sbatch), which makes this job
# independent of wherever it lands.

: "${SKU_TAG:=h100_sxm5}"

REPO=/oak/stanford/projects/c4u/u/cooray/tengri
HARNESS=/oak/stanford/projects/c4u/u/cooray/tengri-bench-harness
RESULTS=/scratch/users/cooray/tengri_bench/results
mkdir -p "$RESULTS"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

echo "############ allocation ############"
echo "node        : $(hostname)"
echo "slurm job   : ${SLURM_JOB_ID}  partition=${SLURM_JOB_PARTITION}"
echo "cpus/mem    : ${SLURM_CPUS_PER_TASK} / ${SLURM_MEM_PER_NODE} MB"
echo "--- nvidia-smi -L (verbatim, as the brief asks) ---"
nvidia-smi -L
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo "--- CUDA driver / runtime ---"
nvidia-smi | head -4
echo "--- commit under test ---"
cd "$REPO" && git rev-parse HEAD && (git status --porcelain && echo "(tree clean)")
echo "--- what the node's module tree offers (diagnostic only; unused) ---"
(ml devel 2>/dev/null; ml av cuda 2>&1 | head -6; ml math 2>/dev/null; ml av cudnn 2>&1 | head -4) || true
echo "--- CUDA runtime actually in use (from the venv) ---"
"$REPO/.venv/bin/pip" list --format=freeze | grep -i "^nvidia-" | sort
echo "--- jax version in the venv ---"
"$REPO/.venv/bin/python" -c "import jax, jaxlib; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__)"

# Fail fast and loudly. A silent CPU fallback is the one outcome that would
# produce a plausible-looking "GPU" column measured on the host CPU, and the
# brief's first acceptance criterion is precisely that this did not happen.
echo "--- GPU visibility AND real-compute check ---"
if ! JAX_PLATFORMS=cuda "$REPO/.venv/bin/python" -c "
import sys, jax, jax.numpy as jnp
d = jax.devices()
print('devices:', d)
if d[0].platform != 'gpu':
    sys.exit('FATAL: JAX did not find a GPU; refusing to emit CPU rows labelled gpu')
print('device_kind:', d[0].device_kind)
# Visibility is not capability: ptxas missing or a cuDNN too old for the
# plugin both surface only when something actually compiles and runs. Force a
# compile in both precisions before committing to an 8-hour allocation.
x = jnp.ones((512, 512))
print('f32 matmul ok:', float(jax.jit(lambda a: (a @ a).sum())(x)))
with jax.enable_x64():
    y = jnp.ones((512, 512), dtype=jnp.float64)
    print('f64 matmul ok:', float(jax.jit(lambda a: (a @ a).sum())(y)), y.dtype)
print('device of result:', (x @ x).devices())
"; then
    echo "FATAL: CUDA init failed under the module-provided CUDA/cuDNN."
    echo "Nothing was measured. Dumping diagnostics for issue #2444:"
    ldd "$REPO/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" 2>&1 | grep -iE "cudnn|cublas|cudart|not found" | head -20
    exit 1
fi
echo "####################################"

# ---- MANDATORY: shape C, both precisions, GPU then same-node CPU ----------
# Optional shapes follow in the brief's stated priority order (A, B, then the
# rest). E and H are absent: they need blackjax>=1.6, which cannot be installed
# against the jax 0.7.0 that glibc 2.17 caps this cluster at. Reported on #2444.

export SHAPES="C A B D G I"
export PRECISIONS="f64 f32"
export EXTRA_ARGS="--dump-dir /scratch/users/cooray/tengri_bench/arrays"

echo ""
echo "================ GPU ARM (${SKU_TAG}) ================"
DEVICE=gpu TAG="gpu_${SKU_TAG}" \
  OUT_JSON="$RESULTS/device_matrix_gpu_${SKU_TAG}.json" \
  bash "$HARNESS/slurm/_run_cells.sh"

echo ""
echo "================ CPU ARM (same node) ================"
DEVICE=cpu TAG="cpu_${SKU_TAG}node" \
  OUT_JSON="$RESULTS/device_matrix_cpu_${SKU_TAG}node.json" \
  bash "$HARNESS/slurm/_run_cells.sh"

echo ""
echo "############ DONE — results in $RESULTS ############"
ls -la "$RESULTS"
