#!/bin/bash
#SBATCH --job-name=tengri-h100b
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/users/cooray/tengri_bench/logs/%x-%j.out
#SBATCH --error=/scratch/users/cooray/tengri_bench/logs/%x-%j.err

# Follow-up to 44335875. Three jobs in one allocation:
#
#   1. Stack verification (issue #2444): device matmul recorded as a JSON
#      field, and the compiled HLO scanned for cuDNN custom calls.
#   2. Shape C extended past 2048, to find where the H100 finally leaves the
#      ~13.6 ms launch-latency floor and the per-galaxy curve flattens. The
#      first run showed per-CALL wall time constant to within 3% from batch 1
#      to batch 2048, so the whole default sweep sits in the 1/n regime and
#      acceptance criterion 3 ("falls then flattens") cannot be evaluated on it.
#   3. Shape I with chunk sizes that actually fill the card; the default
#      --chunk-size 1000 pays the latency floor every 1000 galaxies.
#
# Module CUDA on purpose, matching 44335875 exactly (cuda/12.8.0 +
# cudnn/9.14.0.64). 03_restore_module_cuda.sbatch removes the pip nvidia-*
# wheels first, because those shadow the modules and would put this run on
# cuDNN 9.5.0.50 -- a different stack from the one being extended.

set -uo pipefail
ml purge
ml devel python/3.12.1
ml load cuda/12.8.0
ml math
ml load cudnn/9.14.0.64

REPO=/oak/stanford/projects/c4u/u/cooray/tengri
HARNESS=/oak/stanford/projects/c4u/u/cooray/tengri-bench-harness
RESULTS=/scratch/users/cooray/tengri_bench/results
WORK=/scratch/users/cooray/tengri_bench
PY="$REPO/.venv/bin/python"
mkdir -p "$RESULTS"

export TENGRI_REPO="$REPO"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

echo "############ allocation ############"
echo "node   : $(hostname)   job ${SLURM_JOB_ID}"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
cd "$REPO" && git rev-parse HEAD && (git status --porcelain && echo "(tree clean)")
echo "CUDA_HOME=[${CUDA_HOME:-unset}]"
ml list 2>&1 | tail -6
echo "-- pip nvidia wheels must be ABSENT for the module stack to be in use --"
"$REPO/.venv/bin/pip" list --format=freeze | grep -i "^nvidia-" || echo "(none -- correct)"
echo "####################################"

# ---- 1. stack verification -----------------------------------------------
echo ""
echo "================ STACK VERIFICATION ================"
DUMP=$WORK/hlo_dump_$SLURM_JOB_ID
rm -rf "$DUMP"; mkdir -p "$DUMP"
# Set BEFORE python starts: XLA reads the dump flag at backend init.
XLA_FLAGS="--xla_dump_to=$DUMP" \
JAX_PLATFORMS=cuda \
TENGRI_PRECOMP_CACHE_DIR=$WORK/precomp_verify \
  "$PY" "$HARNESS/verify_stack.py" \
      --dump-dir "$DUMP" \
      --out "$RESULTS/stack_verification_h100.json"
echo "verify exit: $?"

# ---- 2 & 3. extended sweeps ----------------------------------------------
export SHAPES="C"
export PRECISIONS="f64 f32"
export EXTRA_ARGS="--batches 4096 8192 16384 32768 65536 131072"

echo ""
echo "================ GPU: shape C extended ================"
DEVICE=gpu TAG="gpu_h100_sxm5_ext" \
  OUT_JSON="$RESULTS/device_matrix_gpu_h100_sxm5_ext.json" \
  bash "$HARNESS/slurm/_run_cells.sh"

echo ""
echo "================ CPU (same node): shape C extended ================"
# 131072 x f64 on 8 cores is slow; the sweep stops itself on OOM/err per cell.
DEVICE=cpu TAG="cpu_h100_sxm5node_ext" \
  OUT_JSON="$RESULTS/device_matrix_cpu_h100_sxm5node_ext.json" \
  bash "$HARNESS/slurm/_run_cells.sh"

export SHAPES="I"
export EXTRA_ARGS="--chunk-size 1000 8192 32768 131072 --n-total 1000000"

echo ""
echo "================ GPU: shape I, chunks that fill the card ================"
DEVICE=gpu TAG="gpu_h100_sxm5_thr" \
  OUT_JSON="$RESULTS/device_matrix_gpu_h100_sxm5_thr.json" \
  bash "$HARNESS/slurm/_run_cells.sh"

echo ""
echo "================ CPU (same node): shape I ================"
DEVICE=cpu TAG="cpu_h100_sxm5node_thr" \
  OUT_JSON="$RESULTS/device_matrix_cpu_h100_sxm5node_thr.json" \
  bash "$HARNESS/slurm/_run_cells.sh"

echo ""
echo "############ DONE ############"
ls -la "$RESULTS"
