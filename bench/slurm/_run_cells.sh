#!/bin/bash
# Run a set of benchmark_device_matrix cells, one child process each, and
# collect the __ROW__ JSON lines into a single array.
#
# The script's own --all driver loops over BOTH devices, which a Slurm
# allocation cannot honour (a CPU-partition job has no GPU). So the cells are
# spawned here instead, with the same per-cell isolation --all uses: its own
# JAX_PLATFORMS, its own x64 flag, and its own precompute cache (that cache is
# keyed on neither dtype nor backend, so a shared one lets f32 contaminate f64).
#
# Required env: DEVICE (cpu|gpu), PRECISIONS, SHAPES, OUT_JSON, TAG
set -uo pipefail

REPO=/oak/stanford/projects/c4u/u/cooray/tengri
VENV="$REPO/.venv"
PY="$VENV/bin/python"
SCRIPT="$REPO/bench/scripts/benchmark_device_matrix.py"
WORK=/scratch/users/cooray/tengri_bench

: "${DEVICE:?set DEVICE}"
: "${PRECISIONS:?set PRECISIONS}"
: "${SHAPES:?set SHAPES}"
: "${OUT_JSON:?set OUT_JSON}"
: "${TAG:=untagged}"
: "${EXTRA_ARGS:=}"

mkdir -p "$(dirname "$OUT_JSON")"
ROWS_DIR="$WORK/rows/${TAG}"
rm -rf "$ROWS_DIR"; mkdir -p "$ROWS_DIR"

cd "$REPO"

echo "=== host: $(hostname)  device=$DEVICE  tag=$TAG ==="
echo "--- cpu ---"; lscpu | grep -E "^Model name|^CPU\(s\)|^Socket|^Thread" || true
echo "--- allocation ---"; echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?}"
if [ "$DEVICE" = "gpu" ]; then
  echo "--- gpu ---"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true
fi

# Shape is the OUTER loop so both precisions of the mandatory shape C land
# before any optional shape starts. If the allocation dies early, the cells the
# brief actually requires are the ones already on disk.
for SHAPE in $SHAPES; do
  for PREC in $PRECISIONS; do
    echo ""
    echo ">>> cell device=$DEVICE precision=$PREC shape=$SHAPE"

    ENVS=(
      "JAX_PLATFORMS=$([ "$DEVICE" = gpu ] && echo cuda || echo cpu)"
      "XLA_PYTHON_CLIENT_PREALLOCATE=false"
      "TENGRI_NO_BACKGROUND_COMPILE=1"
      "TENGRI_PRECOMP_CACHE_DIR=$WORK/precomp_${DEVICE}_${PREC}"
      "TENGRI_JAX_CACHE_DIR=$WORK/jax_cache_${DEVICE}_${PREC}"
      "XDG_CACHE_HOME=$WORK/xdg_cache"
      "HOME=$HOME"
    )
    if [ "$PREC" = "f32" ]; then
      ENVS+=("JAX_ENABLE_X64=0")
    fi

    CELL_ARGS=(--shape "$SHAPE" --precision "$PREC" --emit-json)
    # shellcheck disable=SC2206
    CELL_ARGS+=($EXTRA_ARGS)

    OUT_FILE="$ROWS_DIR/${DEVICE}_${PREC}_${SHAPE}.json"
    LOG_FILE="$ROWS_DIR/${DEVICE}_${PREC}_${SHAPE}.log"

    start=$(date +%s)
    env "${ENVS[@]}" "$PY" "$SCRIPT" "${CELL_ARGS[@]}" >"$LOG_FILE" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    if grep -q '^__ROW__' "$LOG_FILE"; then
      grep '^__ROW__' "$LOG_FILE" | head -1 | sed 's/^__ROW__//' > "$OUT_FILE"
      echo "    ok (${elapsed}s): $(head -c 200 "$OUT_FILE")"
    else
      echo "    FAILED rc=$rc (${elapsed}s). Last lines:"
      tail -15 "$LOG_FILE" | sed 's/^/      /'
      "$PY" - "$DEVICE" "$PREC" "$SHAPE" "$LOG_FILE" "$OUT_FILE" <<'PY'
import json, sys
device, prec, shape, log, out = sys.argv[1:6]
tail = [l.rstrip() for l in open(log, errors="replace").read().splitlines() if l.strip()][-5:]
json.dump({"shape": shape, "device": device, "precision": prec,
           "error": " | ".join(tail)[:600]}, open(out, "w"))
PY
    fi
  done
done

echo ""
echo "=== merging rows -> $OUT_JSON ==="
"$PY" - "$ROWS_DIR" "$OUT_JSON" "$TAG" <<'PY'
import json, pathlib, sys, socket, subprocess, datetime
rows_dir, out_json, tag = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for p in sorted(pathlib.Path(rows_dir).glob("*.json")):
    try:
        rows.append(json.load(open(p)))
    except Exception as e:
        rows.append({"error": f"unreadable {p.name}: {e}"})
meta = {"tag": tag, "host": socket.gethostname(),
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job": __import__("os").environ.get("SLURM_JOB_ID")}
try:
    meta["gpu"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10
    ).stdout.strip()
except Exception:
    meta["gpu"] = None
try:
    meta["cpu"] = [l.split(":",1)[1].strip() for l in open("/proc/cpuinfo")
                   if l.startswith("model name")][0]
except Exception:
    meta["cpu"] = None
json.dump({"meta": meta, "rows": rows}, open(out_json, "w"), indent=2)
ok = sum(1 for r in rows if "error" not in r)
print(f"wrote {len(rows)} rows ({ok} ok, {len(rows)-ok} failed) to {out_json}")
PY

echo "=== DONE $TAG ==="
