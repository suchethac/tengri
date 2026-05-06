#!/bin/bash
# Execute every spine notebook in-place so outputs embed for nbsphinx rendering.
# Runs from project root. Per-notebook timeout 600s (10 min).
set -u
cd /Users/suchethacooray/Projects/tengri
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
LOG=/tmp/execute_spine.log
: > "$LOG"

declare -a NOTEBOOKS=(
    "00_quickstart"
    "01_why_jax"
    "02_sed_anatomy"
    "03_fitting_photometry"
    "04_fitting_spectra"
    "05_joint_photometry_spectroscopy"
    "06_inference_methods"
    "07_degeneracies"
    "08_sfh_advanced"
    "09_dust_emission"
    "10_agn_advanced"
    "11_population"
    "12_diagnostics"
    "13_extending_tengri"
    "14_stochastic_sfh"
    "15_vi_inference"
    "16_simulation_interface"
    "17_emission_line_measurements"
)

for nb in "${NOTEBOOKS[@]}"; do
    src="notebooks/${nb}.ipynb"
    dst="docs/spine/${nb}.ipynb"
    if [ ! -f "$src" ]; then
        # Sync from .py first
        .venv/bin/jupytext --sync "notebooks/${nb}.py" 2>/dev/null
    fi
    if [ ! -f "$src" ]; then
        echo "SKIP ${nb} (no .ipynb after sync)" | tee -a "$LOG"
        continue
    fi
    cp "$src" "$dst"
    echo "EXEC ${nb}..." | tee -a "$LOG"
    # Use nbclient directly with timeout
    .venv/bin/python -c "
import nbformat, sys
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import signal

def timeout_handler(signum, frame):
    raise TimeoutError('notebook too slow')

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(600)

path = '${dst}'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=540, allow_errors=True, kernel_name='python3')
try:
    client.execute()
    nbformat.write(nb, path)
    print('OK ${nb}')
except TimeoutError:
    nbformat.write(nb, path)
    print('TIMEOUT ${nb}')
except Exception as e:
    print(f'ERROR ${nb}: {type(e).__name__}: {e}')
finally:
    signal.alarm(0)
" 2>&1 | tee -a "$LOG" | grep -E "OK|TIMEOUT|ERROR" | head -1
done
echo "DONE"
