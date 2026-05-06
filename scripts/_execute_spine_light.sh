#!/bin/bash
# Execute LIGHT spine notebooks only so outputs render in nbsphinx.
# Skip heavy inference notebooks (03-06, 11, 14-15) that take 10+ minutes each.
# Per-notebook timeout 300s.
set -u
cd /Users/suchethacooray/Projects/tengri
export JAX_PLATFORMS=cpu
# NOTE: do NOT set MPLBACKEND=Agg here — it stops the ipykernel inline backend
# from capturing figures as image/png outputs in the .ipynb (which nbsphinx
# renders as <img>).
LOG=/tmp/execute_spine.log
: > "$LOG"

declare -a NOTEBOOKS=(
    "00_quickstart"
    "01_why_jax"
    "02_sed_anatomy"
    "07_degeneracies"
    "08_sfh_advanced"
    "09_dust_emission"
    "10_agn_advanced"
    "12_diagnostics"
    "13_extending_tengri"
    "16_simulation_interface"
    "17_emission_line_measurements"
)

for nb in "${NOTEBOOKS[@]}"; do
    src="notebooks/${nb}.ipynb"
    dst="docs/spine/${nb}.ipynb"
    if [ ! -f "$src" ]; then
        .venv/bin/jupytext --sync "notebooks/${nb}.py" 2>/dev/null
    fi
    cp "$src" "$dst" 2>/dev/null || { echo "SKIP ${nb} (no src)" | tee -a "$LOG"; continue; }
    echo "EXEC ${nb}..." | tee -a "$LOG"
    .venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb_path = '${dst}'
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=240, allow_errors=True, kernel_name='python3')
try:
    client.execute()
    nbformat.write(nb, nb_path)
    print('OK ${nb}')
except Exception as e:
    nbformat.write(nb, nb_path)
    print(f'FAIL ${nb}: {type(e).__name__}')
" 2>&1 | grep -E "^OK|^FAIL" | head -1 | tee -a "$LOG"
done
echo "DONE"
