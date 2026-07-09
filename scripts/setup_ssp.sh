#!/usr/bin/env bash
# Convenience wrapper: download the default tengri SSP grid.
# Usage: bash scripts/setup_ssp.sh [name] [dest]
# Names are the short keys from tengri.list_known_ssps().
# Examples:
#   bash scripts/setup_ssp.sh                    # default grid → $TENGRI_DATA_DIR or data/
#   bash scripts/setup_ssp.sh bc03_pdva_stelib_chabrier /scratch/ssp

set -euo pipefail

NAME="${1:-fsps_prsc_miles_chabrier}"
DEST="${2:-${TENGRI_DATA_DIR:-data}}"

mkdir -p "$DEST"
exec python -c "import tengri; tengri.download_ssp(name='$NAME', dest='$DEST')"
