#!/usr/bin/env bash
# Convenience wrapper: download the default tengri SSP grid.
# Usage: bash scripts/setup_ssp.sh [name] [dest]
# Examples:
#   bash scripts/setup_ssp.sh                    # FSPS v3.2 → $TENGRI_DATA_DIR or data/
#   bash scripts/setup_ssp.sh bc03_v3.2 /scratch/ssp

set -euo pipefail

NAME="${1:-fsps_v3.2}"
DEST="${2:-${TENGRI_DATA_DIR:-data}}"

mkdir -p "$DEST"
exec python -c "import tengri; tengri.download_ssp(name='$NAME', dest='$DEST')"
