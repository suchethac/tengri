#!/bin/bash
# Build a virtual environment whose python-fsps is compiled with one isochrone set.
#
# Usage:
#   SPS_HOME=/path/to/fsps-checkout FC=gfortran \
#     scripts/build_fsps_isochrone_env.sh --isoc {parsec|padova|basti} --venv /path/to/venv
#
# SPS_HOME is only recorded (its git commit goes into the table attributes); the Fortran
# compiled is the source bundled in the python-fsps 0.4.7 sdist. FC is the Fortran
# compiler (default: gfortran). The compile flags are written to <venv>/FSPS_BUILD_FLAGS,
# which scripts/build_mass_remaining_fsps.py reads and records.
#
# The isochrone switch is a compile-time choice: MIST is the default set, so only the
# other sets need a flag. The isolated pip build is avoided (it pulls a scikit-build-core
# that rejects cmake.minimum-version), so the build backend is pinned here.
set -euo pipefail

ISOC=""
VENV=""
while [ $# -gt 0 ]; do
  case "$1" in
    --isoc) ISOC="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$ISOC" in
  parsec) FLAGS="-DMIST=0 -DPARSEC=1 -DMILES=1" ;;
  padova) FLAGS="-DMIST=0 -DPADOVA=1 -DMILES=1" ;;
  basti)  FLAGS="-DMIST=0 -DBASTI=1 -DMILES=1" ;;
  *) echo "--isoc must be one of parsec, padova, basti" >&2; exit 2 ;;
esac
[ -n "$VENV" ] || { echo "--venv PATH is required" >&2; exit 2; }
: "${SPS_HOME:?set SPS_HOME to the python-fsps source checkout}"
FC="${FC:-gfortran}"
export SPS_HOME FC

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q numpy h5py "scikit-build-core[pyproject]<0.8" setuptools \
  setuptools_scm cmake ninja
FFLAGS="$FLAGS" "$VENV/bin/pip" install --no-cache-dir --no-build-isolation \
  --no-binary fsps "fsps==0.4.7" > "$VENV/build.log" 2>&1
printf '%s\n' "$FLAGS" > "$VENV/FSPS_BUILD_FLAGS"

"$VENV/bin/python" -c "
import fsps
sp = fsps.StellarPopulation(zcontinuous=0)
print('libraries', sp.libraries)
print('version', fsps.__version__, fsps.__file__)
"
