#!/usr/bin/env python3
"""Download the pre-converted Fritz et al. (2006) torus grid (no CIGALE needed).

The Fritz 2006 smooth-dust AGN torus library is distributed upstream only as
~24,000 pcigale-pickled objects bundled inside the ``pcigale`` package — those
pickles cannot even be deserialized without ``pcigale`` importable. To let
end-users use the Fritz torus with zero CIGALE dependency, tengri hosts the
*converted* grid (``fritz2006_torus_grid.h5``, ~28 MB, triweight-ready) on the
same public host as the SSP catalogue.

This script fetches that converted grid into ``data/``. It is the user-facing
counterpart to ``scripts/build_fritz2006_grid.py`` (the developer tool that
*regenerates* the grid from a local CIGALE install).

Usage
-----
::

    python scripts/download_fritz2006_templates.py
    python scripts/download_fritz2006_templates.py --dest /scratch/templates
    python scripts/download_fritz2006_templates.py --force

References
----------
.. [1] O. Fritz, A. Franceschini & E. Hatziminaoglou, "Revisiting the
   infrared spectra of active galactic nuclei with a new torus emission
   model," MNRAS, 366, 767 (2006). https://doi.org/10.1111/j.1365-2966.2005.09866.x
.. [2] M. Boquien et al., "CIGALE: Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). arXiv:1811.03094.
"""

from __future__ import annotations

import argparse
import sys

_GRID_FILENAME = "fritz2006_torus_grid.h5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=None,
        help="Target directory (default: $TENGRI_DATA_DIR or ./data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    args = parser.parse_args()

    try:
        from tengri._data_setup import download_template
    except ImportError:
        print(
            "Error: tengri is not importable. Install tengri (or run from the "
            "repo with PYTHONPATH=src) before fetching templates.",
            file=sys.stderr,
        )
        return 1

    try:
        path = download_template(_GRID_FILENAME, dest=args.dest, force=args.force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Fritz2006 torus grid ready at: {path}")
    print("Use via SEDModel.build(agn={'torus': {'type': 'fritz'}}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
