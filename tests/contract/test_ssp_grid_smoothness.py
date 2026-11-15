# SPDX-License-Identifier: BSD-3-Clause
"""Contract: no bundled SSP grid carries a single-cell outlier at 6200 A (#3406).

Each grid is loaded through :func:`load_ssp_data`, so the load-time repair of known
defects is included. A cell at age >= 30 Myr is flagged when it deviates by more than
30% from BOTH its metallicity-neighbor and its age-neighbor geometric means.
"""

import numpy as np
import pytest

from tengri._data_setup import find_ssp_files
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.contract

REF_WAVE_AA = 6200.0
MIN_AGE_YR = 30e6
MAX_DEV = 0.30


def _one_per_name(paths):
    # ``find_ssp_files`` walks every data directory; the first hit for a name wins.
    seen: dict[str, object] = {}
    for path in paths:
        seen.setdefault(path.name, path)
    return [seen[name] for name in sorted(seen)]


_GRID_PATHS = _one_per_name(find_ssp_files())


def _outlier_cells(ssp) -> list[tuple[float, float]]:
    flux = np.asarray(ssp.ssp_flux)
    wave = np.asarray(ssp.ssp_wave)
    lgmet = np.asarray(ssp.ssp_lgmet)
    age_yr = 10.0 ** np.asarray(ssp.ssp_lg_age_gyr) * 1e9
    j = int(np.argmin(np.abs(wave - REF_WAVE_AA)))
    f6 = flux[:, :, j]
    flagged = []
    for k in range(1, f6.shape[0] - 1):
        for i in range(1, f6.shape[1] - 1):
            if age_yr[i] < MIN_AGE_YR:
                continue
            cell = f6[k, i]
            vs_met = cell / np.sqrt(f6[k - 1, i] * f6[k + 1, i])
            vs_age = cell / np.sqrt(f6[k, i - 1] * f6[k, i + 1])
            if abs(vs_met - 1.0) > MAX_DEV and abs(vs_age - 1.0) > MAX_DEV:
                flagged.append((float(lgmet[k]), float(ssp.ssp_lg_age_gyr[i])))
    return flagged


#: Grids that still carry flagged cells after the load-time repair (#3406). Each is an
#: expected failure, strict, so a fix that clears it makes the test fail until the entry goes.
#: Neither is in the repair table: their provenance is not yet checked.
_UNRESOLVED = {
    "bpss_stars_c3k_a_chabrier": "BPASS coarse age grid; 19 flagged cells, not one spike",
    "pgny_mist_c3k_chabrier": "one flagged cell at the lowest metallicity node, 5.6 Gyr",
}


def _grid_param(path):
    if path.stem in _UNRESOLVED:
        return pytest.param(
            path, marks=pytest.mark.xfail(strict=True, reason=_UNRESOLVED[path.stem])
        )
    return pytest.param(path)


@pytest.mark.skipif(not _GRID_PATHS, reason="no SSP grids in the data directory")
@pytest.mark.parametrize("path", [_grid_param(p) for p in _GRID_PATHS], ids=lambda p: p.stem)
def test_loaded_grid_has_no_single_cell_outlier_at_6200(path):
    # Only the flux is checked, so the surviving-mass source does not matter; grids with no
    # shipped surviving-mass table load with the DSPS fit instead of failing.
    ssp = load_ssp_data(str(path), mass_remaining="dsps_fit")
    flagged = _outlier_cells(ssp)
    assert flagged == [], f"{path.stem}: single-cell outliers at 6200 A {flagged}"
