# SPDX-License-Identifier: BSD-3-Clause
"""Shared fixtures for cross-validation tests against community SED codes.

These tests compare tengri's physics implementations against bagpipes
(Carnall et al. 2018) to verify we produce sensible results. They are
NOT run by default — invoke with:

    pytest -m crossval

External packages (bagpipes, etc.) are imported via pytest.importorskip
so the tests are silently skipped when the dependency is missing.
"""

from pathlib import Path

import numpy as np
import pytest

# ── SSP data paths (needed for tengri SEDModel tests) ────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_PATH = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
SSP_EXISTS = _SSP_PATH.is_file()


@pytest.fixture(scope="session")
def ssp_data():
    """Load SSP data for tengri SEDModel (skip if files missing)."""
    if not SSP_EXISTS:
        pytest.skip("SSP data not found")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_PATH))


@pytest.fixture(scope="session")
def rest_wavelengths():
    """Common rest-frame wavelength grid (Angstrom)."""
    return np.linspace(800, 1400, 200)


@pytest.fixture(scope="session")
def optical_wavelengths():
    """Optical/NIR wavelength grid for dust tests (Angstrom)."""
    return np.linspace(1000, 10000, 500)
