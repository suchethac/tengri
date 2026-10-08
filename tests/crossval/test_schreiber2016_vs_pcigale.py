# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate ``schreiber2016`` against CIGALE's ``schreiber2016`` module (#2597).

CIGALE ships the Schreiber et al. (2018, A&A 609, A30) dust library under the
module name ``schreiber2016``: per dust temperature, a dust-continuum and a PAH
template for one kilogram of dust, mixed ``(1 - fpah) dust + fpah pah`` and
renormalized to the absorbed power. tengri's ``schreiber2016`` is the same
library repackaged (``data/schreiber2016_templates.h5``), so its SED must
reproduce the installed module's, band by band, at every (T, f_PAH).

Skipped unless ``pcigale`` is installed (``pytest -m crossval``).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

pytest.importorskip("pcigale")

_H5 = Path(__file__).resolve().parents[2] / "data" / "schreiber2016_templates.h5"
if not _H5.is_file():
    pytest.skip(f"template file {_H5} not available", allow_module_level=True)

_C_AA = 2.99792458e18  # speed of light [Angstrom/s]
_BANDS_UM = ((3, 8), (8, 24), (24, 70), (70, 160), (160, 500), (500, 1000))


def _pcigale_lnu(T: float, f: float) -> tuple[np.ndarray, np.ndarray]:
    """CIGALE's dust SED for 1 W absorbed: ``(wavelength_aa, L_nu up to a constant)``."""
    from pcigale.sed import SED
    from pcigale.sed_modules import get_module

    module = get_module("schreiber2016", tdust=T, fpah=f)
    sed = SED()
    sed.add_info("dust.luminosity", 1.0, True, unit="W")
    module.process(sed)
    wave_aa = np.asarray(sed.wavelength_grid) * 10.0
    return wave_aa, np.asarray(sed.luminosity) * wave_aa**2


def _band_fractions(w: np.ndarray, lnu: np.ndarray) -> np.ndarray:
    def power(lo: float, hi: float) -> float:
        m = (w >= lo * 1e4) & (w <= hi * 1e4)
        return float(-np.trapezoid(lnu[m], _C_AA / w[m]))

    return np.array([power(a, b) / power(1.0, 1000.0) for a, b in _BANDS_UM])


@pytest.mark.parametrize(("T", "f"), [(20.0, 0.05), (35.0, 0.2), (50.0, 0.5), (24.0, 0.9)])
def test_band_fractions_match_the_installed_cigale_module(T, f):
    """Band fractions of the 1 um - 1 mm power agree with CIGALE to 2e-3."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS

    w, ref = _pcigale_lnu(T, f)
    ours = np.asarray(
        DUST_EMISSION_MODELS["schreiber2016"](jnp.asarray(w), 1.0, dust_T=T, dust_f_pah=f)
    )
    np.testing.assert_allclose(
        _band_fractions(w, ours), _band_fractions(w, ref), rtol=2e-3, err_msg=f"T={T} f_pah={f}"
    )


def test_f_pah_default_is_cigales():
    """CIGALE's declared ``fpah`` default equals the registry ``dust_f_pah`` default."""
    from pcigale.sed_modules.schreiber2016 import Schreiber2016

    from tengri.components.dust._params import DEFAULT_DUST_F_PAH

    assert Schreiber2016.parameters["fpah"][2] == DEFAULT_DUST_F_PAH
