# SPDX-License-Identifier: BSD-3-Clause
"""Contract: SpectrumPrecomp must include the AGN contribution in predict_spectrum.

``AGNSEDComponent`` previously published only its *photometry* LUT family
(``agn_phot_lnu_precomp``), never the *spectrum* one. ``predict_spectrum_via_precomp``
sums whatever ``*_spec_lnu_precomp`` families exist, so under
``approx=SpectrumPrecomp()`` the AGN was **silently dropped** from
``predict_spectrum`` (up to ~87 % flux deficit in AGN-dominated bands). The fix
publishes ``agn_spec_lnu_precomp`` by point-sampling the rest-frame AGN SED at the
pixel wavelengths — exact, since a spectrum pixel is a single wavelength (mirrors
radio / X-ray). Runs on the synthetic wide SSP with an analytic disc — no
``data/`` grids needed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, SEDModel, SpectrumPrecomp
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract

_WAVE = jnp.logspace(np.log10(3000.0), np.log10(50000.0), 200)  # optical → MIR, observed Å


def _build(ssp, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=Spectroscopy(wave_obs=_WAVE)),
        approx=approx,
        redshift=Fixed(0.5),
        igm={"type": "none"},
        sfh={"type": "tsnorm", "all_params": FIXED, "log_total_mass": Fixed(10.0)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "log_lbol": Fixed(12.0),
        },
    )


def test_spectrum_precomp_publishes_nonzero_agn_spec_family(synthetic_ssp_wide):
    """The spectrum LUT family must be published AND carry real AGN flux.

    This is the precise regression for the bug: ``AGNSEDComponent`` published
    only ``agn_phot_lnu_precomp``, so ``predict_spectrum_via_precomp`` (which
    sums whatever ``*_spec_lnu_precomp`` families exist) silently omitted the
    AGN. A missing or all-zero family means the AGN is dropped from the
    spectrum LUT.
    """
    m = _build(synthetic_ssp_wide, SpectrumPrecomp())
    sf = {**m.spec.get_fixed_values()}
    derived = m.predict_state(sf).derived
    assert "agn_spec_lnu_precomp" in derived.field_names()
    fam = derived.get("agn_spec_lnu_precomp")
    assert fam is not None
    fam = np.asarray(fam)
    assert fam.shape == (_WAVE.shape[0],)
    assert np.all(np.isfinite(fam))
    assert np.any(fam > 0.0), "AGN spectrum LUT family is all-zero — AGN dropped"


def test_spectrum_precomp_matches_exact_with_agn(synthetic_ssp_wide):
    """predict_spectrum under SpectrumPrecomp reproduces the exact path with AGN.

    The published AGN family must equal the exact-path AGN contribution
    (point-sampling a smooth AGN SED is exact), so the LUT spectrum tracks the
    exact spectrum to well under a percent — and crucially the LUT family alone
    matches the AGN's exact-path delta (caught vacuity guard: the family is
    non-zero per the test above).
    """
    sf = {**_build(synthetic_ssp_wide, None).spec.get_fixed_values()}
    s_exact = np.asarray(_build(synthetic_ssp_wide, None).predict_spectrum(sf))
    s_lut = np.asarray(_build(synthetic_ssp_wide, SpectrumPrecomp()).predict_spectrum(sf))
    ratio = s_lut / np.where(s_exact == 0, np.nan, s_exact)
    worst = float(np.nanmax(np.abs(ratio - 1.0)))
    assert worst < 0.01, f"AGN spectrum LUT diverges from exact: worst |ratio-1| = {worst}"
