# SPDX-License-Identifier: BSD-3-Clause
"""Regression #622: dust IR emission must be included in the WavePrecomp /
SpectrumPrecomp photometry projection.

Before the fix, the dust IR re-emission was computed (``L_ir`` published) but
never published as a ``*_phot_lnu_precomp`` family, so ``predict_via_precomp``
omitted it entirely — the far-IR photometry was ~100% wrong under
``approx=WavePrecomp()``. The two-component dust component now publishes
``dust_emission_phot_lnu_precomp`` (the IR template at filter effective
wavelengths, scaled by the energy-balance ``L_ir``), which the projector sums
as an unattenuated additive family.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"


def _ssp_or_skip():
    if not Path(_BARE_SSP).is_file():
        pytest.skip("No bare-stellar SSP grid available under data/.")
    from tengri import load_ssp_data

    return load_ssp_data(_BARE_SSP)


def _ir_obs():
    from tengri import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    # u, IRAC, MIPS24, PACS70, PACS160 (Angstrom) — spans optical → far-IR.
    cs = [3500.0, 36000.0, 240000.0, 700000.0, 1600000.0]
    curves = tuple(
        FilterCurve(
            wave=jnp.linspace(c * 0.85, c * 1.15, 40), trans=jnp.ones(40) * 0.5, name=f"b{int(c)}"
        )
        for c in cs
    )
    return Observation(photometry=Photometry(filters=curves))


@pytest.mark.parametrize("emission", ["dale2014", "draine_li2007", "draine_li2014", "themis"])
def test_dust_ir_in_waveprecomp_photometry(emission):
    """Far-IR photometry under WavePrecomp must match the exact path (was ~100%
    off because dust IR was missing). Tolerance covers the effective-wavelength
    LUT residual (themis's PAH features are the worst case, ~8%)."""
    import warnings

    from tengri import FIXED, Fixed, SEDModel, WavePrecomp

    ssp = _ssp_or_skip()
    obs = _ir_obs()
    groups = dict(
        sfh={"type": "dpl", "*": FIXED},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_diff": 0.5,  # real attenuation → real L_ir to re-emit
            "emission": {"type": emission, "*": FIXED},
        },
        neb={"type": "none"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_exact = SEDModel.build(
            ssp_data=ssp, observation=obs, redshift=Fixed(0.05), approx=None, **groups
        )
        m_lut = SEDModel.build(
            ssp_data=ssp, observation=obs, redshift=Fixed(0.05), approx=WavePrecomp(), **groups
        )

    # The dust-IR LUT family must actually be published (the regression guard:
    # without it, far-IR → ~100% error).
    assert "dust_emission_phot_lnu_precomp" in m_lut.predict_state({}).derived.field_names()

    pe = np.asarray(m_exact.predict_photometry({}))
    pl = np.asarray(m_lut.predict_photometry({}))
    rel = np.abs(pl - pe) / np.maximum(np.abs(pe), 1e-30)
    # Far-IR bands (24/70/160 µm) carry the dust emission — these were ~1.0 before.
    assert rel[2:].max() < 0.10, f"{emission}: far-IR WavePrecomp err {rel[2:].max():.2%}"
    assert rel.max() < 0.10, f"{emission}: max WavePrecomp err {rel.max():.2%}"
