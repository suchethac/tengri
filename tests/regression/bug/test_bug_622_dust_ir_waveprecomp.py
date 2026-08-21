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
    off because dust IR was missing). The IR re-emission is now integrated
    through the true filter transmission (not sampled at the effective
    wavelength), so the IR bands are essentially exact — even THEMIS's PAH
    features, which the effective-wavelength sampling missed by ~8%."""
    import warnings

    from tengri import FIXED, Fixed, SEDModel, WavePrecomp

    ssp = _ssp_or_skip()
    obs = _ir_obs()
    groups = dict(
        sfh={"type": "dpl", "*": FIXED},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "*": FIXED,
            "tau_diff": 0.5,  # real attenuation → real L_ir to re-emit
        },
        dust_emission={"type": emission, "*": FIXED},
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
    # IR bands (IRAC/24/70/160 µm) carry the dust emission — were ~1.0 before
    # the family was published, ~8% under effective-wavelength sampling, and are
    # now exact (filter-integrated): MIPS24/PACS ~0%, IRAC ~0.1%.
    assert rel[1:].max() < 0.01, f"{emission}: IR WavePrecomp err {rel[1:].max():.2%}"
    # u-band (index 0) is attenuated stellar — the effective-wavelength dust
    # attenuation residual, not dust emission. Two-component dust now applies the
    # same first-order Taylor (Ψ) correction as single-component (#617, on by
    # default); on this feature-rich wNE SSP the first-order term slightly
    # overshoots in the steep UV (4000 Å + Balmer breaks + baked nebular lines),
    # giving ~3% — still far below SSP/dust systematics, and the test's focus
    # (dust IR in the bands above) is unaffected.
    assert rel.max() < 0.04, f"{emission}: max WavePrecomp err {rel.max():.2%}"


@pytest.mark.parametrize(
    "filters",
    [
        ["sdss_z"],
        ["sdss_z", "sdss_i"],
        ["sdss_i", "sdss_z"],
        ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    ],
)
def test_dust_ir_optical_reddest_band_not_inflated(filters):
    """Second #622 bug: an *all-optical* filter set has no far-IR band to anchor
    the IR-template normalization, so the original ``_em_fn(filter_eff, L_ir)``
    call renormalized L_ir over a handful of optical pivots and inflated the
    reddest band by ~4× (293% on ``sdss_z``). The failure was order-dependent —
    fine when the red band was alone or first, broken when it followed another —
    because a single pivot integrates to 0 (silently zeroing the band) while two
    optical pivots give a tiny bogus integral that blows up.

    Dust IR re-emission is negligible (≈0) across an optical bandpass, so every
    band must match the exact path. The fix samples the dense, correctly
    normalized ``sed_ir`` at the pivots via ``jnp.interp`` instead.
    """
    import warnings

    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

    ssp = _ssp_or_skip()
    obs = Observation(photometry=Photometry.from_names(filters))
    groups = dict(
        sfh={"type": "dpl", "*": FIXED},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "*": FIXED,
            "tau_diff": 0.5,  # real attenuation → real L_ir to (not) re-emit in the optical
        },
        dust_emission={"type": "modified_blackbody", "*": FIXED},
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
    pe = np.asarray(m_exact.predict_photometry({}))
    pl = np.asarray(m_lut.predict_photometry({}))
    rel = np.abs(pl - pe) / np.maximum(np.abs(pe), 1e-30)
    # 5% generously covers the known blue effective-wavelength dust-factorization
    # residual (worst case u-band ≈ 2.4% at tau_diff=0.5) while still catching the
    # regression by 60× — the inflated reddest band was 290%+.
    assert rel.max() < 0.05, (
        f"{filters}: WavePrecomp photometry err {rel.max():.2%} (per-band {rel})"
    )
