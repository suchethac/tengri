# SPDX-License-Identifier: BSD-3-Clause
"""CI-runnable WavePrecomp accuracy contract — synthetic SSP, no data files.

The precompute-accuracy tests that compare ``approx=WavePrecomp()`` against the
exact path elsewhere are **SSP-gated**: they ``pytest.skip`` when the (gitignored)
``data/ssp_*.h5`` grids are absent, i.e. on CI. That gap let two silent
correctness regressions reach ``main`` while the suite looked green:

* the dust-IR family being omitted from the LUT projection (#622), and
* additive emitters being sampled at the filter *effective wavelength* instead
  of integrated through the band — which inflated the reddest band of a
  wavelength-sorted set by ~4× (293% on the reddest optical filter), and
  ~8% on PAH features (#629).

This module reproduces both failure modes with a **synthetic** SSP and
**synthetic** filters, so it runs everywhere — no SSP grids, no filter data,
no network. It is the regression guard that would have caught #622/#629 on the
PR rather than after merge.

Design:
* SSP wavelength grid spans UV → far-IR (≈100 Å – 1 mm) so the stellar SED
  drives the dust energy balance (L_absorbed) in the UV/optical and the dust IR
  re-emission has a grid to live on in the far-IR.
* Stellar flux is smooth (a declining continuum), so the stellar Φ-tensor LUT is
  near machine-exact and any residual beyond the documented dust-attenuation
  effective-wavelength approximation flags a real bug.
* Filters are top-hats in increasing-wavelength order (the ordering that
  triggered #629) spanning optical → IR.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)


# ── Synthetic, self-contained fixtures ────────────────────────────


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Bare-stellar SSP on a UV→far-IR grid with a smooth declining continuum.

    Smoothness matters: the Φ-tensor integrates the SSP through each filter
    exactly, so for a smooth SED the only WavePrecomp residual is the
    dust-attenuation effective-wavelength approximation. A noisy SED would
    inject large band-to-band interpolation error and mask real bugs.
    """
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_met, n_age = 3, 25
    # 100 Å – 1 mm (1e7 Å), log-spaced, dense enough to integrate filters + MBB.
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)  # log10(age/Gyr): ~1 Myr – 13.8 Gyr
    lgmet = jnp.array([-2.5, -1.85, -1.2])  # absolute log10(Z)

    # Smooth declining continuum (∝ (5000 Å / λ)^2 in Lν), bright in the
    # UV/optical and ~0 in the far-IR, mildly modulated by age and metallicity.
    # Strictly positive, no noise → the Φ-tensor LUT is near machine-exact.
    base = (5000.0 / wave) ** 2  # (n_wave,)
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )  # (n_met, n_age, n_wave)
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


def _tophat(center_aa: float, frac_width: float = 0.18, n: int = 48):
    """A synthetic top-hat filter (FilterCurve) centred at ``center_aa`` [Å]."""
    from tengri.observation.photometry import FilterCurve

    lo, hi = center_aa * (1.0 - frac_width), center_aa * (1.0 + frac_width)
    wave = jnp.linspace(lo, hi, n)
    # Taper the edges to 0 so the padded-integral denominator is well-behaved
    # (mirrors real filter curves).
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center_aa)}")


def _build(
    ssp, filters, approx, *, emission=None, with_radio=False, with_xray=False, redshift=None
):
    """Build an SEDModel with a dust component (optionally IR re-emission).

    ``redshift`` defaults to ``Fixed(0.05)``; pass a free distribution to
    exercise the free-z ztable LUT path.
    """
    import warnings

    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel

    if redshift is None:
        redshift = Fixed(0.05)
    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    dust = {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FIXED,
        "tau_diff": 0.5,  # real attenuation → real L_ir to (re-)emit
    }
    if emission is not None:
        dust["emission"] = {"type": emission, "*": FIXED}
    groups = dict(sfh={"type": "dpl", "*": FIXED}, dust=dust, neb={"type": "none"})
    if with_radio:
        groups["radio"] = {"type": "condon92", "*": FIXED}
    if with_xray:
        groups["xray"] = {"type": "simple", "*": FIXED}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp, observation=obs, redshift=redshift, approx=approx, **groups
        )


# ── The regression guards ─────────────────────────────────────────


@pytest.mark.regression_bug
@pytest.mark.parametrize(
    "centers",
    [
        [8000.0],  # single band
        [8000.0, 6000.0],  # reddest first → was fine even pre-#629
        [6000.0, 8000.0],  # reddest LAST → the #629 293% failure
        [3500.0, 5000.0, 6500.0, 8000.0],  # sorted optical (reddest = last)
    ],
)
def test_waveprecomp_matches_exact_with_dust_emission(synthetic_ssp, centers):
    """#629: with dust IR emission active, every optical band must match the
    exact path. Pre-fix the reddest band in a sorted set was ~4× too high
    (additive emission sampled at the effective wavelength, where the
    self-normalising model blew up). Now it is filter-integrated → exact."""
    filters = [_tophat(c) for c in centers]
    m_exact = _build(synthetic_ssp, filters, None, emission="modified_blackbody")
    m_lut = _build(synthetic_ssp, filters, _WP(), emission="modified_blackbody")
    pe = np.asarray(m_exact.predict_photometry({}))
    pl = np.asarray(m_lut.predict_photometry({}))
    rel = np.abs(pl - pe) / np.maximum(np.abs(pe), 1e-30)
    # 2% covers the stellar dust-attenuation effective-wavelength residual; the
    # regression was 290%+, so this catches it by >100×.
    assert rel.max() < 0.02, f"centers={centers}: WavePrecomp err {rel.max():.2%} (bands {rel})"


@pytest.mark.regression_bug
def test_waveprecomp_additive_emitters_are_exact(synthetic_ssp):
    """Additive emitters (dust IR, radio, X-ray) are integrated through the true
    filter transmission under WavePrecomp, so an IR band carrying real dust
    emission matches the exact path to near machine precision — not the ~8%
    of effective-wavelength sampling."""
    # Optical anchor + a mid-IR band where the MBB re-emission contributes.
    filters = [_tophat(5000.0), _tophat(8000.0), _tophat(8.0e4)]
    m_exact = _build(
        synthetic_ssp,
        filters,
        None,
        emission="modified_blackbody",
        with_radio=True,
        with_xray=True,
    )
    m_lut = _build(
        synthetic_ssp,
        filters,
        _WP(),
        emission="modified_blackbody",
        with_radio=True,
        with_xray=True,
    )
    pe = np.asarray(m_exact.predict_photometry({}))
    pl = np.asarray(m_lut.predict_photometry({}))
    rel = np.abs(pl - pe) / np.maximum(np.abs(pe), 1e-30)
    # The IR band (index 2) is dominated by additive emission → must be exact.
    assert rel[2] < 5e-3, f"IR band additive-emitter err {rel[2]:.3%} (all bands {rel})"


@pytest.mark.regression_bug
def test_waveprecomp_free_z_additive_emitters_exact(synthetic_ssp):
    """The free-z (ztable) LUT path must also publish the padded filter curves
    and integrate additive emitters exactly. The #629 plumbing was added to both
    the fixed-z and free-z branches of the stellar component, so an IR band under
    a *free* redshift matches the exact path too."""
    from tengri import Uniform

    z = {"redshift": 0.05}
    filters = [_tophat(5000.0), _tophat(8000.0), _tophat(8.0e4)]
    m_exact = _build(
        synthetic_ssp, filters, None, emission="modified_blackbody", redshift=Uniform(0.01, 0.5)
    )
    m_lut = _build(
        synthetic_ssp, filters, _WP(), emission="modified_blackbody", redshift=Uniform(0.01, 0.5)
    )
    pe = np.asarray(m_exact.predict_photometry(z))
    pl = np.asarray(m_lut.predict_photometry(z))
    rel = np.abs(pl - pe) / np.maximum(np.abs(pe), 1e-30)
    assert rel[2] < 1e-2, f"free-z IR additive-emitter err {rel[2]:.3%} (all bands {rel})"


@pytest.mark.regression_bug
def test_waveprecomp_publishes_padded_filter_curves(synthetic_ssp):
    """The exact-projection path requires the stellar component to publish the
    padded filter curves into ``state.derived`` (the #629 plumbing)."""
    filters = [_tophat(5000.0), _tophat(8000.0)]
    m = _build(synthetic_ssp, filters, _WP(), emission="modified_blackbody")
    derived = m.predict_state({}).derived
    assert "phot_filter_waves_padded" in derived.field_names()
    assert "phot_filter_trans_padded" in derived.field_names()
    assert "dust_emission_phot_lnu_precomp" in derived.field_names()


def _WP():
    from tengri import WavePrecomp

    return WavePrecomp()
