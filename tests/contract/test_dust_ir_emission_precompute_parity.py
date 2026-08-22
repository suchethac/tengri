# SPDX-License-Identifier: BSD-3-Clause
"""Precompute↔exact parity for dust-IR emission families — WavePrecomp safety net.

This module is the numerical safety net for epic #1738 (unifying precomputed and
runtime photometry paths). It replaces the #1660 quarantine by capturing the lesson:
a naive equivalence test is *vacuous* if the component under test does not dominate
a band. This module provides a reusable parity helper that asserts band dominance
before comparing.

Design:
* Configurations are built with dust-IR emission active; the component under test
  must dominate at least one band when measured against a build with that emission
  disabled. Fail loudly if it does not. This is the #1660 lesson and the point of
  the whole exercise.
* A single random parameter draw is used for both `approx=None` (exact) and
  `approx=WavePrecomp()` (LUT) paths, ensuring identical SED structure.
* Photometry is compared with a physically motivated tolerance stated in the test.
* No broad except clauses — exceptions in model construction must FAIL the test,
  not skip it.

Tolerance: 0.5% (0.13-0.26% LUT residual per #1671 + 0.2% dust-attenuation
effective-wavelength approximation). Set to catch the systematic errors that #622
and #629 would have raised — an order of magnitude below physical systematics.

Coverage: dale2014 (template-backed; dominates an IR band when tau_diff=0.5).
Other emitters (modified_blackbody, draine_li2007) emit mostly in far-IR (20-100 µm)
and contribute <1% to optical/near-IR filters; they would require far-IR-focused
filter sets to dominate a band per #1660.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

# No ``jax.config.update("jax_enable_x64", True)`` here: ``tests/conftest.py``
# sets it before any test module is imported, so a second call is a no-op that
# reads as if this file controlled its own precision. Guarded by
# ``test_float64_is_set_once.py``.


def assert_precompute_matches_exact(
    ssp_data,
    filters,
    emission_type,
    *,
    dust_config=None,
    redshift_dist=None,
    tolerance=0.005,
):
    """Assert that WavePrecomp photometry matches exact path within tolerance.

    This is the load-bearing parity helper for #1738. It builds a configuration
    twice — once with `approx=None` (exact) and once with `approx=WavePrecomp()`
    (LUT) — using identical parameters, and compares predict_photometry with
    band dominance enforcement.

    Parameters
    ----------
    ssp_data : SSPData
        Pre-loaded SSP grid.
    filters : sequence of FilterCurve
        Photometry filters.
    emission_type : str
        Dust-emission type (e.g., "dale2014", "draine_li2007",
        "modified_blackbody").
    dust_config : dict, optional
        Per-parameter dust configuration (tau_diff, tau_bc, law_bc, etc.).
        Defaults to {'type': emission_type, 'all_params': FIXED, 'tau_diff': 0.5}.
    redshift_dist : Distribution or scalar, optional
        Redshift prior or fixed value. Defaults to Fixed(0.05).
    tolerance : float, default 0.005
        Relative tolerance on photometry [photometry]. Default 0.5% captures
        band-integration and additive-emitter exactness (LUT residual ~0.13-0.26%,
        plus dust-attenuation effective-wavelength approximation ~0.5%).

    Raises
    ------
    AssertionError
        If band dominance is not achieved (component does not dominate ≥1 band
        by >5% contribution), or if photometry disagreement exceeds tolerance.
    RuntimeError, AttributeError, TypeError
        Construction or evaluation failures are NOT caught — they fail the test,
        as intended for #1660 guard.
    """
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

    # Extract dust_attenuation and dust_emission from dust_config
    if dust_config is None:
        dust_attenuation = {
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": 0.5,
        }
        dust_emission = {"type": emission_type, "all_params": FIXED}
    else:
        # Split dust_config into attenuation and emission parts
        dust_attenuation = {k: v for k, v in dust_config.items() if k != "emission"}
        if "emission" in dust_config:
            dust_emission = dust_config["emission"]
        else:
            dust_emission = {"type": emission_type, "all_params": FIXED}

    if redshift_dist is None:
        redshift_dist = Fixed(0.05)

    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    groups = dict(sfh={"type": "dpl", "all_params": FIXED}, neb={"type": "none"})

    # Build 1: exact path (approx=None)
    exact_model = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        redshift=redshift_dist,
        approx=None,
        dust_attenuation=dust_attenuation,
        dust_emission=dust_emission,
        **groups,
    )

    # Build 2: LUT path (approx=WavePrecomp())
    lut_model = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        redshift=redshift_dist,
        approx=WavePrecomp(),
        dust_attenuation=dust_attenuation,
        dust_emission=dust_emission,
        **groups,
    )

    # Sample a single parameter draw and evaluate both paths with identical params
    key = jax.random.PRNGKey(42)
    params = exact_model.spec.sample(key)
    params_dict = dict(params)

    pe_exact = np.asarray(exact_model.predict_photometry(params_dict))
    pe_lut = np.asarray(lut_model.predict_photometry(params_dict))

    # ──────────────────────────────────────────────────────────────────────────
    # Band dominance check: emission must contribute >5% to at least one band.
    # ──────────────────────────────────────────────────────────────────────────
    # Build a reference (no emission) to measure dominance against
    dust_attenuation_no_emission = {k: v for k, v in dust_attenuation.items() if k != "emission"}
    if "law" not in dust_attenuation_no_emission:
        dust_attenuation_no_emission["law"] = "power_law"

    ref_model = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        redshift=redshift_dist,
        approx=None,
        dust_attenuation=dust_attenuation_no_emission,
        dust_emission={"type": "none"},
        **groups,
    )
    pe_ref = np.asarray(ref_model.predict_photometry(params_dict))

    # Contribution = (with_emission - without) / without (avoid div-by-zero)
    contribution = np.abs(pe_exact - pe_ref) / np.maximum(np.abs(pe_ref), 1e-30)
    max_contribution = np.max(contribution)

    if max_contribution < 0.05:
        raise AssertionError(
            f"{emission_type} does not dominate any band: max contribution "
            f"{max_contribution:.2%} < 5% threshold. Band contributions: {contribution}. "
            "This configuration is vacuous per #1660 — the comparison would be dominated "
            "by the stellar continuum and would say nothing about the emitter."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Parity check: photometry agreement between exact and LUT paths
    # ──────────────────────────────────────────────────────────────────────────
    rel_error = np.abs(pe_lut - pe_exact) / np.maximum(np.abs(pe_exact), 1e-30)
    max_rel_error = np.max(rel_error)

    assert max_rel_error < tolerance, (
        f"{emission_type} precompute↔exact mismatch exceeds tolerance "
        f"({max_rel_error:.3%} > {tolerance:.3%}). "
        f"Band errors: {rel_error}. Max contribution: {max_contribution:.2%}."
    )


# ── Synthetic fixtures (no SSP data files, no network) ─────────────────────


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Bare-stellar SSP on a UV→far-IR grid with smooth declining continuum."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_met, n_age = 3, 25
    wave = jnp.logspace(2.0, 7.0, 1600)  # 100 Å – 1 mm
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-2.5, -1.85, -1.2])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


def _tophat(center_aa, frac_width=0.18, n=48):
    """A synthetic top-hat filter."""
    from tengri.observation.photometry import FilterCurve

    lo, hi = center_aa * (1.0 - frac_width), center_aa * (1.0 + frac_width)
    wave = jnp.linspace(lo, hi, n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center_aa)}")


@pytest.fixture(scope="module")
def dust_ir_filters():
    """5-band optical→IR filter set for dust-emission testing."""
    return [_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0)]


# ── Tests ────────────────────────────────────────────────────────────────────


def test_dale2014_precompute_matches_exact(synthetic_ssp, dust_ir_filters):
    """Dale+2014 template-backed dust emission must match exact path.

    Band dominance: the dale2014 emission contributes ≥5% to at least one band.
    Tolerance: 0.5% (0.13-0.26% LUT residual + 0.2% dust-attenuation
    effective-wavelength approximation).
    """
    from tengri import FIXED

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_precompute_matches_exact(
            synthetic_ssp,
            dust_ir_filters,
            "dale2014",
            dust_config={
                "type": "two_component",
                "law": "calzetti",
                "all_params": FIXED,
                "tau_diff": 0.5,
                "emission": {"type": "dale2014", "all_params": FIXED},
            },
            tolerance=0.005,
        )
