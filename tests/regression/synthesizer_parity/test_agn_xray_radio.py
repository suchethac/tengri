# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for X-ray and radio wiring in unified_nlr_blr.

Tests backward compatibility (flags off) and correct spectral composition
(flags on) for the new X-ray and radio components.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.agn.unified import unified_nlr_blr
from tengri.utils.physics_constants import L_SUN as L_SUN_ERG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wave_uv_to_radio() -> jnp.ndarray:
    """Wide rest-frame wavelength grid covering UV through radio [Angstrom].

    Range: 1 Å (soft X-ray) to 1e9 Å (1 m radio wavelength).
    """
    return jnp.logspace(0.0, 9.0, 512)


@pytest.fixture(scope="module")
def wave_xray_only() -> jnp.ndarray:
    """X-ray wavelength range: 1–100 Å."""
    return jnp.logspace(0.0, 2.0, 128)


@pytest.fixture(scope="module")
def wave_radio_only() -> jnp.ndarray:
    """Radio wavelength range: 1e7–1e9 Å (1 mm–1 m)."""
    return jnp.logspace(7.0, 9.0, 128)


@pytest.fixture(scope="module")
def wave_uv_ir() -> jnp.ndarray:
    """UV-optical-IR range: 1e2–1e7 Å."""
    return jnp.logspace(2.0, 7.0, 256)


@pytest.fixture(scope="module")
def agn_lbol_physical() -> float:
    """Physical AGN luminosity: log10(L_bol / L_sun) = 12.0 (bright quasar)."""
    return 12.0


@pytest.fixture(scope="module")
def agn_lbol_erg(agn_lbol_physical) -> float:
    """Same as above in erg/s."""
    return 10.0**agn_lbol_physical * L_SUN_ERG


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


def test_xray_off_radio_off_matches_default(wave_uv_ir, agn_lbol_physical):
    """With include_xray=False, include_radio=False, output unchanged from original.

    This guards against regression where new parameters or code paths
    accidentally alter the existing UV-FIR SED.
    """
    # Original call (no new params)
    sed_original = unified_nlr_blr(
        wave_uv_ir,
        agn_log_lbol=agn_lbol_physical,
        agn_cos_inc=0.6,
        agn_theta_torus=30.0,
        agn_nlr_cf=0.1,
        agn_blr_cf=0.1,
    )

    # Explicit flags off (should be identical)
    sed_flags_off = unified_nlr_blr(
        wave_uv_ir,
        agn_log_lbol=agn_lbol_physical,
        agn_cos_inc=0.6,
        agn_theta_torus=30.0,
        agn_nlr_cf=0.1,
        agn_blr_cf=0.1,
        include_xray=False,
        include_radio=False,
    )

    # Should match to floating-point precision
    assert jnp.allclose(sed_original, sed_flags_off, rtol=1e-14), (
        "Original unified_nlr_blr output differs with explicit flags=False; "
        "new parameters may have altered the code path."
    )


def test_xray_off_radio_off_baseline_is_nonnegative(wave_uv_ir, agn_lbol_physical):
    """Baseline SED (no X-ray, no radio) must be non-negative everywhere."""
    sed = unified_nlr_blr(
        wave_uv_ir,
        agn_log_lbol=agn_lbol_physical,
        include_xray=False,
        include_radio=False,
    )
    assert jnp.all(sed >= 0.0), "Baseline SED has negative values."
    chex.assert_tree_all_finite(sed)


# ---------------------------------------------------------------------------
# X-ray component tests
# ---------------------------------------------------------------------------


def test_xray_adds_short_wavelength_emission(wave_uv_to_radio, agn_lbol_physical):
    """With include_xray=True, emission appears at λ < 100 Å (not before)."""
    sed_no_xray = unified_nlr_blr(
        wave_uv_to_radio,
        agn_log_lbol=agn_lbol_physical,
        include_xray=False,
        include_radio=False,
    )

    sed_with_xray = unified_nlr_blr(
        wave_uv_to_radio,
        agn_log_lbol=agn_lbol_physical,
        include_xray=True,
        xray_gamma_agn=1.8,
        xray_alpha_ox=-1.4,
        xray_E_cut=300.0,
        include_radio=False,
    )

    # X-ray mask boundary: λ < 124 Å
    xray_mask = wave_uv_to_radio < 124.0
    ir_mask = wave_uv_to_radio >= 124.0

    # At X-ray wavelengths, the new SED should be brighter
    sed_diff_xray = sed_with_xray[xray_mask] - sed_no_xray[xray_mask]
    assert jnp.any(sed_diff_xray > 0.0), "X-ray component should add positive flux at λ < 124 Å"

    # At IR wavelengths, SED should be unchanged (X-ray is shortward)
    assert jnp.allclose(sed_with_xray[ir_mask], sed_no_xray[ir_mask], rtol=1e-14), (
        "X-ray component leaked into IR wavelengths."
    )


def test_xray_is_nonnegative_and_finite(wave_uv_to_radio, agn_lbol_physical):
    """X-ray component produces valid (non-negative, finite) flux."""
    sed = unified_nlr_blr(
        wave_uv_to_radio,
        agn_log_lbol=agn_lbol_physical,
        include_xray=True,
        xray_gamma_agn=1.9,
        xray_alpha_ox=-1.3,
        xray_E_cut=250.0,
        include_radio=False,
    )
    chex.assert_tree_all_finite(sed)
    assert jnp.all(sed >= 0.0), "X-ray SED has negative values."


# ---------------------------------------------------------------------------
# Radio component tests
# ---------------------------------------------------------------------------


def test_radio_adds_long_wavelength_emission(wave_uv_to_radio, agn_lbol_physical):
    """With include_radio=True, emission appears at λ > 1 mm (not before)."""
    sed_no_radio = unified_nlr_blr(
        wave_uv_to_radio,
        agn_log_lbol=agn_lbol_physical,
        include_xray=False,
        include_radio=False,
    )

    sed_with_radio = unified_nlr_blr(
        wave_uv_to_radio,
        agn_log_lbol=agn_lbol_physical,
        include_xray=False,
        include_radio=True,
        radio_q_ir=2.64,
        radio_alpha_sf=0.8,
        radio_loudness=1.0,  # Non-zero to enable AGN radio
        radio_alpha_agn=0.7,
    )

    # Radio mask boundary: λ > 1e7 Å (1 mm)
    radio_mask = wave_uv_to_radio > 1e7

    # At radio wavelengths, the new SED should be brighter
    sed_diff_radio = sed_with_radio[radio_mask] - sed_no_radio[radio_mask]
    assert jnp.any(sed_diff_radio > 0.0), "Radio component should add positive flux at λ > 1 mm"

    # The jet must not touch the UV/optical, where the disc dominates.
    #
    # This used to assert the SED was unchanged everywhere shortward of 1 mm
    # (rtol=1e-14). That premise died with #1071: the AGN jet no longer hard-cuts
    # at 1 mm, so it legitimately contributes in the submm and far-IR — the
    # aging cutoff (nu_cut = 10^13 Hz ~ 30 um) confines it, not a cliff. The old
    # mask even included 1 mm itself, where this AGN-only SED carries no torus
    # flux at all and the jet is *supposed* to dominate.
    #
    # What must still hold is that the jet is invisible where the disc lives.
    # At 1000 A the jet is ~1e-134 of its 1.4 GHz value, so any residual here is
    # float64 roundoff in the component sum (~1e-14), not physics.
    optical_mask = wave_uv_to_radio <= 1e4  # UV/optical, λ ≤ 1 um
    assert jnp.allclose(sed_with_radio[optical_mask], sed_no_radio[optical_mask], rtol=1e-10), (
        "Radio component leaked into the UV/optical, where the disc dominates."
    )


def test_radio_is_nonnegative_and_finite(wave_uv_to_radio, agn_lbol_physical):
    """Radio component produces valid (non-negative, finite) flux."""
    sed = unified_nlr_blr(
        wave_uv_to_radio,
        agn_log_lbol=agn_lbol_physical,
        include_xray=False,
        include_radio=True,
        radio_q_ir=2.5,
        radio_alpha_sf=0.9,
        radio_loudness=0.5,
        radio_alpha_agn=0.65,
    )
    chex.assert_tree_all_finite(sed)
    assert jnp.all(sed >= 0.0), "Radio SED has negative values."


# ---------------------------------------------------------------------------
# Combined (X-ray + radio) tests
# ---------------------------------------------------------------------------


def test_both_xray_and_radio_on_spans_full_multiwavelength(wave_uv_to_radio, agn_lbol_physical):
    """With both flags on, SED spans 6+ orders of magnitude without gaps."""
    sed = unified_nlr_blr(
        wave_uv_to_radio,
        agn_log_lbol=agn_lbol_physical,
        include_xray=True,
        xray_gamma_agn=1.8,
        xray_alpha_ox=-1.4,
        xray_E_cut=300.0,
        include_radio=True,
        radio_q_ir=2.64,
        radio_alpha_sf=0.8,
        radio_loudness=0.5,
        radio_alpha_agn=0.7,
    )

    # Check validity
    chex.assert_tree_all_finite(sed)
    assert jnp.all(sed >= 0.0), "Combined SED has negative values."

    # Check that there are non-zero contributions in all three regimes
    x_ray_band = wave_uv_to_radio < 124.0
    ir_band = (wave_uv_to_radio >= 124.0) & (wave_uv_to_radio < 1e7)
    radio_band = wave_uv_to_radio >= 1e7

    assert jnp.any(sed[x_ray_band] > 0.0), "No X-ray contribution."
    assert jnp.any(sed[ir_band] > 0.0), "No optical-IR contribution."
    assert jnp.any(sed[radio_band] > 0.0), "No radio contribution."


# ---------------------------------------------------------------------------
# Gradient/JIT compatibility tests
# ---------------------------------------------------------------------------


def test_xray_radio_jit_compatible(wave_uv_to_radio, agn_lbol_physical):
    """Function must be JIT-compilable with new parameters."""

    @jax.jit
    def call_with_xray_radio(wave, lbol):
        return unified_nlr_blr(
            wave,
            agn_log_lbol=lbol,
            include_xray=True,
            xray_gamma_agn=1.8,
            xray_alpha_ox=-1.4,
            xray_E_cut=300.0,
            include_radio=True,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=0.3,
            radio_alpha_agn=0.7,
        )

    # Should JIT without error
    sed = call_with_xray_radio(wave_uv_to_radio, agn_lbol_physical)
    chex.assert_tree_all_finite(sed)


def test_xray_radio_grad_compatible(wave_uv_to_radio, agn_lbol_physical):
    """Function must support gradient through new parameters."""

    def objective(lbol, gamma_agn, radio_loud):
        sed = unified_nlr_blr(
            wave_uv_to_radio,
            agn_log_lbol=lbol,
            include_xray=True,
            xray_gamma_agn=gamma_agn,
            xray_alpha_ox=-1.4,
            xray_E_cut=300.0,
            include_radio=True,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=radio_loud,
            radio_alpha_agn=0.7,
        )
        # Dummy loss: sum of log-flux
        return jnp.sum(jnp.log(jnp.clip(sed, 1e-40, jnp.inf)))

    # Compute gradients
    grad_fn = jax.grad(objective, argnums=(0, 1, 2))
    grads = grad_fn(agn_lbol_physical, 1.8, 0.5)

    # All gradients should be finite
    assert all(jnp.all(jnp.isfinite(g)) for g in grads), (
        "Gradient computation produced NaN or Inf."
    )


# ---------------------------------------------------------------------------
# Parameter sensitivity tests
# ---------------------------------------------------------------------------


def test_xray_alpha_ox_sensitivity(wave_xray_only, agn_lbol_physical):
    """Varying alpha_ox offset should change X-ray normalization (not shape).

    xray_alpha_ox is now a delta offset to the empirical Just+2007 relation.
    delta=0.2 → absolute alpha_ox=-1.2 (harder corona)
    delta=-0.2 → absolute alpha_ox=-1.6 (softer corona)
    """
    sed_ao1 = unified_nlr_blr(
        wave_xray_only,
        agn_log_lbol=agn_lbol_physical,
        include_xray=True,
        xray_gamma_agn=1.8,
        xray_alpha_ox=0.2,  # Delta → absolute -1.2 (harder UV-X coupling)
        xray_E_cut=300.0,
        include_radio=False,
    )

    sed_ao2 = unified_nlr_blr(
        wave_xray_only,
        agn_log_lbol=agn_lbol_physical,
        include_xray=True,
        xray_gamma_agn=1.8,
        xray_alpha_ox=-0.2,  # Delta → absolute -1.6 (softer UV-X coupling)
        xray_E_cut=300.0,
        include_radio=False,
    )

    # SEDs should differ
    assert not jnp.allclose(sed_ao1, sed_ao2, rtol=0.05), (
        "Varying alpha_ox offset should change X-ray flux."
    )
    # Fluxes should be in the same ballpark (not inverted or zeroed)
    assert jnp.all(sed_ao1 > 0.0) and jnp.all(sed_ao2 > 0.0)


def test_radio_loudness_sensitivity(wave_radio_only, agn_lbol_physical):
    """Varying radio_loudness should change radio normalization."""
    sed_loud0 = unified_nlr_blr(
        wave_radio_only,
        agn_log_lbol=agn_lbol_physical,
        include_xray=False,
        include_radio=True,
        radio_q_ir=2.64,
        radio_alpha_sf=0.8,
        radio_loudness=0.0,  # No AGN radio
        radio_alpha_agn=0.7,
    )

    sed_loud1 = unified_nlr_blr(
        wave_radio_only,
        agn_log_lbol=agn_lbol_physical,
        include_xray=False,
        include_radio=True,
        radio_q_ir=2.64,
        radio_alpha_sf=0.8,
        radio_loudness=1.0,  # Strong AGN radio
        radio_alpha_agn=0.7,
    )

    # With radio_loudness=1.0, radio SED should generally be brighter
    # (not strictly monotonic due to other contributions, but geometric mean
    # should increase)
    log_ratio = jnp.log10(jnp.clip(sed_loud1, 1e-40, jnp.inf)) - jnp.log10(
        jnp.clip(sed_loud0, 1e-40, jnp.inf)
    )
    assert jnp.mean(log_ratio) > 0.0, (
        "Increasing radio_loudness should increase radio flux on average."
    )
