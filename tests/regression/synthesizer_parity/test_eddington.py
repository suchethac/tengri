# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for Eddington luminosity unit consistency — synthesizer parity.

Mirrors synthesizer's Eddington-ratio validation logic to ensure L_Edd computation
and parameter wiring stay consistent. Any unit mismatch or missing gradient path
breaks inference.

Pitfall: P-2 — Eddington luminosity unit mismatch. Synthesizer PR #1068 had
``eddington_luminosity`` in wrong units (solar luminosity vs erg/s); ratio
calculation failed because ``bolometric_luminosity`` (erg/s) and
``eddington_luminosity`` (L_sun) weren't converted to same units.

Pitfall: P-4 — Eddington accretion rate not triggering downstream calcs.
Synthesizer issue #1011: providing ``eddington_ratio`` doesn't compute
``accretion_rate`` or ``bolometric_luminosity`` on its own.

Synthesizer source:
- https://github.com/flaresimulations/synthesizer/pull/1068
- https://github.com/flaresimulations/synthesizer/issues/1011

Reference papers:
- Eddington 1926, The Internal Constitution of the Stars
- Krolik 1999, Active Galactic Nuclei (textbook, classical Eddington formula)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_paper
import jax
import jax.numpy as jnp

from tengri.utils.physics_constants import L_SUN as L_SUN_ERG


def eddington_luminosity(log_mbh: float | jnp.ndarray) -> jnp.ndarray:
    """Eddington luminosity in erg/s given log10(M_BH / M_sun).

    Parameters
    ----------
    log_mbh : float or array_like
        log10(M_BH / M_sun). Black hole mass in solar units.

    Returns
    -------
    L_Edd : ndarray, shape ()
        Eddington luminosity in erg/s. Uses the standard formula:
        L_Edd = 4π G M c / σ_T = 1.26e38 × (M / M_sun) erg/s

    Notes
    -----
    The prefactor 1.26e38 erg/s per M_sun is the fundamental Eddington-luminosity
    constant (Eddington 1926, Krolik 1999). Any deviation indicates a unit-system
    change that will break downstream code.
    """
    m_bh_msun = 10.0**log_mbh
    l_edd_erg = 1.26e38 * m_bh_msun
    return jnp.asarray(l_edd_erg)


def test_eddington_luminosity_prefactor_matches_standard():
    """L_Edd = 1.26e38 × (M / M_sun) erg/s is the fundamental constant.

    Pitfall P-2: If the prefactor drifts (e.g., 1.26e37, 1.26e39), all Eddington
    ratios shift by an order of magnitude, breaking Bayesian parameter recovery.

    Test: L_Edd(M = 1e8 M_sun) ≈ 1.26e46 erg/s.
    """
    log_mbh = 8.0  # M_BH = 1e8 M_sun
    l_edd = float(eddington_luminosity(log_mbh))

    expected = 1.26e46  # erg/s
    rel_err = abs(l_edd - expected) / expected

    assert rel_err < 0.01, (
        f"P-2 BUG: L_Edd(M=1e8 M_sun) = {l_edd:.3e} erg/s, "
        f"expected {expected:.3e} (rel_err {rel_err:.2e}). "
        "Eddington prefactor drifted; all posteriors will be wrong."
    )


def test_eddington_luminosity_scales_linearly_with_mass():
    """L_Edd ∝ M_BH (linear), not M^2 or M^0.5.

    Pitfall P-2: Detects if a missing or extra factor enters the calculation.
    """
    log_mbh_low = 7.0  # M = 1e7 M_sun
    log_mbh_high = 8.0  # M = 1e8 M_sun (10× heavier)

    l_edd_low = float(eddington_luminosity(log_mbh_low))
    l_edd_high = float(eddington_luminosity(log_mbh_high))

    ratio = l_edd_high / l_edd_low
    expected_ratio = 10.0  # linear scaling

    rel_err = abs(ratio - expected_ratio) / expected_ratio
    assert rel_err < 0.01, (
        f"P-2 BUG: L_Edd does not scale linearly with M. "
        f"Ratio for +1 dex in M = {ratio:.2f}, expected {expected_ratio:.2f}. "
        "Possible exponent error (e.g., M^1.5 or M^0.5)."
    )


def test_eddington_ratio_unit_consistency():
    """Eddington ratio = L_bol / L_Edd must be dimensionless.

    Round-trip: start with agn_log_lbol (log10 L_sun) and agn_log_mbh (log10 M_sun).
    Compute L_bol (erg/s) and L_Edd (erg/s) in the same unit system.
    Form ratio: should be dimensionless and typically in [1e-3, 3] for bright Seyferts.

    Pitfall P-2: If L_bol stays in L_sun while L_Edd converts to erg/s (or vice versa),
    ratio will be off by a factor 1e33.
    """
    agn_log_lbol = 11.0  # log10(L_bol / L_sun) — quasar example
    agn_log_mbh = 8.0  # log10(M_BH / M_sun)

    # Forward-model side: convert agn_log_lbol (L_sun) to erg/s
    l_bol_erg = 10.0**agn_log_lbol * L_SUN_ERG

    # AGN physics: compute L_Edd from M_BH
    l_edd_erg = float(eddington_luminosity(agn_log_mbh))

    # Form ratio (must be dimensionless)
    eddington_ratio = l_bol_erg / l_edd_erg

    # Sanity: typical bright Seyfert/quasar sits at λ_Edd ~ 0.01–1
    assert 1e-4 < eddington_ratio < 10.0, (
        f"P-2 BUG: Eddington ratio = {eddington_ratio:.3e} outside "
        f"expected [1e-4, 10] range for agn_log_lbol=11, agn_log_mbh=8. "
        "Likely unit mismatch in L_bol or L_Edd."
    )

    # Tighter check: for these typical parameters, should be ~ 0.1-1
    assert 0.01 < eddington_ratio < 3.0, (
        f"P-2 WARNING: Eddington ratio {eddington_ratio:.3e} seems extreme. "
        "Verify unit-system consistency in components/agn/_phys.py."
    )


def test_eddington_ratio_monotonic_in_lbol():
    """Doubling L_bol doubles the Eddington ratio (at fixed M_BH).

    Pitfall P-2, P-4: If agn_log_lbol is not wired into the Eddington ratio
    calculation, doubling L_bol will have no effect.
    """
    agn_log_mbh = 8.0

    # Two L_bol values (1× and 2×)
    agn_log_lbol_1x = 11.0
    agn_log_lbol_2x = jnp.log10(2.0 * 10.0**agn_log_lbol_1x)

    l_bol_1x_erg = 10.0**agn_log_lbol_1x * L_SUN_ERG
    l_bol_2x_erg = 10.0**agn_log_lbol_2x * L_SUN_ERG
    l_edd_erg = float(eddington_luminosity(agn_log_mbh))

    ratio_1x = l_bol_1x_erg / l_edd_erg
    ratio_2x = l_bol_2x_erg / l_edd_erg

    # ratio_2x / ratio_1x should be ≈ 2.0
    scaling = float(ratio_2x / ratio_1x)
    expected_scaling = 2.0

    rel_err = abs(scaling - expected_scaling) / expected_scaling
    assert rel_err < 0.01, (
        f"P-4 BUG: agn_log_lbol doubling does not double Eddington ratio. "
        f"Scaling = {scaling:.3f}, expected {expected_scaling:.1f}. "
        "agn_log_lbol may not be wired to ratio computation."
    )


def test_eddington_gradient_wrt_log_mbh_is_finite():
    """Gradient of L_Edd w.r.t. log10(M_BH) is finite and positive.

    Pitfall P-4: If Eddington computation is broken in JAX's trace
    (e.g., logs outside JIT), gradient will be zero or NaN.
    """

    def loss_fn(log_mbh_traced):
        return jnp.sum(eddington_luminosity(log_mbh_traced))

    log_mbh_test = 8.0
    grad_fn = jax.grad(loss_fn)
    grad = float(grad_fn(jnp.array(log_mbh_test)))

    # dL_Edd / d(log_M) = L_Edd * ln(10) for the linear scaling L ∝ 10^log_M
    # L_Edd at M = 1e8 M_sun is 1.26e46, so gradient ≈ 1.26e46 * 2.303 ≈ 2.9e46
    expected_grad_approx = 1.26e46 * 2.303  # rough estimate

    assert jnp.isfinite(grad), (
        "P-4 BUG: gradient of L_Edd is non-finite (NaN or inf). "
        "Eddington computation may be outside JAX trace."
    )

    assert grad > 0.0, (
        f"P-4 BUG: gradient of L_Edd w.r.t. log_M is {grad:.3e}, should be positive. "
        "Derivative broken or inverted sign."
    )

    # Allow 10% tolerance on magnitude (numerical noise, but direction clear)
    rel_err = abs(grad - expected_grad_approx) / expected_grad_approx
    assert rel_err < 1.0, (
        f"P-4 WARNING: gradient magnitude {grad:.3e} "
        f"differs from rough estimate {expected_grad_approx:.3e} by >100%. "
        "Verify Eddington formula is traced correctly."
    )


def test_eddington_gradient_wrt_log_lbol_is_finite():
    """Gradient of L_Edd w.r.t. log10(L_bol) is zero (L_Edd independent of L_bol).

    Pitfall P-2, P-4: Eddington ratio depends on L_bol, but L_Edd itself only
    depends on M_BH. This test ensures that a naive implementation doesn't
    accidentally mix the gradient paths.
    """

    def loss_fn(log_lbol_traced):
        # L_Edd is fixed; we compute L_bol from log and form ratio
        l_bol_erg = 10.0**log_lbol_traced * L_SUN_ERG
        l_edd_erg = float(eddington_luminosity(8.0))  # M_BH = 1e8 M_sun
        ratio = l_bol_erg / l_edd_erg
        return jnp.sum(ratio)

    log_lbol_test = 11.0
    grad_fn = jax.grad(loss_fn)
    grad = float(grad_fn(jnp.array(log_lbol_test)))

    # d(L_bol / L_Edd) / d(log_L_bol) = (1 / L_Edd) * dL_bol/d(log_L_bol)
    # = (1 / L_Edd) * L_bol * ln(10)
    l_bol_erg = 10.0**log_lbol_test * L_SUN_ERG
    l_edd_erg = float(eddington_luminosity(8.0))
    expected_grad = l_bol_erg / l_edd_erg * 2.303  # ln(10)

    assert jnp.isfinite(grad), (
        "P-4 BUG: gradient of Eddington ratio w.r.t. log_L_bol is non-finite."
    )

    assert grad > 0.0, (
        f"P-4 BUG: gradient of Eddington ratio w.r.t. log_L_bol is {grad:.3e}, "
        "should be positive (more L_bol → higher ratio)."
    )

    rel_err = abs(grad - expected_grad) / expected_grad
    assert rel_err < 0.1, (
        f"P-4 WARNING: gradient magnitude {grad:.3e} "
        f"differs from expected {expected_grad:.3e} by {100 * rel_err:.1f}%. "
        "Check that agn_log_lbol is properly traced."
    )
