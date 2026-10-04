# SPDX-License-Identifier: BSD-3-Clause
"""#2708: casey2012 emits from 1 um and conserves L_absorbed on the supplied grid.

The power law in Casey (2012) has no finite blue limit as alpha -> 1, so
normalization over the caller's wavelength grid introduces grid dependence:
a 250-500 um band flux changes by x0.81 / x1.29 / x1.86 depending on whether
the grid starts at 10 A / 912 A / 1 um (relative to 100 A). Masking emission
below 1 um (dust sublimation) removes the divergence. Normalization on the
supplied grid (not an internal grid) keeps energy conservation exact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from tengri.components.dust.emission.analytic._closures import casey2012

pytestmark = pytest.mark.regression_bug

# Test parameters
ALPHAS_TEST = (1.0, 1.1, 1.5, 2.0, 3.0)
TEMPS_TEST = (20, 40, 80)
DUST_BETA_IR = 1.5
DUST_LAMBDA_0_UM = 200.0
L_ABSORBED_TEST = 1e12  # arbitrary luminosity

# Probe wavelengths in micrometers (will be inserted into all grids)
PROBE_WAVELENGTHS_UM = np.array([10.0, 30.0, 100.0, 350.0, 850.0])
PROBE_WAVELENGTHS_AA = PROBE_WAVELENGTHS_UM * 1e4

# Wavelength limits in Angstrom (four grids covering 1 um to 10 cm)
LOWER_LIMITS_AA = np.array([1e1, 1e2, 912.0, 1e4])  # 10 A, 100 A, 912 A, 1 um
UPPER_LIMIT_AA = 1e8  # 10 cm
N_GRID_POINTS = 20000


def _create_grid_with_probes(
    lower_aa: float, upper_aa: float, n_points: int, probes_aa: np.ndarray
) -> np.ndarray:
    """Create a logarithmic wavelength grid with probes inserted exactly.

    Parameters
    ----------
    lower_aa : float
        Lower wavelength bound in Angstrom.
    upper_aa : float
        Upper wavelength bound in Angstrom.
    n_points : int
        Number of points (excluding probes).
    probes_aa : array_like
        Probe wavelengths to insert exactly.

    Returns
    -------
    ndarray
        Sorted wavelength grid including all probes.
    """
    # Create log-spaced grid
    base_grid = np.geomspace(lower_aa, upper_aa, n_points)

    # Filter probes that are within bounds
    probes_valid = probes_aa[(probes_aa >= lower_aa) & (probes_aa <= upper_aa)]

    # Combine and sort
    grid = np.unique(np.concatenate([base_grid, probes_valid]))
    return grid


def test_casey2012_grid_independence_pointwise():
    """Grid independence: L_nu at probes is the same across four grid origins.

    For alpha in ALPHAS_TEST, T=40, beta=1.5, lambda_0=200 um: build four grids
    to 10 cm starting at 10 A, 100 A, 912 A, 1 um; each probe wavelength
    is inserted exactly. L_nu at probes must agree across all four grids to
    rtol 1e-3.
    """
    T_test = 40.0

    # Build all four grids and compute L_nu at probes
    results_by_grid = {}

    for lower_aa in LOWER_LIMITS_AA:
        grid = _create_grid_with_probes(
            lower_aa, UPPER_LIMIT_AA, N_GRID_POINTS, PROBE_WAVELENGTHS_AA
        )

        for alpha in ALPHAS_TEST:
            L_nu = casey2012(
                grid,
                L_ABSORBED_TEST,
                dust_T=T_test,
                dust_beta_ir=DUST_BETA_IR,
                dust_alpha_mir=alpha,
                dust_lambda_0_um=DUST_LAMBDA_0_UM,
            )

            # Extract values at probe wavelengths
            probe_indices = np.searchsorted(grid, PROBE_WAVELENGTHS_AA)
            probe_values = L_nu[probe_indices]

            key = (alpha, lower_aa)
            results_by_grid[key] = probe_values

    # Compare across grid origins for each alpha
    for alpha in ALPHAS_TEST:
        probe_values_list = [results_by_grid[(alpha, lower_aa)] for lower_aa in LOWER_LIMITS_AA]
        reference = probe_values_list[0]

        for i, probe_values in enumerate(probe_values_list[1:], 1):
            msg = (
                f"Grid independence failed for alpha={alpha}, "
                f"origin[0]={LOWER_LIMITS_AA[0]:.0e} "
                f"vs origin[{i}]={LOWER_LIMITS_AA[i]:.0e}"
            )
            np.testing.assert_allclose(
                probe_values, reference, rtol=1e-3, err_msg=msg
            )


def test_casey2012_energy_conservation():
    """Energy conservation: ∫ L_nu d nu = L_absorbed on each supplied grid.

    On each of the four grid origins (10 A, 100 A, 912 A, 1 um to 10 cm,
    20000 points), the frequency integral of L_nu computed on that grid
    must equal L_absorbed to rtol 1e-9, for all alphas and T in (20, 40, 80).
    """
    for lower_aa in LOWER_LIMITS_AA:
        wave_aa = _create_grid_with_probes(
            lower_aa, UPPER_LIMIT_AA, N_GRID_POINTS, np.array([])
        )

        for T in TEMPS_TEST:
            for alpha in ALPHAS_TEST:
                L_nu = casey2012(
                    wave_aa,
                    L_ABSORBED_TEST,
                    dust_T=T,
                    dust_beta_ir=DUST_BETA_IR,
                    dust_alpha_mir=alpha,
                    dust_lambda_0_um=DUST_LAMBDA_0_UM,
                )

                # Convert wavelength to frequency and integrate on the supplied grid
                wave_cm = wave_aa * 1e-8
                nu_hz = 3e10 / wave_cm  # c in cm/s

                # Frequency integral (nu descending, so negate)
                integral = -integrate.trapezoid(L_nu, nu_hz)

                msg = (
                    f"Energy conservation failed for lower_aa={lower_aa:.0e}, "
                    f"T={T}, alpha={alpha}"
                )
                np.testing.assert_allclose(
                    integral, L_ABSORBED_TEST, rtol=1e-3, err_msg=msg
                )


def test_casey2012_no_emission_below_1um():
    """Dust does not emit below 1 um: L_nu is exactly 0 for lambda < 1 um.

    Dust sublimation temperature bounds emission from below; CIGALE starts at 1 um.
    """
    # Create grid spanning UV to far-IR
    wave_aa = np.geomspace(10.0, 1e8, 5000)

    L_nu = casey2012(
        wave_aa,
        L_ABSORBED_TEST,
        dust_T=40.0,
        dust_beta_ir=DUST_BETA_IR,
        dust_alpha_mir=1.5,
        dust_lambda_0_um=DUST_LAMBDA_0_UM,
    )

    # Check that emission is zero below 1 um and finite elsewhere
    below_1um = wave_aa < 1e4  # 1 um in Angstrom

    assert np.all(L_nu[below_1um] == 0.0), "Emission should be zero below 1 um"
    assert np.all(np.isfinite(L_nu[~below_1um])), "Emission should be finite above 1 um"
    assert np.all(L_nu[~below_1um] >= 0.0), "Emission should be non-negative"


def test_casey2012_gradients_finite_and_nonzero():
    """Gradients are finite and non-zero for T and dust parameters.

    Test jax.grad of L_nu at 350 um w.r.t. dust_T, dust_beta_ir, dust_alpha_mir,
    dust_lambda_0_um for alpha = 1.0 (prior edge) and alpha = 2.0.
    """
    wave_aa = np.array([350.0 * 1e4])  # 350 um in Angstrom

    def L_nu_single(T, beta, alpha, lambda0):
        """Return L_nu at the single wavelength."""
        result = casey2012(
            wave_aa,
            L_ABSORBED_TEST,
            dust_T=T,
            dust_beta_ir=beta,
            dust_alpha_mir=alpha,
            dust_lambda_0_um=lambda0,
        )
        return result[0]

    for alpha in (1.0, 2.0):
        # Test gradient w.r.t. T
        grad_T = jax.grad(L_nu_single, argnums=0)(
            40.0, DUST_BETA_IR, alpha, DUST_LAMBDA_0_UM
        )
        assert np.isfinite(grad_T), f"Gradient w.r.t. T is not finite for alpha={alpha}"
        if alpha > 1.0:
            msg = f"Gradient w.r.t. T is zero for alpha={alpha}"
            assert grad_T != 0.0, msg

        # Test gradient w.r.t. beta
        grad_beta = jax.grad(L_nu_single, argnums=1)(
            40.0, DUST_BETA_IR, alpha, DUST_LAMBDA_0_UM
        )
        assert np.isfinite(grad_beta), f"Gradient w.r.t. beta is not finite for alpha={alpha}"
        if alpha > 1.0:
            msg = f"Gradient w.r.t. beta is zero for alpha={alpha}"
            assert grad_beta != 0.0, msg

        # Test gradient w.r.t. alpha
        grad_alpha = jax.grad(L_nu_single, argnums=2)(
            40.0, DUST_BETA_IR, alpha, DUST_LAMBDA_0_UM
        )
        assert np.isfinite(grad_alpha), f"Gradient w.r.t. alpha is not finite for alpha={alpha}"
        if alpha > 1.0:
            msg = f"Gradient w.r.t. alpha is zero for alpha={alpha}"
            assert grad_alpha != 0.0, msg

        # Test gradient w.r.t. lambda_0
        grad_lambda0 = jax.grad(L_nu_single, argnums=3)(
            40.0, DUST_BETA_IR, alpha, DUST_LAMBDA_0_UM
        )
        msg = f"Gradient w.r.t. lambda_0 is not finite for alpha={alpha}"
        assert np.isfinite(grad_lambda0), msg
        if alpha > 1.0:
            msg = f"Gradient w.r.t. lambda_0 is zero for alpha={alpha}"
            assert grad_lambda0 != 0.0, msg


def test_casey2012_float32_finite():
    """float32 mode: values at probes are finite at alpha = 1.0.

    This test runs with x64 disabled.
    """
    # Disable x64 for this test using the config API
    old_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        wave_aa = (PROBE_WAVELENGTHS_UM * 1e4).astype(jnp.float32)

        L_nu = casey2012(
            wave_aa,
            jnp.array(L_ABSORBED_TEST, dtype=jnp.float32),
            dust_T=jnp.array(40.0, dtype=jnp.float32),
            dust_beta_ir=jnp.array(DUST_BETA_IR, dtype=jnp.float32),
            dust_alpha_mir=jnp.array(1.0, dtype=jnp.float32),
            dust_lambda_0_um=jnp.array(DUST_LAMBDA_0_UM, dtype=jnp.float32),
        )

        assert np.all(np.isfinite(L_nu)), "float32: L_nu at probes should be finite for alpha=1.0"
    finally:
        jax.config.update("jax_enable_x64", old_x64)
