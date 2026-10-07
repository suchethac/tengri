# SPDX-License-Identifier: BSD-3-Clause
"""#2708: casey2012 emits from 1 um and conserves L_absorbed on the supplied grid.

The energy integral of the Casey (2012) mid-IR power law has no finite blue
limit as alpha -> 1, so the share of L_absorbed assigned to the blue side
depends on where the caller's grid starts: a 250-500 um band flux changes by
x0.81 / x1.29 / x1.86 on grids starting at 10 A / 912 A / 1 um (relative to
100 A). Setting emission below 1 um to zero, a bound that follows the range of
CIGALE's casey2012 template, removes the divergence. Normalizing on the
supplied grid keeps energy conservation exact.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest

from tengri.components.dust.emission._physics import (
    cmb_contrast_factor,
    cmb_corrected_temperature,
)
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

# Wavelength limits in Angstrom (four grids to 10 cm, starting at or below 1 um)
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
            np.testing.assert_allclose(probe_values, reference, rtol=1e-3, err_msg=msg)


def test_casey2012_energy_conservation():
    """On each grid, the normalized shape integrates to L_absorbed to rtol 1e-9.

    The CMB contrast at z = 0 removes a physical fraction of the emitted
    energy after normalization; it is divided out here so the check isolates
    the normalization.
    """
    for lower_aa in LOWER_LIMITS_AA:
        wave_aa = _create_grid_with_probes(lower_aa, UPPER_LIMIT_AA, N_GRID_POINTS, np.array([]))
        nu_hz = 2.99792458e18 / wave_aa
        for T in TEMPS_TEST:
            for alpha in ALPHAS_TEST:
                L_nu = np.asarray(
                    casey2012(
                        wave_aa,
                        L_ABSORBED_TEST,
                        dust_T=T,
                        dust_beta_ir=DUST_BETA_IR,
                        dust_alpha_mir=alpha,
                        dust_lambda_0_um=DUST_LAMBDA_0_UM,
                    )
                )
                contrast = np.asarray(
                    cmb_contrast_factor(
                        wave_aa, cmb_corrected_temperature(T, 0.0, DUST_BETA_IR), 0.0
                    )
                )
                np.testing.assert_allclose(
                    -np.trapezoid(L_nu / contrast, nu_hz),
                    L_ABSORBED_TEST,
                    rtol=1e-9,
                    err_msg=f"lower_aa={lower_aa:.0e}, T={T}, alpha={alpha}",
                )


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_casey2012_coarse_grid_matches_fine_reference(alpha):
    """A 512-point grid with a node at 1 um agrees with a fine grid at the probes (rtol 1e-3)."""
    fine = _create_grid_with_probes(1e1, UPPER_LIMIT_AA, 200000, PROBE_WAVELENGTHS_AA)
    coarse = np.unique(np.concatenate([np.geomspace(1e4, 1e8, 512), PROBE_WAVELENGTHS_AA]))
    kwargs = {
        "dust_T": 40.0,
        "dust_beta_ir": DUST_BETA_IR,
        "dust_alpha_mir": alpha,
        "dust_lambda_0_um": DUST_LAMBDA_0_UM,
    }
    got = np.asarray(casey2012(coarse, L_ABSORBED_TEST, **kwargs))
    ref = np.asarray(casey2012(fine, L_ABSORBED_TEST, **kwargs))
    np.testing.assert_allclose(
        got[np.searchsorted(coarse, PROBE_WAVELENGTHS_AA)],
        ref[np.searchsorted(fine, PROBE_WAVELENGTHS_AA)],
        rtol=1e-3,
    )


def test_casey2012_no_emission_below_1um():
    """L_nu is exactly 0 for lambda < 1 um and finite, non-negative elsewhere."""
    wave_aa = np.geomspace(10.0, 1e8, 5000)
    L_nu = np.asarray(
        casey2012(
            wave_aa,
            L_ABSORBED_TEST,
            dust_T=40.0,
            dust_beta_ir=DUST_BETA_IR,
            dust_alpha_mir=1.5,
            dust_lambda_0_um=DUST_LAMBDA_0_UM,
        )
    )
    below_1um = wave_aa < 1e4
    assert np.all(L_nu[below_1um] == 0.0)
    assert np.all(np.isfinite(L_nu[~below_1um]))
    assert np.all(L_nu[~below_1um] >= 0.0)


@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("argnum", [0, 1, 2, 3], ids=["T", "beta", "alpha", "lambda_0"])
def test_casey2012_gradients_finite_and_nonzero(alpha, argnum):
    """d L_nu(350 um) / d {T, beta, alpha_mir, lambda_0} is finite and non-zero."""
    wave_aa = _create_grid_with_probes(1e3, UPPER_LIMIT_AA, 2000, PROBE_WAVELENGTHS_AA)
    probe_index = int(np.searchsorted(wave_aa, 350.0e4))

    def L_nu_single(T, beta, a, lambda0):
        return casey2012(
            wave_aa,
            L_ABSORBED_TEST,
            dust_T=T,
            dust_beta_ir=beta,
            dust_alpha_mir=a,
            dust_lambda_0_um=lambda0,
        )[probe_index]

    grad = jax.grad(L_nu_single, argnums=argnum)(40.0, DUST_BETA_IR, alpha, DUST_LAMBDA_0_UM)
    assert np.isfinite(grad)
    assert grad != 0.0


def test_casey2012_float32_finite():
    """float32 values at alpha = 1.0 are finite, and non-zero from 1 um."""
    code = textwrap.dedent(
        """
        import numpy as np
        import jax
        jax.config.update("jax_enable_x64", False)
        import jax.numpy as jnp
        from tengri.components.dust.emission.analytic._closures import casey2012
        wave = jnp.asarray(np.geomspace(10.0, 1e8, 4000), dtype=jnp.float32)
        out = casey2012(wave, jnp.float32(1e12), dust_T=jnp.float32(40.0),
                        dust_beta_ir=jnp.float32(1.5), dust_alpha_mir=jnp.float32(1.0),
                        dust_lambda_0_um=jnp.float32(200.0))
        assert out.dtype == jnp.float32, out.dtype
        assert bool(jnp.all(jnp.isfinite(out)))
        assert bool(jnp.all(out[wave >= 1e4] > 0.0))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
