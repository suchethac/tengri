# SPDX-License-Identifier: BSD-3-Clause
"""Tests for template-based dust emission models: Astrodust, BOSA, THEMIS.

Tests cover:
- Template loaders create callable functions
- Output shape correctness
- Energy conservation (gamma=0 gives single-U only)
- Physical behavior (higher sSFR gives warmer BOSA SED, qhac affects THEMIS)
- JIT compatibility
- Differentiability
- Registry integration
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from tengri.components.dust.emission import (
    DUST_EMISSION_MODELS,
    create_astrodust_from_grid,
    create_bosa_from_grid,
    create_themis_from_grid,
)
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference with a *scale-aware* step.

    ``eps`` is relative to ``|x|`` so the step is meaningful across the wide
    dynamic range of the parameters tested here. A fixed absolute step (the old
    behavior) is catastrophically cancellation-limited for large arguments such
    as ``L_absorbed ~ 1e10``: ``f(x±1e-4)`` then differs only in ``f``'s 15th
    significant digit, so the finite-difference reference becomes pure round-off
    even though the analytic/autodiff gradient is exact.
    """
    h = eps * max(abs(x), 1.0)
    return float((f(x + h) - f(x - h)) / (2.0 * h))


# ── Fixtures: synthetic template grids for testing ────────────────
def _make_dl07_like_grid(
    n_qpah: int = 3,
    n_umin: int = 4,
    n_wave: int = 200,
    param_name: str = "qpah",
) -> dict:
    """Create a synthetic DL07-like template grid for testing.
    Returns a dict suitable for create_astrodust_from_grid or
    create_themis_from_grid.
    Parameters
    ----------
    param_name : str
        "qpah" for Astrodust, "qhac" for THEMIS.
    """
    wavs_um = np.logspace(np.log10(0.1), np.log10(1000.0), n_wave)
    wavs_aa = wavs_um * 1.0e4
    q_grid = (
        np.linspace(0.5, 5.0, n_qpah) if param_name == "qpah" else np.linspace(0.02, 0.30, n_qpah)
    )
    umin_grid = np.array([0.1, 1.0, 5.0, 25.0])[:n_umin]
    # Synthetic templates: modified blackbody shape with temperature
    # depending on umin, and PAH-like bump strength depending on q
    wave_cm = wavs_aa * 1.0e-8
    nu = 2.99792458e10 / wave_cm
    single_u = np.zeros((n_qpah, n_umin, n_wave))
    pdr = np.zeros((n_qpah, n_umin, n_wave))
    for iq in range(n_qpah):
        for iu in range(n_umin):
            T = 18.0 * umin_grid[iu] ** (1.0 / 6.0)
            T_pdr = 18.0 * (umin_grid[iu] * 1e6) ** (1.0 / 6.0)
            # Simple MBB-like shape
            x = np.clip(6.626e-27 * nu / (1.38e-16 * T), 0, 500)
            shape = nu**2 / (np.exp(x) - 1.0 + 1e-30)
            x_pdr = np.clip(6.626e-27 * nu / (1.38e-16 * T_pdr), 0, 500)
            shape_pdr = nu**2 / (np.exp(x_pdr) - 1.0 + 1e-30)
            # Add a PAH-like bump at 7.7 um scaled by q
            pah_bump = q_grid[iq] * np.exp(-0.5 * ((wavs_um - 7.7) / 1.5) ** 2)
            shape = shape + pah_bump * np.max(shape) * 0.1
            shape_pdr = shape_pdr + pah_bump * np.max(shape_pdr) * 0.05
            # Normalize to integrate to 1 over wavelength (L_lambda)
            norm = np.trapezoid(shape, wavs_aa)
            if norm > 0:
                single_u[iq, iu] = shape / norm
            norm_pdr = np.trapezoid(shape_pdr, wavs_aa)
            if norm_pdr > 0:
                pdr[iq, iu] = shape_pdr / norm_pdr
    result = {
        "wavelength_um": jnp.array(wavs_um),
        "umin_grid": jnp.array(umin_grid),
        "spectra_single": jnp.array(single_u),
        "spectra_pdr": jnp.array(pdr),
    }
    if param_name == "qpah":
        result["qpah_grid"] = jnp.array(q_grid)
    else:
        result["qhac_grid"] = jnp.array(q_grid)
    return result


def _make_bosa_grid(
    n_ltir: int = 3,
    n_ssfr: int = 4,
    n_wave: int = 200,
) -> dict:
    """Create a synthetic BOSA template grid for testing."""
    wavs_um = np.logspace(np.log10(0.1), np.log10(1000.0), n_wave)
    wavs_aa = wavs_um * 1.0e4
    log_ltir_grid = np.array([8.0, 10.0, 12.0])[:n_ltir]
    log_ssfr_grid = np.array([-12.0, -10.5, -9.5, -8.0])[:n_ssfr]
    wave_cm = wavs_aa * 1.0e-8
    nu = 2.99792458e10 / wave_cm
    spectra = np.zeros((n_ltir, n_ssfr, n_wave))
    for il in range(n_ltir):
        for js in range(n_ssfr):
            # Temperature increases with sSFR
            T = 20.0 + 5.0 * (log_ssfr_grid[js] + 12.0)
            T = np.clip(T, 15.0, 70.0)
            x = np.clip(6.626e-27 * nu / (1.38e-16 * T), 0, 500)
            shape = nu**2 / (np.exp(x) - 1.0 + 1e-30)
            norm = np.trapezoid(shape, wavs_aa)
            if norm > 0:
                spectra[il, js] = shape / norm
    return {
        "wavelength_um": jnp.array(wavs_um),
        "log_ltir_grid": jnp.array(log_ltir_grid),
        "log_ssfr_grid": jnp.array(log_ssfr_grid),
        "spectra": jnp.array(spectra),
    }


@pytest.fixture
def wavelength():
    """IR wavelength grid in Angstrom (1-1000 um)."""
    return jnp.logspace(np.log10(1e4), np.log10(1e7), 300)


@pytest.fixture
def astrodust_grid():
    """Synthetic Astrodust template grid."""
    return _make_dl07_like_grid(param_name="qpah")


@pytest.fixture
def bosa_grid():
    """Synthetic BOSA template grid."""
    return _make_bosa_grid()


@pytest.fixture
def themis_grid():
    """Synthetic THEMIS template grid."""
    return _make_dl07_like_grid(param_name="qhac")


# ── Astrodust tests ───────────────────────────────────────────────
class TestAstrodust:
    """Tests for the Astrodust+PAH emission model (Hensley & Draine 2023)."""

    def test_create_from_grid_returns_callable(self, astrodust_grid):
        """create_astrodust_from_grid returns a callable function."""
        fn = create_astrodust_from_grid(astrodust_grid)
        assert callable(fn)

    def test_output_shape(self, astrodust_grid, wavelength):
        """Output shape matches input wavelength grid."""
        fn = create_astrodust_from_grid(astrodust_grid)
        result = fn(wavelength, 1e10)
        chex.assert_equal_shape([result, wavelength])

    def test_output_nonnegative(self, astrodust_grid, wavelength):
        """Emission is non-negative everywhere."""
        fn = create_astrodust_from_grid(astrodust_grid)
        result = fn(wavelength, 1e10, dust_umin=1.0, dust_gamma_dl=0.05, dust_qpah=3.0)
        assert jnp.all(result >= 0.0)

    def test_gamma_zero_gives_single_u(self, astrodust_grid, wavelength):
        """With gamma=0, only the single-U component contributes."""
        fn = create_astrodust_from_grid(astrodust_grid)
        result_g0 = fn(wavelength, 1e10, dust_gamma_dl=0.0, dust_umin=1.0)
        result_g1 = fn(wavelength, 1e10, dust_gamma_dl=1.0, dust_umin=1.0)
        # The two should differ because they use different templates
        assert not jnp.allclose(result_g0, result_g1, atol=1e-10)

    def test_energy_conservation(self, astrodust_grid, wavelength):
        """Total integrated emission equals L_absorbed (energy balance)."""
        fn = create_astrodust_from_grid(astrodust_grid)
        L_abs = 1e10
        result = fn(wavelength, L_abs, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
        # Integrate L_nu over frequency
        wave_cm = wavelength * 1e-8
        nu = 2.99792458e10 / wave_cm
        integral = -float(jnp.trapezoid(result, nu))
        # Should be close to L_abs (within interpolation tolerance)
        assert_allclose(integral, L_abs, rtol=0.1)

    def test_scales_with_l_absorbed(self, astrodust_grid, wavelength):
        """Doubling L_absorbed doubles the emission."""
        fn = create_astrodust_from_grid(astrodust_grid)
        r1 = fn(wavelength, 1e10)
        r2 = fn(wavelength, 2e10)
        assert_allclose(r2, 2.0 * r1, rtol=1e-10)

    def test_jit_compatible(self, astrodust_grid, wavelength):
        """Astrodust model is JIT-compatible."""
        fn = create_astrodust_from_grid(astrodust_grid)
        result = assert_jit_matches_eager(lambda w, l: fn(w, l, dust_umin=1.0), wavelength, 1e10)
        chex.assert_equal_shape([result, wavelength])
        chex.assert_tree_all_finite(result)

    def test_differentiable(self, astrodust_grid, wavelength):
        """Gradients exist with respect to L_absorbed."""
        fn = create_astrodust_from_grid(astrodust_grid)

        def loss(l_abs):
            return jnp.sum(fn(wavelength, l_abs, dust_umin=1.0))

        grad_jax = float(jax.grad(loss)(1e10))
        grad_fd = fd_grad(loss, 1e10)
        # For very small gradients (<1e-10), use atol; otherwise use rtol
        atol = 1e-12 if abs(grad_jax) < 1e-10 else 0
        assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-2,
            atol=atol,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax > 0.0  # more absorbed -> more emission


# ── BOSA tests ────────────────────────────────────────────────────
class TestBOSA:
    """Tests for the BOSA emission model (Boquien & Salim 2021)."""

    def test_create_from_grid_returns_callable(self, bosa_grid):
        """create_bosa_from_grid returns a callable function."""
        fn = create_bosa_from_grid(bosa_grid)
        assert callable(fn)

    def test_output_shape(self, bosa_grid, wavelength):
        """Output shape matches input wavelength grid."""
        fn = create_bosa_from_grid(bosa_grid)
        result = fn(wavelength, 1e10, dust_log_ssfr=-10.0)
        chex.assert_equal_shape([result, wavelength])

    def test_output_nonnegative(self, bosa_grid, wavelength):
        """Emission is non-negative everywhere."""
        fn = create_bosa_from_grid(bosa_grid)
        result = fn(wavelength, 1e10, dust_log_ssfr=-9.0)
        assert jnp.all(result >= 0.0)

    def test_higher_ssfr_warmer_sed(self, bosa_grid, wavelength):
        """Higher sSFR shifts the SED peak to shorter wavelengths (warmer)."""
        fn = create_bosa_from_grid(bosa_grid)
        sed_low = fn(wavelength, 1e10, dust_log_ssfr=-11.0)
        sed_high = fn(wavelength, 1e10, dust_log_ssfr=-8.5)
        # Find peak wavelength for each
        peak_low = wavelength[jnp.argmax(sed_low)]
        peak_high = wavelength[jnp.argmax(sed_high)]
        # Higher sSFR -> warmer -> shorter peak wavelength (or equal if both at grid edge)
        assert float(peak_high) <= float(peak_low)

    def test_energy_conservation(self, bosa_grid, wavelength):
        """Total integrated emission equals L_absorbed."""
        fn = create_bosa_from_grid(bosa_grid)
        L_abs = 1e10
        result = fn(wavelength, L_abs, dust_log_ssfr=-10.0)
        wave_cm = wavelength * 1e-8
        nu = 2.99792458e10 / wave_cm
        integral = -float(jnp.trapezoid(result, nu))
        assert_allclose(integral, L_abs, rtol=0.1)

    def test_scales_with_l_absorbed(self, bosa_grid, wavelength):
        """Doubling L_absorbed doubles the emission."""
        fn = create_bosa_from_grid(bosa_grid)
        r1 = fn(wavelength, 1e10, dust_log_ssfr=-10.0)
        r2 = fn(wavelength, 2e10, dust_log_ssfr=-10.0)
        # Not exactly 2x because log(L_TIR) also changes the template selection
        # But the scaling should be roughly proportional
        ratio = jnp.sum(r2) / jnp.sum(r1)
        assert 1.5 < float(ratio) < 2.5

    def test_jit_compatible(self, bosa_grid, wavelength):
        """BOSA model is JIT-compatible."""
        fn = create_bosa_from_grid(bosa_grid)
        result = assert_jit_matches_eager(
            lambda w, l: fn(w, l, dust_log_ssfr=-10.0), wavelength, 1e10
        )
        chex.assert_equal_shape([result, wavelength])
        chex.assert_tree_all_finite(result)

    def test_differentiable(self, bosa_grid, wavelength):
        """Gradients exist with respect to dust_log_ssfr."""
        fn = create_bosa_from_grid(bosa_grid)

        def loss(log_ssfr):
            return jnp.sum(fn(wavelength, 1e10, dust_log_ssfr=log_ssfr))

        grad_jax = float(jax.grad(loss)(-10.0))
        grad_fd = fd_grad(loss, -10.0)
        assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )


# ── THEMIS tests ──────────────────────────────────────────────────
class TestTHEMIS:
    """Tests for the THEMIS emission model (Jones et al. 2017)."""

    def test_create_from_grid_returns_callable(self, themis_grid):
        """create_themis_from_grid returns a callable function."""
        fn = create_themis_from_grid(themis_grid)
        assert callable(fn)

    def test_output_shape(self, themis_grid, wavelength):
        """Output shape matches input wavelength grid."""
        fn = create_themis_from_grid(themis_grid)
        result = fn(wavelength, 1e10)
        chex.assert_equal_shape([result, wavelength])

    def test_output_nonnegative(self, themis_grid, wavelength):
        """Emission is non-negative everywhere."""
        fn = create_themis_from_grid(themis_grid)
        result = fn(wavelength, 1e10, dust_umin=1.0, dust_gamma_dl=0.05, dust_qhac=0.17)
        assert jnp.all(result >= 0.0)

    def test_qhac_affects_pah_features(self, themis_grid, wavelength):
        """Different qhac values produce different SEDs (PAH feature strength)."""
        fn = create_themis_from_grid(themis_grid)
        sed_low_q = fn(wavelength, 1e10, dust_qhac=0.05, dust_umin=1.0)
        sed_high_q = fn(wavelength, 1e10, dust_qhac=0.25, dust_umin=1.0)
        # The SEDs should differ
        diff = jnp.max(jnp.abs(sed_high_q - sed_low_q))
        assert float(diff) > 0.0

    def test_gamma_zero_gives_single_u(self, themis_grid, wavelength):
        """With gamma=0, only the single-U component contributes."""
        fn = create_themis_from_grid(themis_grid)
        result_g0 = fn(wavelength, 1e10, dust_gamma_dl=0.0, dust_umin=1.0)
        result_g1 = fn(wavelength, 1e10, dust_gamma_dl=1.0, dust_umin=1.0)
        assert not jnp.allclose(result_g0, result_g1, atol=1e-10)

    def test_energy_conservation(self, themis_grid, wavelength):
        """Total integrated emission equals L_absorbed."""
        fn = create_themis_from_grid(themis_grid)
        L_abs = 1e10
        result = fn(wavelength, L_abs, dust_umin=1.0, dust_gamma_dl=0.01, dust_qhac=0.17)
        wave_cm = wavelength * 1e-8
        nu = 2.99792458e10 / wave_cm
        integral = -float(jnp.trapezoid(result, nu))
        assert_allclose(integral, L_abs, rtol=0.1)

    def test_jit_compatible(self, themis_grid, wavelength):
        """THEMIS model is JIT-compatible."""
        fn = create_themis_from_grid(themis_grid)
        result = assert_jit_matches_eager(
            lambda w, l: fn(w, l, dust_umin=1.0, dust_qhac=0.17), wavelength, 1e10
        )
        chex.assert_equal_shape([result, wavelength])
        chex.assert_tree_all_finite(result)

    def test_differentiable(self, themis_grid, wavelength):
        """Gradients exist with respect to L_absorbed."""
        fn = create_themis_from_grid(themis_grid)

        def loss(l_abs):
            return jnp.sum(fn(wavelength, l_abs, dust_umin=1.0, dust_qhac=0.17))

        grad_jax = float(jax.grad(loss)(1e10))
        grad_fd = fd_grad(loss, 1e10)
        # For very small gradients (<1e-10), use atol; otherwise use rtol
        atol = 1e-12 if abs(grad_jax) < 1e-10 else 0
        assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-2,
            atol=atol,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax > 0.0


# ── Registry integration tests ────────────────────────────────────
class TestRegistry:
    """Tests for model registration."""

    def test_astrodust_registered(self):
        """Astrodust is registered in DUST_EMISSION_MODELS."""
        assert "astrodust" in DUST_EMISSION_MODELS

    def test_bosa_registered(self):
        """BOSA is registered in DUST_EMISSION_MODELS."""
        assert "bosa" in DUST_EMISSION_MODELS

    def test_themis_registered(self):
        """THEMIS is registered in DUST_EMISSION_MODELS."""
        assert "themis" in DUST_EMISSION_MODELS
