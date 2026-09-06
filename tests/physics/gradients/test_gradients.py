# SPDX-License-Identifier: BSD-3-Clause
"""Finite-difference gradient checks for SED model components.

These tests catch sign errors, missing terms, and non-differentiable branches
that isfinite() checks would miss. Each test uses jax.grad + manual FD.

All tests run without SSP data (mock inputs only). Suite completes in < 30 s.

Coverage:
  1. SFH parametric transforms (dpl)
  2. Dust attenuation (two_component_dust)
  3. IGM transmission (igm_transmission)
  4. Nebular marginalization (marginalize_emission_lines_cloudy)
  5. AGN disc SED (multicolor_disc)
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.agn.disc import multicolor_disc
from tengri.components.dust.attenuation import two_component_dust
from tengri.components.igm import igm_transmission
from tengri.components.stellar.sfh.gp_sfh import compute_sqrt_power_drw, gp_from_xi
from tengri.components.stellar.sfh.mean_sfh import double_powerlaw, dpl
from tengri.components.stellar.sfh.psd_models import drw_variance

# Age of the universe today [yr], from the default cosmology — never a
# literal. SFH formation anchor (age_gyr) for dpl/lnorm shape tests.
from tengri.cosmology import age_at_z0 as _age_at_z0
from tengri.observation.eline_priors import marginalize_emission_lines_cloudy
from tengri.utils.grid import grid_spacing, log_age_to_age_yr, make_log_age_grid
from tests._grad_parity import assert_grad_matches_fd

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9


pytestmark = pytest.mark.gradient


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


_EPS = 1e-4
_REL_TOL = 0.005  # 0.5% relative tolerance
N_GRID = 256


def _check_grad_scalar(fn: callable, x0: float, eps: float = _EPS, tol: float = _REL_TOL) -> None:
    """Assert analytic gradient matches finite-difference at x0 (scalar input).

    Parameters
    ----------
    fn : callable
        Scalar function of one parameter.
    x0 : float
        Point at which to check gradient.
    eps : float
        Finite difference step size.
    tol : float
        Relative tolerance (default 0.5%).

    Raises
    ------
    AssertionError
        If analytic and FD gradients differ by > tol.
    """
    g_analytic = float(jax.grad(fn)(x0))
    g_fd = float((fn(x0 + eps) - fn(x0 - eps)) / (2.0 * eps))
    rel_err = abs(g_analytic - g_fd) / (abs(g_fd) + 1e-12)
    assert rel_err < tol, (
        f"Gradient mismatch: analytic={g_analytic:.6f}, FD={g_fd:.6f}, "
        f"rel_err={rel_err:.4f} (tol={tol})"
    )


class TestGradientFiniteDifference:
    """Autodiff vs finite-difference comparison (T5 from roadmap)."""

    @pytest.fixture
    def setup(self):
        log_age_grid = make_log_age_grid(N_GRID)
        d = grid_spacing(log_age_grid)
        sqrt_power = compute_sqrt_power_drw(N_GRID, float(d), 1.0, 50e6)
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))
        return sqrt_power, xi

    def test_gp_grad_vs_finite_diff(self, setup):
        """GP gradient w.r.t. xi matches finite differences."""
        sqrt_power, xi = setup

        def f(xi):
            return jnp.sum(gp_from_xi(xi, sqrt_power, N_GRID) ** 2)

        # Autodiff
        grad_auto = jax.grad(f)(xi)

        # Finite difference (central)
        eps = 1e-5
        grad_fd = jnp.zeros_like(xi)
        for i in range(min(10, N_GRID)):  # Check first 10 components
            xi_plus = xi.at[i].add(eps)
            xi_minus = xi.at[i].add(-eps)
            grad_fd_i = (f(xi_plus) - f(xi_minus)) / (2 * eps)
            assert_allclose(
                float(grad_auto[i]),
                float(grad_fd_i),
                rtol=1e-3,
                err_msg=f"Gradient mismatch at xi[{i}]",
            )


class TestGradientCorrectness:
    """5 core gradient correctness tests using jax.grad vs finite-difference."""

    def test_1_sfh_parametric_transform_gradient(self) -> None:
        """Test 1: SFH parametric transforms must have correct gradients (dpl alpha).

        Tests gradient of double power law SFH wrt the alpha (slope) parameter.
        """
        t = jnp.logspace(7, 10, 100)

        def sfh_sum(alpha: float) -> float:
            return jnp.sum(dpl(t, alpha, beta=1.0, tau=1e9, age=_AGE_UNIV_YR, log_total_mass=10.0))

        _check_grad_scalar(sfh_sum, 1.5, eps=1e-5, tol=0.005)

    def test_1b_dpl_gradient_finite_across_formation_boundary(self) -> None:
        """Regression (#514 follow-up): dpl alpha/beta gradients stay finite when
        the lookback grid extends past the formation anchor (lookback >= age).

        ``dpl`` masks the pre-formation region (T = age - lookback <= 0) with a
        ``jnp.where``. Using ``jnp.inf`` as the masked dummy made ``inf**alpha``
        and its derivative NaN, which leaked through the where VJP and poisoned
        the gradient w.r.t. ``alpha`` for *any* grid point older than ``age``.
        The finite-dummy double-where fix keeps both branches finite. This bit
        the AGN fused photometry gradient test once the age default became the
        cosmology-derived age of the universe.
        """
        # Grid deliberately extends to 14 Gyr, past every plausible age anchor,
        # so a chunk of points fall in the masked T <= 0 region.
        t = jnp.linspace(1e5, 14e9, 200)
        for age in (5.0e9, 13.6e9, _AGE_UNIV_YR):
            g_alpha = jax.grad(
                lambda a, age=age: jnp.sum(
                    dpl(t, alpha=a, beta=1.0, tau=3e9, age=age, log_total_mass=10.0)
                )
            )(1.5)
            g_beta = jax.grad(
                lambda b, age=age: jnp.sum(
                    dpl(t, alpha=1.5, beta=b, tau=3e9, age=age, log_total_mass=10.0)
                )
            )(1.0)
            assert jnp.isfinite(g_alpha), f"dpl ∂/∂alpha non-finite at age={age:.3e}"
            assert jnp.any(g_alpha != 0.0), (
                "`g_alpha` is identically zero — finite is not enough, "
                "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
            )
            assert jnp.isfinite(g_beta), f"dpl ∂/∂beta non-finite at age={age:.3e}"
            assert jnp.any(g_beta != 0.0), (
                "`g_beta` is identically zero — finite is not enough, "
                "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
            )

    def test_2_dust_attenuation_gradient(self) -> None:
        """Test 2: Dust attenuation gradient wrt tau_bc (birth cloud optical depth).

        Tests two-component dust gradient wrt birth cloud V-band optical depth.
        """
        wave = jnp.linspace(3000.0, 8000.0, 50)
        ages = jnp.logspace(6, 10, 30)

        def total_attenuation(tau_bc: float) -> float:
            atten = two_component_dust(
                wave, ages, tau_v1=tau_bc, tau_v2=0.3, law_bc="power_law", law_diff="power_law"
            )
            return jnp.sum(atten)

        _check_grad_scalar(total_attenuation, 0.5, eps=1e-5, tol=0.005)

    def test_3_igm_transmission_gradient(self) -> None:
        """Test 3: IGM transmission gradient wrt redshift (Inoue+2014).

        Tests intergalactic medium absorption gradient wrt source redshift.
        """
        wave_obs = jnp.linspace(800.0, 3000.0, 100)

        def igm_sum(z: float) -> float:
            trans = igm_transmission(wave_obs, z_source=z)
            return jnp.sum(trans)

        _check_grad_scalar(igm_sum, 2.5, eps=1e-4, tol=0.005)

    def test_4_nebular_marginalization_gradient(self) -> None:
        """Test 4: Emission line marginalization ln_L gradient wrt metallicity.

        Tests likelihood gradient wrt gas-phase metallicity log10(Z/Zsun).
        """
        n_pix, n_lines = 40, 5
        key = jax.random.PRNGKey(7)
        residual = jax.random.normal(key, (n_pix,)) * 0.05
        noise = jnp.ones(n_pix) * 0.1
        design_matrix = jax.random.normal(jax.random.PRNGKey(8), (n_pix, n_lines)) * 0.01
        # Provide line wavelengths (vacuum wavelengths in Angstrom)
        line_wavelengths = jnp.array(
            [
                4862.68,  # Hbeta
                5008.24,  # [OIII]5007
                6564.61,  # Halpha
                6717.0,  # [SII]6717
                6731.0,  # [SII]6731
            ]
        )

        def ln_likelihood(log_z: float) -> float:
            result = marginalize_emission_lines_cloudy(
                residual,
                noise,
                design_matrix,
                log_z=log_z,
                neb_logU=-2.5,
                line_wavelengths=line_wavelengths,
                l_hbeta=1.0,
            )
            # Extract ln_likelihood (first return value)
            return result[0] if isinstance(result, (tuple, list)) else result

        _check_grad_scalar(ln_likelihood, 0.0, eps=1e-3, tol=0.005)

    def test_5_agn_disc_gradient(self) -> None:
        """Test 5: AGN disc SED gradient wrt black hole mass (Shakura-Sunyaev).

        Tests multicolor disc SED gradient wrt log10(M_BH / Msun).
        """
        wave = jnp.linspace(100.0, 10000.0, 50)

        def disc_sed_sum(log_mbh: float) -> float:
            sed = multicolor_disc(
                wave,
                # Physical, sub-Eddington L_bol (log10 L_sun): under the
                # luminosity-first parameterization (ADR-0020) the shape depends
                # on M_BH via the derived Eddington ratio. agn_log_lbol=-1.0
                # (0.1 L_sun) clips lambda_Edd to the floor, zeroing
                # d(SED)/d(log_mbh).
                agn_log_lbol=12.0,
                agn_lum_ratio=1.0,
                agn_log_mbh=log_mbh,
                agn_a_spin=0.0,
                agn_cos_inc=0.5,
                n_radii=30,
            )
            # Sum only finite values (filter NaN/Inf)
            return jnp.sum(jnp.where(jnp.isfinite(sed), sed, 0.0))

        _check_grad_scalar(disc_sed_sum, 8.5, eps=1e-4, tol=0.005)


class TestGradientDirection:
    """Verify gradients point in physically correct directions."""

    def test_more_dust_decreases_flux(self):
        """d(total_flux)/d(tau_v1) < 0: more dust = less flux."""
        wave = jnp.linspace(3000, 8000, 50)
        ages = jnp.logspace(6, 10, 30)

        def total_flux(tau_v1):
            return jnp.sum(
                two_component_dust(
                    wave, ages, tau_v1, 0.3, law_bc="power_law", law_diff="power_law"
                )
            )

        grad = assert_grad_matches_fd(total_flux, 0.5)
        assert float(grad) < 0, "More dust should decrease total flux"

    def test_higher_sfr_norm_increases_sfr(self):
        """d(total_SFR)/d(norm) > 0."""
        t = jnp.logspace(7, 10, 100)

        def total_sfr(norm):
            return jnp.sum(double_powerlaw(t, 1.0, 1.0, 1e9, norm))

        grad = assert_grad_matches_fd(total_sfr, 10.0)
        assert float(grad) > 0, "Higher norm should increase total SFR"

    def test_gp_xi_gradient_not_vanishing(self):
        """GP gradients w.r.t. xi are not vanishing."""
        d = (10.14 - 6.0) / (N_GRID - 1)
        sqrt_power = compute_sqrt_power_drw(N_GRID, d, 1.0, 50e6)
        xi = jax.random.normal(jax.random.PRNGKey(42), shape=(N_GRID,))

        def f(xi):
            gp = gp_from_xi(xi, sqrt_power, N_GRID)
            return jnp.sum(gp**2)

        grad = assert_grad_matches_fd(f, xi)

        # Gradient should not be all zeros or all tiny
        grad_rms = float(jnp.sqrt(jnp.mean(grad**2)))
        assert grad_rms > 1e-10, f"GP gradient RMS = {grad_rms}, vanishing"

        # Gradient should not be exploding
        grad_max = float(jnp.max(jnp.abs(grad)))
        assert grad_max < 1e10, f"GP gradient max = {grad_max}, exploding"


class TestGradientThroughPipeline:
    """Test gradient flow through composed pipeline components."""

    def test_full_sfh_pipeline_gradient(self):
        """Gradient flows through PSD -> GP -> mean SFH -> full SFR."""
        log_age_grid = make_log_age_grid(N_GRID)
        d = grid_spacing(log_age_grid)
        age_yr = log_age_to_age_yr(log_age_grid)

        def pipeline(xi, sigma_ps):
            sqrt_power = compute_sqrt_power_drw(N_GRID, float(d), sigma_ps, 50e6)
            gp = gp_from_xi(xi, sqrt_power, N_GRID)
            k0_half = drw_variance(sigma_ps) / 2.0
            sfr_mean = double_powerlaw(age_yr, 1.0, 1.0, 1e9, 10.0)
            sfr = sfr_mean * jnp.exp(gp - k0_half)
            return jnp.sum(jnp.log(sfr + 1e-30))

        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))

        # Gradient w.r.t. xi — check one representative component with FD
        grad_xi = jax.grad(pipeline, argnums=0)(xi, 1.0)
        chex.assert_equal_shape([grad_xi, xi])
        xi0_val = float(xi[0])

        def pipeline_xi0(xi0_scalar):
            xi_mod = xi.at[0].set(xi0_scalar)
            return float(pipeline(xi_mod, 1.0))

        fd_xi0 = (pipeline_xi0(xi0_val + _EPS) - pipeline_xi0(xi0_val - _EPS)) / (2.0 * _EPS)
        assert_allclose(
            float(grad_xi[0]),
            fd_xi0,
            rtol=_REL_TOL,
            err_msg="full_pipeline: FD check ∂/∂xi[0]",
        )

        # Gradient w.r.t. sigma_ps
        _check_grad_scalar(lambda sigma: pipeline(xi, sigma), 1.0, eps=0.01)

    def test_dust_in_pipeline_gradient(self):
        """Gradient flows through SFH -> dust -> attenuated SED."""
        wave = jnp.linspace(3000, 8000, 50)
        ages = jnp.logspace(6, 10, 30)

        def pipeline(tau_v1, tau_v2, dust_n):
            atten = two_component_dust(
                wave,
                ages,
                tau_v1,
                tau_v2,
                law_bc="power_law",
                law_diff="power_law",
                n_slope=dust_n,
            )
            # Simulate CSP: sum over ages, then sum over wavelengths
            return jnp.sum(atten)

        grads = jax.grad(pipeline, argnums=(0, 1, 2))(0.5, 0.3, -0.7)

        # Test tau_v1 gradient
        def pipeline_tau_v1(tau_v1):
            atten = two_component_dust(
                wave, ages, tau_v1, 0.3, law_bc="power_law", law_diff="power_law", n_slope=-0.7
            )
            return jnp.sum(atten)

        grad_jax_tau1 = float(grads[0])
        grad_fd_tau1 = fd_grad(pipeline_tau_v1, 0.5)
        np.testing.assert_allclose(
            grad_jax_tau1,
            grad_fd_tau1,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"tau_v1: autodiff={grad_jax_tau1:.4e}, FD={grad_fd_tau1:.4e}",
        )
        assert float(jnp.abs(grads[0])) > 1e-10, f"Gradient 0 vanishing: {grads[0]}"

        # Test tau_v2 gradient
        def pipeline_tau_v2(tau_v2):
            atten = two_component_dust(
                wave, ages, 0.5, tau_v2, law_bc="power_law", law_diff="power_law", n_slope=-0.7
            )
            return jnp.sum(atten)

        grad_jax_tau2 = float(grads[1])
        grad_fd_tau2 = fd_grad(pipeline_tau_v2, 0.3)
        np.testing.assert_allclose(
            grad_jax_tau2,
            grad_fd_tau2,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"tau_v2: autodiff={grad_jax_tau2:.4e}, FD={grad_fd_tau2:.4e}",
        )
        assert float(jnp.abs(grads[1])) > 1e-10, f"Gradient 1 vanishing: {grads[1]}"

        # Test dust_n gradient
        def pipeline_dust_n(dust_n):
            atten = two_component_dust(
                wave, ages, 0.5, 0.3, law_bc="power_law", law_diff="power_law", n_slope=dust_n
            )
            return jnp.sum(atten)

        grad_jax_n = float(grads[2])
        grad_fd_n = fd_grad(pipeline_dust_n, -0.7)
        np.testing.assert_allclose(
            grad_jax_n,
            grad_fd_n,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"dust_n: autodiff={grad_jax_n:.4e}, FD={grad_fd_n:.4e}",
        )
        assert float(jnp.abs(grads[2])) > 1e-10, f"Gradient 2 vanishing: {grads[2]}"
