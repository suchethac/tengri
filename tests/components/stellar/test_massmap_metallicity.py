# SPDX-License-Identifier: BSD-3-Clause
"""Tests for massmap_lin and massmap_box metallicity modes.

Verifies:
1. Monotonicity: Z(age) is monotonic from Zstart (oldest) to Zfinal (present)
2. Boundary conditions: Z at oldest age ≈ Zstart, Z at present ≈ Zfinal
3. Limiting cases: massmap_box → massmap_lin in small-enrichment limit
4. Gradient safety: autodiff w.r.t. parameters is finite and matches FD
5. Integration: models build and produce finite SEDs via SEDModel
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sfh.metallicity_history import (
    massmap_box_metallicity,
    massmap_lin_metallicity,
)
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-5) -> float:
    """Central finite difference gradient: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_sfh():
    """Simple exponentially declining SFH on a log-age grid.

    Returns
    -------
    ssp_ages_yr : ndarray, shape (n_age,)
        Age in years, ascending lookback order (youngest first).
    sfr_on_ssp : ndarray, shape (n_age,)
        SFR at each age in Msun/yr.
    ssp_lg_age_gyr : ndarray, shape (n_age,)
        log10(age/Gyr) of each age.
    """
    ssp_lg_age_gyr = jnp.linspace(-3.0, 1.114, 20)  # 1 Myr to ~13 Gyr
    ssp_ages_yr = 10.0 ** (ssp_lg_age_gyr + 9.0)
    # Exponentially declining SFH: SFR ~ exp(-age / tau)
    # Reversal to get lookback time, then normalize
    tau = 5.0e9  # 5 Gyr timescale
    sfr_on_ssp = jnp.exp(-ssp_ages_yr / tau)
    sfr_on_ssp = sfr_on_ssp / jnp.sum(sfr_on_ssp)  # Normalize
    return ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr


@pytest.fixture
def log_z_abs_values():
    """Typical absolute log10(Z) range.

    Returns
    -------
    log_z_start : float
        log10(Z) at oldest age (e.g., 1e-4 in linear space).
    log_z_final : float
        log10(Z) at present day (e.g., 0.02 in linear space).
    """
    # 1e-4 in linear space
    log_z_start = jnp.log10(1e-4)
    # 0.02 in linear space (roughly solar)
    log_z_final = jnp.log10(0.02)
    return log_z_start, log_z_final


# ── Tests: massmap_lin ────────────────────────────────────────────


class TestMassmapLinMetallicity:
    """Tests for linear massmap metallicity evolution."""

    def test_output_shape(self, simple_sfh, log_z_abs_values):
        """Output shape matches input SSP age grid."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_lin_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        assert result.shape == ssp_lg_age_gyr.shape

    def test_monotonic_decreasing(self, simple_sfh, log_z_abs_values):
        """Z(age) is monotonically decreasing from youngest to oldest.

        Since cmf goes from 1 (present) to 0 (oldest), and Z =
        Zstart + (Zfinal - Zstart) * cmf, Z must decrease monotonically
        from Zfinal (youngest) to Zstart (oldest).
        """
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_lin_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        # Convert back to linear Z to check sign (should be monotonically decreasing)
        z_linear = 10.0**result
        dz = jnp.diff(z_linear)
        # Allow for small numerical noise
        tolerance = 1e-14 * jnp.mean(z_linear)
        assert jnp.all(dz <= tolerance), f"Non-monotonic: dz = {dz}, tolerance = {tolerance}"

    @pytest.mark.bounds
    def test_boundary_conditions(self, simple_sfh, log_z_abs_values):
        """Z at oldest age ≈ Zstart, at present ≈ Zfinal."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_lin_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        # Oldest age is last element in ascending lookback order
        z_oldest = result[-1]
        z_youngest = result[0]
        # Should be close to boundaries (within a few dex due to rounding)
        assert_allclose(z_oldest, log_z_start, atol=0.1)
        assert_allclose(z_youngest, log_z_final, atol=0.1)

    def test_finite_values(self, simple_sfh, log_z_abs_values):
        """All output values are finite (no NaN or inf)."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_lin_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        assert jnp.all(jnp.isfinite(result)), f"Non-finite values: {result}"

    def test_gradient_wrt_zfinal(self, simple_sfh, log_z_abs_values):
        """Gradient w.r.t. Zfinal is finite and matches finite difference."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values

        def fn(z_final):
            result = massmap_lin_metallicity(
                ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, z_final
            )
            return jnp.mean(result)

        # Analytical gradient via autodiff
        grad_auto = float(jax.grad(fn)(log_z_final))
        # Finite difference
        grad_fd = fd_grad(fn, log_z_final)
        assert np.isfinite(grad_auto), f"Gradient is {grad_auto}"
        assert_allclose(grad_auto, grad_fd, rtol=1e-3)

    def test_zero_sfr_safe(self, log_z_abs_values):
        """Zero SFR handled safely (no division by zero)."""
        ssp_lg_age_gyr = jnp.linspace(-3.0, 1.114, 10)
        ssp_ages_yr = 10.0 ** (ssp_lg_age_gyr + 9.0)
        sfr_on_ssp = jnp.zeros_like(ssp_ages_yr)  # All zero
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_lin_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        assert jnp.all(jnp.isfinite(result))


# ── Tests: massmap_box ────────────────────────────────────────────


class TestMassmapBoxMetallicity:
    """Tests for closed-box massmap metallicity evolution."""

    def test_output_shape(self, simple_sfh, log_z_abs_values):
        """Output shape matches input SSP age grid."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_box_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        assert result.shape == ssp_lg_age_gyr.shape

    def test_monotonic_decreasing(self, simple_sfh, log_z_abs_values):
        """Z(age) is monotonically decreasing from youngest to oldest (box model)."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_box_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        z_linear = 10.0**result
        dz = jnp.diff(z_linear)
        tolerance = 1e-14 * jnp.mean(z_linear)
        assert jnp.all(dz <= tolerance), f"Non-monotonic: dz = {dz}, tolerance = {tolerance}"

    @pytest.mark.bounds
    def test_boundary_conditions(self, simple_sfh, log_z_abs_values):
        """Z at oldest age ≈ Zstart, at present ≈ Zfinal."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_box_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        z_oldest = result[-1]
        z_youngest = result[0]
        assert_allclose(z_oldest, log_z_start, atol=0.1)
        assert_allclose(z_youngest, log_z_final, atol=0.1)

    def test_finite_values(self, simple_sfh, log_z_abs_values):
        """All output values are finite."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values
        result = massmap_box_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        assert jnp.all(jnp.isfinite(result))

    def test_gradient_wrt_zfinal(self, simple_sfh, log_z_abs_values):
        """Gradient w.r.t. Zfinal is finite and matches FD."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values

        def fn(z_final):
            result = massmap_box_metallicity(
                ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, z_final
            )
            return jnp.mean(result)

        grad_auto = float(jax.grad(fn)(log_z_final))
        grad_fd = fd_grad(fn, log_z_final)
        assert np.isfinite(grad_auto)
        assert_allclose(grad_auto, grad_fd, rtol=1e-3)

    @pytest.mark.limit
    def test_small_enrichment_limit(self, simple_sfh):
        """In small-enrichment limit, massmap_box ≈ massmap_lin.

        When (Zfinal - Zstart) << yield, the box model should
        approximately reduce to the linear model.
        """
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        # Small enrichment in linear space
        z_start = 1e-4
        z_final = 1e-4 + 1e-5  # Only 10% enrichment
        log_z_start = jnp.log10(z_start)
        log_z_final = jnp.log10(z_final)
        yield_rho = 0.03  # Large relative to enrichment

        result_lin = massmap_lin_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final
        )
        result_box = massmap_box_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final, yield_rho
        )
        # In the small-enrichment limit, they should be similar
        # (within ~5% relative error in Z space)
        z_lin = 10.0**result_lin
        z_box = 10.0**result_box
        rel_err = jnp.abs(z_box - z_lin) / (z_lin + 1e-20)
        assert jnp.all(rel_err < 0.1), f"Limit test failed: max rel_err = {jnp.max(rel_err)}"

    def test_yield_effect(self, simple_sfh, log_z_abs_values):
        """Smaller yield slows metallicity growth (log-linear effect)."""
        ssp_ages_yr, sfr_on_ssp, ssp_lg_age_gyr = simple_sfh
        log_z_start, log_z_final = log_z_abs_values

        result_large_yield = massmap_box_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final, yield_rho=0.05
        )
        result_small_yield = massmap_box_metallicity(
            ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start, log_z_final, yield_rho=0.01
        )
        # With smaller yield, metallicity grows more slowly
        # At intermediate ages, large_yield should be higher
        z_large = 10.0**result_large_yield
        z_small = 10.0**result_small_yield
        mid_idx = len(ssp_lg_age_gyr) // 2
        assert float(z_large[mid_idx]) > float(z_small[mid_idx])


# ── Tests: Integration via Parameters ──────────────────────────


class TestMassmapIntegration:
    """Tests for massmap modes integrated into Parameters."""

    @pytest.mark.contract
    def test_massmap_lin_explicit_mode(self):
        """Explicitly setting met_mode='massmap_lin' builds correctly with free params."""
        params = Parameters(
            met_mode="massmap_lin",
            met_logzsol_start=Uniform(-4.0, -2.0),
            met_logzsol_final=Uniform(-2.0, 0.0),
        )
        assert params.met_mode == "massmap_lin"
        # Both params should be free with the uniform priors
        free_params = params.free_params
        assert any("met_logzsol_start" in p for p in free_params)
        assert any("met_logzsol_final" in p for p in free_params)

    @pytest.mark.contract
    def test_massmap_box_with_yield(self):
        """Explicitly set massmap_box with yield parameter."""
        # When combining start, final, and yield, explicitly set met_mode to avoid ambiguity
        params = Parameters(
            met_mode="massmap_box",
            met_logzsol_start=Uniform(-4.0, -2.0),
            met_yield=Fixed(0.03),
        )
        assert params.met_mode == "massmap_box"
        # met_logzsol_start should be free
        free_params = params.free_params
        assert any("met_logzsol_start" in p for p in free_params)


@pytest.mark.regression_bug
def test_massmap_lin_is_linear_in_Z_not_logZ():
    """massmap_lin maps Z linearly in cumulative mass (ProSpect Zfunc_massmap_lin).

    Regression for the log-vs-linear bug: the original code interpolated in
    log10(Z), giving a *geometric* map (~2x off vs ProSpect at the half-mass
    point). For a constant SFR on a uniform age grid, cmf is linear in age, so Z
    at the mid-age must be the *arithmetic* midpoint (Zstart+Zfinal)/2 — not the
    geometric mean sqrt(Zstart*Zfinal).
    """
    n = 401
    ages_yr = jnp.linspace(1.0e6, 13.0e9, n)  # uniform grid
    lg_age_gyr = jnp.log10(ages_yr / 1e9)
    sfr = jnp.ones(n)  # constant SFR -> cmf linear in age
    z_start, z_final = 1.0e-4, 2.0e-2
    log_z = np.asarray(
        massmap_lin_metallicity(
            lg_age_gyr, ages_yr, sfr, float(np.log10(z_start)), float(np.log10(z_final))
        )
    )
    z = 10.0**log_z
    z_mid = float(z[n // 2])  # cmf ~ 0.5
    arithmetic = 0.5 * (z_start + z_final)
    geometric = float(np.sqrt(z_start * z_final))
    (
        assert_allclose(z_mid, arithmetic, rtol=0.02),
        (
            f"massmap_lin at half-mass Z={z_mid:.3e} should be the arithmetic midpoint "
            f"{arithmetic:.3e} (ProSpect linear map), not the geometric {geometric:.3e}"
        ),
    )
    # Endpoints: present-day -> Zfinal, oldest -> Zstart.
    assert_allclose(z[0], z_final, rtol=1e-3)
    assert_allclose(z[-1], z_start, rtol=5e-2)


@pytest.mark.regression_bug
def test_massmap_box_builds_via_group_dict_grammar(synthetic_ssp):
    """massmap_box is reachable through SEDModel.build's dict grammar.

    Regression for two coupled bugs: (1) ProSpect's ``yield`` param is a Python
    keyword and could not be a builder group key — renamed to ``met_yield``;
    (2) inference raised "ambiguous" (massmap_box's keys are a superset of
    massmap_lin's) even when ``met_mode`` was set explicitly. Either one made
    ``SEDModel.build(stellar={'met_mode': 'massmap_box', ...})`` crash.
    """
    from tengri import FIXED, Fixed, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        stellar={
            "met_mode": "massmap_box",
            "met_logzsol_start": Fixed(-2.15),
            "met_logzsol_final": Fixed(0.15),
            "met_yield": Fixed(0.03),
            "*": FIXED,
        },
        sfh={"type": "const", "log_total_mass": Fixed(10.0), "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
    )
    state = model.predict_state({})
    z_hist = np.asarray(state.derived["log_metallicity_history"])
    assert np.isfinite(z_hist).all()
    # Monotonic enrichment: present-day (youngest) >= oldest.
    age = np.asarray(state.derived["sfh_grid_lbt_yr"])
    assert z_hist[np.argmin(age)] >= z_hist[np.argmax(age)]
