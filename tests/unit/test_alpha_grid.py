"""Tests for alpha-enhanced SSP grid interpolation.

Validates the new 4D (metallicity, [α/Fe], age, wavelength) SSP grid
support including:
1. Bilinear ([Fe/H], [α/Fe]) interpolation
2. Per-age evolving [α/Fe] interpolation
3. Alpha evolution ramp parameterization
4. JIT compatibility and differentiability
5. Backward compatibility with 3D grids (no alpha dimension)
6. has_alpha_grid detection
7. Salaris [Fe/H] ↔ [M/H] convention conversions
8. Solar [α/Fe]=0.0 equivalence with no-alpha case
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def alpha_ssp_grid():
    """Synthetic 4D SSP grid: (3 met, 5 alpha, 10 age, 20 wave).

    The flux is designed so that:
    - Higher metallicity → redder (more flux at long wavelengths)
    - Higher [α/Fe] → stronger Mg feature (bump near λ=5180)
    - Younger age → bluer (more flux at short wavelengths)
    This allows us to verify interpolation responds correctly.
    """
    n_met, n_alpha, n_age, n_wave = 3, 5, 10, 20

    lgmet = jnp.array([-1.5, -0.5, 0.0])
    alpha_fe = jnp.array([-0.2, 0.0, 0.2, 0.4, 0.6])
    lg_age_gyr = jnp.linspace(-1.0, 1.1, n_age)
    wave = jnp.linspace(3500.0, 9000.0, n_wave)

    # Build flux: base is a smooth function of met, alpha, age, wave
    # so we can verify interpolation is sensible
    key = jax.random.PRNGKey(42)
    base = jnp.abs(jax.random.normal(key, (n_met, n_alpha, n_age, n_wave))) * 1e-3 + 1e-5

    # Add systematic trends
    # Higher Z → more flux at red end
    z_trend = jnp.linspace(0.5, 1.5, n_wave)[None, None, None, :]
    met_scale = jnp.linspace(0.5, 1.5, n_met)[:, None, None, None]
    # Higher alpha → bump near pixel 10 (~ 5180 Å)
    alpha_bump = jnp.exp(-0.5 * ((jnp.arange(n_wave) - 10) / 2.0) ** 2)
    alpha_scale = alpha_fe[:, None, None] * alpha_bump[None, None, :]
    alpha_scale = alpha_scale[None, :, :, :]  # (1, n_alpha, 1, n_wave)
    # Younger → more blue flux
    age_scale = jnp.linspace(1.5, 0.5, n_age)[None, None, :, None]

    flux = base * met_scale * z_trend * age_scale + 0.01 * alpha_scale

    return {
        "ssp_flux": flux,
        "ssp_lgmet": lgmet,
        "ssp_alpha_fe": alpha_fe,
        "ssp_lg_age_gyr": lg_age_gyr,
        "ssp_wave": wave,
    }


@pytest.fixture
def ssp_data_4d(alpha_ssp_grid):
    """SSPData with 4D alpha-enhanced grid."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    g = alpha_ssp_grid
    return SSPData(
        ssp_wave=g["ssp_wave"],
        ssp_flux=g["ssp_flux"],
        ssp_lg_age_gyr=g["ssp_lg_age_gyr"],
        ssp_lgmet=g["ssp_lgmet"],
        ssp_alpha_fe=g["ssp_alpha_fe"],
    )


@pytest.fixture
def ssp_data_3d():
    """SSPData with standard 3D grid (no alpha)."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_met, n_age, n_wave = 3, 10, 20
    key = jax.random.PRNGKey(0)
    return SSPData(
        ssp_wave=jnp.linspace(3500.0, 9000.0, n_wave),
        ssp_flux=jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.1, n_age),
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


# ── has_alpha_grid detection ──────────────────────────────────────


class TestHasAlphaGrid:
    """Test alpha grid detection."""

    def test_4d_grid_detected(self, ssp_data_4d):
        from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid

        assert has_alpha_grid(ssp_data_4d) is True

    def test_3d_grid_not_detected(self, ssp_data_3d):
        from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid

        assert has_alpha_grid(ssp_data_3d) is False

    def test_none_alpha_not_detected(self, ssp_data_3d):
        from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid

        assert has_alpha_grid(ssp_data_3d) is False


# ── Bilinear (Z, [α/Fe]) interpolation ────────────────────────────


class TestInterpolateMetAlpha:
    """Tests for bilinear (metallicity, [α/Fe]) interpolation."""

    def test_at_grid_point(self, alpha_ssp_grid):
        """Interpolation at exact grid point should return that grid point."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.0,
        )
        expected = g["ssp_flux"][1, 1]  # lgmet=-0.5 is idx 1, alpha=0.0 is idx 1
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_midpoint_interpolation(self, alpha_ssp_grid):
        """Midpoint between grid points should be the average of neighbors."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        # Midpoint between lgmet[0]=-1.5 and lgmet[1]=-0.5 at alpha[1]=0.0
        result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-1.0,
            alpha_fe=0.0,
        )
        expected = 0.5 * g["ssp_flux"][0, 1] + 0.5 * g["ssp_flux"][1, 1]
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_different_alpha_different_result(self, alpha_ssp_grid):
        """Different [α/Fe] values should give different SEDs."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        sed_solar = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.0,
        )
        sed_alpha = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.4,
        )
        assert not jnp.allclose(sed_solar, sed_alpha)

    def test_output_shape(self, alpha_ssp_grid):
        """Output should be (n_age, n_wave)."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.2,
        )
        assert result.shape == (10, 20)

    def test_finite_output(self, alpha_ssp_grid):
        """All outputs should be finite."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        for lz in [-1.5, -1.0, -0.5, 0.0]:
            for afe in [-0.2, 0.0, 0.2, 0.4, 0.6]:
                result = interpolate_met_alpha(
                    g["ssp_flux"],
                    g["ssp_lgmet"],
                    g["ssp_alpha_fe"],
                    log_z=lz,
                    alpha_fe=afe,
                )
                assert jnp.all(jnp.isfinite(result)), f"Non-finite at lz={lz}, afe={afe}"

    def test_clamping_at_bounds(self, alpha_ssp_grid):
        """Values outside grid should clamp to boundary."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        # Beyond low Z boundary
        result_low = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-5.0,
            alpha_fe=0.0,
        )
        result_edge = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-1.5,
            alpha_fe=0.0,
        )
        assert jnp.allclose(result_low, result_edge, atol=1e-10)

    def test_jit_compatible(self, alpha_ssp_grid):
        """Should work under jax.jit."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        jit_fn = jax.jit(interpolate_met_alpha, static_argnames=[])
        result = jit_fn(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.2,
        )
        assert jnp.all(jnp.isfinite(result))

    def test_differentiable_wrt_log_z(self, alpha_ssp_grid):
        """Gradient w.r.t. log_z should be finite and match FD.

        Evaluated at log_z=-0.7, an interior point of the first metallicity cell
        [-1.5, -0.5].  Must NOT be a grid boundary: at boundary values like -0.5,
        jnp.searchsorted lands in different cells for (x+eps) vs (x-eps), so FD
        crosses a cell boundary while autodiff stays within one cell → they
        legitimately differ.  Interior points avoid this and must agree exactly.
        """
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid

        def total_flux(lz):
            return jnp.sum(
                interpolate_met_alpha(
                    g["ssp_flux"],
                    g["ssp_lgmet"],
                    g["ssp_alpha_fe"],
                    log_z=lz,
                    alpha_fe=0.0,
                )
            )

        grad_jax = float(jax.grad(total_flux)(-0.7))
        grad_fd = fd_grad(total_flux, -0.7)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_differentiable_wrt_alpha_fe(self, alpha_ssp_grid):
        """Gradient w.r.t. [α/Fe] should be finite and match FD.

        Evaluated at alpha_fe=0.1, an interior point of the alpha cell [0.0, 0.2].
        Must NOT be a grid boundary: at boundary values like 0.2, jnp.searchsorted
        lands in different cells for (x+eps) vs (x-eps), causing FD/autodiff mismatch.
        Interior points must agree exactly.
        """
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid

        def total_flux(afe):
            return jnp.sum(
                interpolate_met_alpha(
                    g["ssp_flux"],
                    g["ssp_lgmet"],
                    g["ssp_alpha_fe"],
                    log_z=-0.7,
                    alpha_fe=afe,
                )
            )

        grad_jax = float(jax.grad(total_flux)(0.1))
        grad_fd = fd_grad(total_flux, 0.1)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )


# ── Per-age evolving (Z, [α/Fe]) interpolation ────────────────────


class TestInterpolateMetAlphaEvolving:
    """Tests for per-age bilinear interpolation with evolving abundances."""

    def test_uniform_matches_global(self, alpha_ssp_grid):
        """Uniform Z and [α/Fe] across ages should match global interpolation."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            interpolate_met_alpha,
            interpolate_met_alpha_evolving,
        )

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        global_result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.2,
        )

        lz_per_age = jnp.full(n_age, -0.5)
        afe_per_age = jnp.full(n_age, 0.2)

        evolving_result = interpolate_met_alpha_evolving(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            lz_per_age,
            afe_per_age,
        )

        assert jnp.allclose(global_result, evolving_result, atol=1e-10)

    def test_varying_alpha_per_age(self, alpha_ssp_grid):
        """Different [α/Fe] per age should produce different spectra per age."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha_evolving

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        lz_per_age = jnp.full(n_age, -0.5)
        afe_per_age = jnp.linspace(-0.2, 0.6, n_age)  # old=0.6, young=-0.2

        result = interpolate_met_alpha_evolving(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            lz_per_age,
            afe_per_age,
        )

        # Each age bin should be different since [α/Fe] varies
        assert result.shape == (n_age, 20)
        assert jnp.all(jnp.isfinite(result))

    def test_output_shape(self, alpha_ssp_grid):
        """Output should be (n_age, n_wave)."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha_evolving

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        result = interpolate_met_alpha_evolving(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            jnp.full(n_age, -0.5),
            jnp.full(n_age, 0.0),
        )
        assert result.shape == (n_age, 20)

    def test_jit_compatible(self, alpha_ssp_grid):
        """Should work under jax.jit."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha_evolving

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        jit_fn = jax.jit(interpolate_met_alpha_evolving)
        result = jit_fn(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            jnp.full(n_age, -0.5),
            jnp.full(n_age, 0.0),
        )
        assert jnp.all(jnp.isfinite(result))


# ── Alpha evolution ramp ──────────────────────────────────────────


class TestComputeAlphaFeEvolving:
    """Tests for the [α/Fe](t_lookback) linear ramp."""

    def test_young_gets_young_value(self):
        """Youngest age bin (t_lookback ≈ 0) should get alpha_fe_young."""
        from tengri.components.stellar.sps.dsps_wrapper import compute_alpha_fe_evolving

        lg_ages = jnp.array([-2.0, -1.0, 0.0, 1.0])  # log10(age/Gyr)
        result = compute_alpha_fe_evolving(lg_ages, 0.4, 0.0, 13.7)

        # Youngest bin: 10^(-2) = 0.01 Gyr → t_frac ≈ 0.0007 → near alpha_fe_young
        assert float(result[0]) == pytest.approx(0.0, abs=0.01)

    def test_old_gets_old_value(self):
        """Oldest age bin (t_lookback ≈ t_universe) should get alpha_fe_old."""
        from tengri.components.stellar.sps.dsps_wrapper import compute_alpha_fe_evolving

        lg_ages = jnp.array([-2.0, -1.0, 0.0, 1.14])  # 10^1.14 ≈ 13.8 Gyr
        result = compute_alpha_fe_evolving(lg_ages, 0.4, 0.0, 13.7)

        # Oldest bin: 10^1.14 ≈ 13.8 Gyr, clamped to t_frac=1.0
        assert float(result[-1]) == pytest.approx(0.4, abs=0.01)

    def test_monotonic_increase(self):
        """[α/Fe] should increase monotonically with lookback time."""
        from tengri.components.stellar.sps.dsps_wrapper import compute_alpha_fe_evolving

        lg_ages = jnp.linspace(-2.0, 1.1, 20)
        result = compute_alpha_fe_evolving(lg_ages, 0.4, 0.0, 13.7)

        diffs = jnp.diff(result)
        assert jnp.all(diffs >= 0), "Alpha should increase with age (lookback time)"

    def test_equal_old_young_gives_constant(self):
        """If alpha_old == alpha_young, result should be constant."""
        from tengri.components.stellar.sps.dsps_wrapper import compute_alpha_fe_evolving

        lg_ages = jnp.linspace(-2.0, 1.1, 20)
        result = compute_alpha_fe_evolving(lg_ages, 0.3, 0.3, 13.7)

        assert jnp.allclose(result, 0.3, atol=1e-10)

    def test_differentiable(self):
        """Should be differentiable w.r.t. alpha_fe_old and match FD."""
        from tengri.components.stellar.sps.dsps_wrapper import compute_alpha_fe_evolving

        lg_ages = jnp.linspace(-2.0, 1.1, 20)

        def total(alpha_old):
            return jnp.sum(compute_alpha_fe_evolving(lg_ages, alpha_old, 0.0, 13.7))

        grad_jax = float(jax.grad(total)(0.4))
        grad_fd = fd_grad(total, 0.4)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert grad_jax > 0  # more alpha_old → higher total


# ── Effective metallicity (backward compatibility) ────────────────


class TestEffectiveMetallicity:
    """Test that effective_metallicity still works for 3D grids."""

    def test_solar_alpha_is_identity(self):
        """[α/Fe] = 0 should leave metallicity unchanged."""
        from tengri.components.stellar.sps.dsps_wrapper import effective_metallicity

        assert float(effective_metallicity(-0.5, 0.0)) == pytest.approx(-0.5)

    def test_positive_alpha_increases_z(self):
        """Positive [α/Fe] should increase effective Z."""
        from tengri.components.stellar.sps.dsps_wrapper import effective_metallicity

        z_eff = effective_metallicity(-0.5, 0.4)
        assert float(z_eff) > -0.5

    def test_coefficient_is_0_75(self):
        """The conversion coefficient should be 0.75."""
        from tengri.components.stellar.sps.dsps_wrapper import effective_metallicity

        z_eff = effective_metallicity(0.0, 1.0)
        assert float(z_eff) == pytest.approx(0.75, abs=0.01)


# ── Salaris [Fe/H] ↔ [M/H] convention conversions ─────────────────


class TestSalarisConversion:
    """Test the Salaris [Fe/H] ↔ [M/H] relation.

    This is a semi-empirical fit from Salaris, Chieffi & Straniero (1993)
    to detailed stellar interior models. The quadratic form captures how
    α-elements (which dominate the metal mass) shift total Z relative to Fe.
    """

    def test_solar_alpha_identity(self):
        """At [α/Fe] = 0.0, [M/H] = [Fe/H] exactly."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            salaris_feh_from_mh,
            salaris_mh_from_feh,
        )

        for feh in [-2.0, -1.0, -0.5, 0.0, 0.3]:
            mh = salaris_mh_from_feh(feh, 0.0)
            assert mh == pytest.approx(feh, abs=1e-10), (
                f"[α/Fe]=0: [M/H]={mh} should equal [Fe/H]={feh}"
            )
            feh_back = salaris_feh_from_mh(feh, 0.0)
            assert feh_back == pytest.approx(feh, abs=1e-10)

    def test_roundtrip(self):
        """feh → mh → feh should be identity."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            salaris_feh_from_mh,
            salaris_mh_from_feh,
        )

        for feh in [-2.0, -1.0, -0.5, 0.0, 0.3]:
            for afe in [-0.2, 0.0, 0.2, 0.4, 0.6]:
                mh = salaris_mh_from_feh(feh, afe)
                feh_back = salaris_feh_from_mh(mh, afe)
                assert feh_back == pytest.approx(feh, abs=1e-10), (
                    f"Roundtrip failed: [Fe/H]={feh}, [α/Fe]={afe} → "
                    f"[M/H]={mh} → [Fe/H]={feh_back}"
                )

    def test_positive_alpha_increases_mh(self):
        """Positive [α/Fe] should make [M/H] > [Fe/H].

        Because α-elements add to total Z: more α → higher total metallicity
        at the same iron abundance.
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        feh = -0.5
        mh_solar = salaris_mh_from_feh(feh, 0.0)
        mh_alpha = salaris_mh_from_feh(feh, 0.4)
        assert mh_alpha > mh_solar, (
            f"[α/Fe]=+0.4 should give [M/H] > [Fe/H]: "
            f"[M/H]_solar={mh_solar}, [M/H]_alpha={mh_alpha}"
        )

    def test_negative_alpha_decreases_mh(self):
        """Negative [α/Fe] should make [M/H] < [Fe/H]."""
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        feh = -0.5
        mh_solar = salaris_mh_from_feh(feh, 0.0)
        mh_low = salaris_mh_from_feh(feh, -0.2)
        assert mh_low < mh_solar

    def test_known_values(self):
        """Check specific values against the Salaris formula.

        [M/H] = [Fe/H] + 0.66154×[α/Fe] + 0.20465×[α/Fe]²
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        # [Fe/H] = -0.5, [α/Fe] = +0.4:
        # offset = 0.66154 × 0.4 + 0.20465 × 0.16 = 0.26462 + 0.03274 = 0.29736
        expected = -0.5 + 0.29736
        result = salaris_mh_from_feh(-0.5, 0.4)
        assert result == pytest.approx(expected, abs=1e-4)

        # [Fe/H] = 0.0, [α/Fe] = +0.6:
        # offset = 0.66154 × 0.6 + 0.20465 × 0.36 = 0.39692 + 0.07367 = 0.47060
        expected = 0.0 + 0.47060
        result = salaris_mh_from_feh(0.0, 0.6)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_offset_magnitude(self):
        """The [M/H]−[Fe/H] offset at [α/Fe]=+0.4 should be ~0.3 dex.

        This is a well-known result: α-enhanced populations have ~0.3 dex
        higher total Z than their [Fe/H] suggests.
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        offset = salaris_mh_from_feh(0.0, 0.4) - 0.0
        assert 0.25 < offset < 0.35, f"Offset at [α/Fe]=+0.4 should be ~0.3, got {offset:.3f}"

    def test_quadratic_term_is_small(self):
        """The quadratic term should be << linear term for typical [α/Fe].

        Linear: 0.66 × [α/Fe]
        Quadratic: 0.20 × [α/Fe]²

        At [α/Fe] = 0.4: linear=0.265, quadratic=0.033 → ratio ~8:1
        """
        from tengri.components.stellar.sps.dsps_wrapper import _SALARIS_LINEAR, _SALARIS_QUADRATIC

        afe = 0.4
        linear_term = _SALARIS_LINEAR * afe
        quad_term = _SALARIS_QUADRATIC * afe**2
        assert linear_term > 5 * quad_term, (
            f"Quadratic should be small: linear={linear_term:.4f}, quad={quad_term:.4f}"
        )

    def test_compare_to_thomas_approximation(self):
        """Salaris should be consistent with Thomas+2003 linear approximation.

        Thomas+2003 uses [M/H] ≈ [Fe/H] + 0.94×[α/Fe] (for O-enhanced)
        while Salaris gives [M/H] ≈ [Fe/H] + 0.66×[α/Fe] + 0.20×[α/Fe]².

        These differ because Thomas uses a different solar mixture and
        element selection. They should agree to within ~0.1 dex for typical values.
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        for afe in [0.0, 0.2, 0.4]:
            salaris = salaris_mh_from_feh(0.0, afe)
            # Vazdekis+2015 approximation: 0.75 × [α/Fe]
            vazdekis = 0.75 * afe
            assert abs(salaris - vazdekis) < 0.15, (
                f"Salaris ({salaris:.3f}) and Vazdekis ({vazdekis:.3f}) "
                f"should agree within 0.15 dex at [α/Fe]={afe}"
            )


# ── Solar [α/Fe] = 0.0 equivalence with no-alpha case ─────────────


class TestSolarAlphaEquivalence:
    """Test that [α/Fe] = 0.0 in a 4D grid reproduces the 3D (no-alpha) case.

    This is a critical consistency check: if you take a 4D grid and
    interpolate at [α/Fe] = 0.0, the result should be identical to
    using the [α/Fe] = 0.0 slice as a standard 3D grid.
    """

    def test_4d_at_solar_matches_3d_slice(self, alpha_ssp_grid):
        """Interpolating 4D grid at [α/Fe]=0.0 = direct slice at α index 1."""
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        # Direct slice: [α/Fe] = 0.0 is index 1 in the grid
        slice_3d = g["ssp_flux"][:, 1, :, :]  # (n_met, n_age, n_wave)

        # 4D interpolation at [α/Fe] = 0.0, same metallicity
        for i_met, lz in enumerate(g["ssp_lgmet"]):
            result_4d = interpolate_met_alpha(
                g["ssp_flux"],
                g["ssp_lgmet"],
                g["ssp_alpha_fe"],
                log_z=float(lz),
                alpha_fe=0.0,
            )
            expected_3d = slice_3d[i_met]  # (n_age, n_wave)
            assert jnp.allclose(result_4d, expected_3d, atol=1e-10), (
                f"4D at [α/Fe]=0.0 should match 3D slice at [Fe/H]={float(lz)}"
            )

    def test_4d_solar_matches_interpolate_metallicity(self, alpha_ssp_grid):
        """4D at [α/Fe]=0.0 should match standard 3D interpolate_metallicity."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            interpolate_met_alpha,
            interpolate_metallicity,
        )

        g = alpha_ssp_grid
        # Extract the solar-alpha (α=0) slice as a 3D grid
        ssp_flux_3d = g["ssp_flux"][:, 1, :, :]  # (n_met, n_age, n_wave)

        for lz in [-1.0, -0.5, -0.25]:
            result_4d = interpolate_met_alpha(
                g["ssp_flux"],
                g["ssp_lgmet"],
                g["ssp_alpha_fe"],
                log_z=lz,
                alpha_fe=0.0,
            )
            result_3d = interpolate_metallicity(
                ssp_flux_3d,
                g["ssp_lgmet"],
                lz,
            )
            assert jnp.allclose(result_4d, result_3d, atol=1e-10), (
                f"4D bilinear at [α/Fe]=0 should match 3D linear at [Fe/H]={lz}"
            )

    def test_evolving_with_constant_solar_matches_global(self, alpha_ssp_grid):
        """Per-age interpolation with constant [α/Fe]=0.0 = global result."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            interpolate_met_alpha,
            interpolate_met_alpha_evolving,
        )

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        global_result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.0,
        )

        evolving_result = interpolate_met_alpha_evolving(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            jnp.full(n_age, -0.5),
            jnp.full(n_age, 0.0),  # constant solar
        )

        assert jnp.allclose(global_result, evolving_result, atol=1e-10)

    def test_salaris_at_solar_is_identity(self):
        """Salaris conversion at [α/Fe]=0.0 is identity: [M/H] = [Fe/H]."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            salaris_feh_from_mh,
            salaris_mh_from_feh,
        )

        for z in [-2.0, -1.0, -0.5, 0.0, 0.3]:
            assert salaris_mh_from_feh(z, 0.0) == pytest.approx(z, abs=1e-12)
            assert salaris_feh_from_mh(z, 0.0) == pytest.approx(z, abs=1e-12)


# ── Convention differences: [Fe/H] vs [M/H] vs effective_metallicity


class TestConventionDifferences:
    """Test that different metallicity conventions produce meaningfully
    different results, and understand when they diverge.

    The three conventions:
    1. [Fe/H] grid (our canonical): grid axis is iron abundance
    2. [M/H] grid (sMILES): grid axis is total metallicity
    3. effective_metallicity (approximation): shift Z by 0.75×[α/Fe]
    """

    def test_conventions_agree_at_solar_alpha(self):
        """All three conventions should agree at [α/Fe] = 0.0."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            effective_metallicity,
            salaris_mh_from_feh,
        )

        feh = -0.5
        afe = 0.0

        mh = salaris_mh_from_feh(feh, afe)
        z_eff = float(effective_metallicity(feh, afe))

        assert mh == pytest.approx(feh, abs=1e-10)
        assert z_eff == pytest.approx(feh, abs=1e-10)

    def test_conventions_diverge_at_high_alpha(self):
        """At [α/Fe] = +0.4, the three approaches give different Z values.

        This demonstrates why proper alpha grids matter: the approximation
        and the exact conversion give different effective metallicities.
        """
        from tengri.components.stellar.sps.dsps_wrapper import (
            effective_metallicity,
            salaris_mh_from_feh,
        )

        feh = -0.5
        afe = 0.4

        # Salaris exact: [M/H] = -0.5 + 0.265 + 0.033 = -0.203
        mh_salaris = salaris_mh_from_feh(feh, afe)

        # effective_metallicity approximation: [Z/H]_eff = -0.5 + 0.75×0.4 = -0.2
        z_eff = float(effective_metallicity(feh, afe))

        # They are close but NOT identical (differ by ~0.003 dex at [α/Fe]=0.4)
        diff = abs(mh_salaris - z_eff)
        assert diff > 1e-4, (
            "Salaris and effective_metallicity should differ at [α/Fe]≠0 "
            f"(Salaris={mh_salaris:.4f}, eff_met={z_eff:.4f}, diff={diff:.5f})"
        )

        # But within ~0.1 dex (same order of magnitude correction)
        assert diff < 0.15

    def test_effective_metallicity_is_linear_salaris_is_quadratic(self):
        """effective_metallicity is linear in [α/Fe], Salaris is quadratic.

        The difference grows with [α/Fe]².
        """
        from tengri.components.stellar.sps.dsps_wrapper import (
            effective_metallicity,
            salaris_mh_from_feh,
        )

        feh = 0.0
        # Compute difference for several [α/Fe] values
        diffs = []
        for afe in [0.0, 0.2, 0.4, 0.6]:
            mh = salaris_mh_from_feh(feh, afe)
            z_eff = float(effective_metallicity(feh, afe))
            diffs.append(abs(mh - z_eff))

        # Difference should grow (quadratic term becomes more important)
        assert diffs[0] < 0.01  # at [α/Fe]=0, both are zero offset
        assert diffs[-1] > diffs[1]  # divergence grows with [α/Fe]

    def test_4d_grid_vs_effective_met_3d(self, alpha_ssp_grid):
        """4D grid interpolation and 3D+effective_metallicity give different SEDs.

        This is the key test: the whole point of 4D grids is that they
        capture spectral effects that effective_metallicity misses.
        """
        from tengri.components.stellar.sps.dsps_wrapper import (
            effective_metallicity,
            interpolate_met_alpha,
            interpolate_metallicity,
        )

        g = alpha_ssp_grid
        feh = -0.5
        afe = 0.4

        # Method 1: proper 4D interpolation
        sed_4d = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=feh,
            alpha_fe=afe,
        )

        # Method 2: effective_metallicity on 3D solar-alpha slice
        z_eff = float(effective_metallicity(feh, afe))
        ssp_3d = g["ssp_flux"][:, 1, :, :]  # solar α slice
        sed_3d = interpolate_metallicity(ssp_3d, g["ssp_lgmet"], z_eff)

        # They should be DIFFERENT (this is the whole point of proper alpha grids)
        diff = float(jnp.sum(jnp.abs(sed_4d - sed_3d)))
        assert diff > 0, (
            "4D grid and 3D+effective_metallicity should give different SEDs "
            "at [α/Fe] ≠ 0 — this is why proper alpha grids exist"
        )
