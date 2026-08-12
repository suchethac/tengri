# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for alpha-enhanced SSP grids.

References:
- Salaris, Chieffi & Straniero 1993, ApJ 414, 580 (Salaris relation)
- Vazdekis et al. 2015, MNRAS 449, 1177 (α-enhancement effects)
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def alpha_ssp_grid():
    """Synthetic 4D SSP grid for regression tests."""
    n_met, n_alpha, n_age, n_wave = 3, 5, 10, 20

    lgmet = jnp.array([-1.5, -0.5, 0.0])
    alpha_fe = jnp.array([-0.2, 0.0, 0.2, 0.4, 0.6])
    lg_age_gyr = jnp.linspace(-1.0, 1.1, n_age)
    wave = jnp.linspace(3500.0, 9000.0, n_wave)

    key = jax.random.PRNGKey(42)
    base = jnp.abs(jax.random.normal(key, (n_met, n_alpha, n_age, n_wave))) * 1e-3 + 1e-5

    z_trend = jnp.linspace(0.5, 1.5, n_wave)[None, None, None, :]
    met_scale = jnp.linspace(0.5, 1.5, n_met)[:, None, None, None]
    alpha_bump = jnp.exp(-0.5 * ((jnp.arange(n_wave) - 10) / 2.0) ** 2)
    alpha_scale = alpha_fe[:, None, None] * alpha_bump[None, None, :]
    alpha_scale = alpha_scale[None, :, :, :]
    age_scale = jnp.linspace(1.5, 0.5, n_age)[None, None, :, None]

    flux = base * met_scale * z_trend * age_scale + 0.01 * alpha_scale

    return {
        "ssp_flux": flux,
        "ssp_lgmet": lgmet,
        "ssp_alpha_fe": alpha_fe,
        "ssp_lg_age_gyr": lg_age_gyr,
        "ssp_wave": wave,
    }


# ── Salaris [Fe/H] ↔ [M/H] convention (Salaris et al. 1993) ────────


class TestSalarisConversion:
    """Regression test for Salaris, Chieffi & Straniero 1993, ApJ 414, 580.

    The Salaris relation converts between iron abundance ([Fe/H]) and total
    metallicity ([M/H]) using a semi-empirical fit from stellar models:
        [M/H] = [Fe/H] + a × [α/Fe] + b × [α/Fe]²
    where a = 0.66154 and b = 0.20465 (Table 2, Salaris et al. 1993).
    """

    def test_solar_alpha_identity(self):
        """At [α/Fe] = 0.0, [M/H] = [Fe/H] exactly.

        Salaris Table 2 identity case.
        """
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
        """feh → mh → feh should be identity within machine precision.

        Salaris Table 2, roundtrip test.
        """
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

        Salaris Table 2: because α-elements add to total Z, more α → higher
        total metallicity at the same iron abundance. This is a fundamental
        result from stellar models.
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
        """Negative [α/Fe] should make [M/H] < [Fe/H].

        Salaris Table 2: lower α-abundance → lower total metallicity.
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        feh = -0.5
        mh_solar = salaris_mh_from_feh(feh, 0.0)
        mh_low = salaris_mh_from_feh(feh, -0.2)
        assert mh_low < mh_solar

    def test_known_values(self):
        """Check specific values against the Salaris formula.

        Salaris et al. 1993, Table 2:
        [M/H] = [Fe/H] + 0.66154×[α/Fe] + 0.20465×[α/Fe]²

        Tolerance: 1e-4 dex (precision of published formula).
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

        Salaris et al. 1993: α-enhanced populations have ~0.3 dex
        higher total Z than their [Fe/H] suggests. Well-known result
        in the literature.
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        offset = salaris_mh_from_feh(0.0, 0.4) - 0.0
        assert 0.25 < offset < 0.35, f"Offset at [α/Fe]=+0.4 should be ~0.3, got {offset:.3f}"

    def test_quadratic_term_is_small(self):
        """The quadratic term should be << linear term for typical [α/Fe].

        Salaris et al. 1993 Table 2: linear term dominates near the solar
        abundance range.

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


# ── Solar [α/Fe] = 0.0 equivalence (Backward compatibility) ─────────


class TestSolarAlphaEquivalence:
    """Regression test: 4D grid at [α/Fe]=0.0 reproduces 3D (no-alpha) case.

    This is a critical consistency check: if you take a 4D grid and
    interpolate at [α/Fe] = 0.0, the result should be identical to
    using the [α/Fe] = 0.0 slice as a standard 3D grid. This ensures
    backward compatibility.
    """

    def test_4d_at_solar_matches_3d_slice(self, alpha_ssp_grid):
        """Interpolating 4D grid at [α/Fe]=0.0 = direct slice at α index 1.

        Backward compatibility: 4D grids reduce to 3D behavior at [α/Fe]=0.
        """
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

    def test_salaris_at_solar_is_identity(self):
        """Salaris conversion at [α/Fe]=0.0 is identity: [M/H] = [Fe/H].

        Backward compatibility: Salaris formula must give [M/H]=[Fe/H] when
        α-enhancement is zero.
        """
        from tengri.components.stellar.sps.dsps_wrapper import (
            salaris_feh_from_mh,
            salaris_mh_from_feh,
        )

        for z in [-2.0, -1.0, -0.5, 0.0, 0.3]:
            assert salaris_mh_from_feh(z, 0.0) == pytest.approx(z, abs=1e-12)
            assert salaris_feh_from_mh(z, 0.0) == pytest.approx(z, abs=1e-12)

    def test_compare_to_thomas_approximation(self):
        """Salaris should be consistent with Thomas+2003 / Vazdekis+2015 approximations.

        Regression: Salaris et al. 1993 vs literature approximations.
        Thomas+2003 uses [M/H] ≈ [Fe/H] + 0.94×[α/Fe] (for O-enhanced)
        while Salaris gives [M/H] ≈ [Fe/H] + 0.66×[α/Fe] + 0.20×[α/Fe]².
        Vazdekis+2015 approximation: 0.75 × [α/Fe].
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


# ── Convention differences: [Fe/H] vs [M/H] vs effective_metallicity ───


class TestConventionDifferences:
    """Regression test comparing three metallicity conventions from the literature.

    References:
    - Salaris et al. 1993: [M/H] grid convention
    - Vazdekis et al. 2015: effective_metallicity approximation
    - Our implementation: proper 4D grid support

    The three conventions:
    1. [Fe/H] grid (our canonical): grid axis is iron abundance
    2. [M/H] grid (sMILES): grid axis is total metallicity (Salaris corrected)
    3. effective_metallicity (approximation): shift Z by 0.75×[α/Fe]
    """

    def test_conventions_agree_at_solar_alpha(self):
        """All three conventions should agree at [α/Fe] = 0.0.

        At solar α-enhancement, all three approaches give identical results.
        """
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

    def test_effective_metallicity_is_linear_salaris_is_quadratic(self):
        """effective_metallicity is linear in [α/Fe], Salaris is quadratic.

        Vazdekis et al. 2015 vs Salaris et al. 1993: difference grows with [α/Fe]².
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

    def test_conventions_diverge_at_high_alpha(self):
        """At [α/Fe] = +0.4, the three conventions give different Z values.

        Regression: why proper alpha grids matter (Salaris vs effective_metallicity).
        The approximation and the exact conversion give different effective metallicities.
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

    def test_4d_grid_vs_effective_met_3d(self, alpha_ssp_grid):
        """4D grid interpolation and 3D+effective_metallicity give different SEDs.

        Regression: the key test why proper 4D alpha grids are needed.
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

    def test_4d_at_solar_matches_interpolate_metallicity(self, alpha_ssp_grid):
        """4D at [α/Fe]=0.0 should match standard 3D interpolate_metallicity.

        Regression: backward compatibility at solar [α/Fe].
        """
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
        """Per-age interpolation with constant [α/Fe]=0.0 = global result.

        Regression: consistency between global and per-age paths.
        """
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
