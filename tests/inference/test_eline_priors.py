# SPDX-License-Identifier: BSD-3-Clause
"""Tests for CLOUDY-based emission line priors.

Tests cover:
- Correct number of reference lines returned.
- Halpha/Hbeta ratio ~ 2.86 (Case B).
- [OIII] 5007 > [OIII] 4959 by factor ~3.
- Prior width is configurable.
- Metallicity and ionization parameter interpolation.
- Integration with marginalize_emission_lines via CLOUDY wrapper.
- JIT compatibility.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.observation.eline_priors import (
    cloudy_line_priors,
    marginalize_emission_lines_cloudy,
)

# ── cloudy_line_priors tests ──────────────────────────────────────


class TestCloudyLinePriors:
    """Tests for CLOUDY line ratio priors."""

    def test_returns_correct_number_of_lines(self):
        """Should return priors for all 12 reference lines by default."""
        means, sigmas = cloudy_line_priors()
        chex.assert_shape(means, (12,))
        chex.assert_shape(sigmas, (12,))

    def test_halpha_hbeta_ratio(self):
        """Halpha/Hbeta should be ~2.86 (Case B recombination)."""
        means, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)

        # OII 3726 (0), OII 3729 (1), Hdelta (2), Hgamma (3), Hbeta (4),
        # OIII4959 (5), OIII5007 (6), NII6548 (7), Halpha (8), NII6583 (9),
        # SII6716 (10), SII6731 (11)
        halpha_idx = 8
        hbeta_idx = 4
        ratio = means[halpha_idx] / means[hbeta_idx]
        assert jnp.allclose(ratio, 2.86, atol=0.01), f"Ha/Hb = {ratio:.3f}, expected 2.86"

    def test_oiii_5007_gt_4959(self):
        """[OIII] 5007 should be ~3x [OIII] 4959."""
        means, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)

        # [OIII] 4959 is index 5, [OIII] 5007 is index 6
        oiii_4959 = means[5]
        oiii_5007 = means[6]
        ratio = oiii_5007 / oiii_4959
        assert ratio > 2.5, f"[OIII] 5007/4959 = {ratio:.2f}, expected ~3"
        assert ratio < 3.5, f"[OIII] 5007/4959 = {ratio:.2f}, expected ~3"

    def test_hbeta_is_reference(self):
        """Hbeta (index 4) should have ratio = 1.0."""
        means, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)
        assert jnp.allclose(means[4], 1.0, atol=0.01)

    def test_all_positive(self):
        """All line ratios should be positive."""
        means, sigmas = cloudy_line_priors()
        assert jnp.all(means > 0)
        assert jnp.all(sigmas > 0)

    def test_prior_width_configurable(self):
        """Different prior_width_dex should change sigmas."""
        _, sigmas_narrow = cloudy_line_priors(prior_width_dex=0.1)
        _, sigmas_wide = cloudy_line_priors(prior_width_dex=0.5)

        # Wider dex scatter -> larger linear sigmas
        assert jnp.all(sigmas_wide > sigmas_narrow)

    def test_low_metallicity_weaker_nii(self):
        """At low metallicity, [NII] lines should be weaker."""
        means_solar, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)
        means_lowz, _ = cloudy_line_priors(log_z=-0.7, neb_logU=-3.0)

        # [NII] 6583 is index 9 (after split OII doublet at 0,1)
        assert means_lowz[9] < means_solar[9], "Low-Z [NII] should be weaker"

    def test_high_ionization_stronger_oiii(self):
        """At higher logU, [OIII] lines should be stronger."""
        means_lowu, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)
        means_highu, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.0)

        # [OIII] 5007 is index 6 (after split OII doublet at 0,1)
        assert means_highu[6] > means_lowu[6], "Higher logU -> stronger [OIII]"

    def test_metallicity_matters_at_high_logU(self):
        """Metallicity interpolation must remain active at logU=-2 (bilinear fix)."""
        means_solar, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.0)
        means_lowz, _ = cloudy_line_priors(log_z=-0.7, neb_logU=-2.0)

        # [NII] 6583 (index 9) should still be weaker at low metallicity even at logU=-2.
        # Before the bilinear fix, u_frac=1 collapsed to solar Z, making means_lowz == means_solar.
        assert means_lowz[9] < means_solar[9], (
            "Low-Z [NII] at logU=-2 must be weaker than solar-Z [NII]; "
            "may indicate missing _CLOUDY_SUBSOLAR_LOGU2 grid corner"
        )

        # [OIII] 5007 (index 6) should be stronger at low metallicity (less cooling → hotter HII)
        assert means_lowz[6] > means_solar[6], (
            "[OIII] at logU=-2 must be stronger at low Z; "
            "bilinear interpolation must span all 4 grid corners"
        )

    def test_specific_wavelengths(self):
        """Requesting specific wavelengths should return matched priors."""
        # Request only Halpha and Hbeta (vacuum wavelengths)
        lines = jnp.array([6564.61, 4862.68])
        means, sigmas = cloudy_line_priors(line_wavelengths=lines)
        chex.assert_shape(means, (2,))
        chex.assert_shape(sigmas, (2,))
        # First should be Halpha (~2.86), second should be Hbeta (1.0)
        assert means[0] > 2.0  # Halpha
        assert jnp.allclose(means[1], 1.0, atol=0.1)  # Hbeta

    def test_jit_compatible(self):
        """cloudy_line_priors should be JIT-compilable."""

        @jax.jit
        def _eval():
            return cloudy_line_priors(log_z=0.0, neb_logU=-3.0)

        means, _sigmas = _eval()
        chex.assert_shape(means, (12,))
        chex.assert_tree_all_finite(means)

    def test_gradient_matches_finite_difference_log_z(self):
        """AD gradient wrt log_z matches finite-difference at interior grid point.

        Regression for Phase 6C: verifies that the bilinear interpolation over
        all 4 (Z, logU) corners is correctly differentiable. Tests at an interior
        point (-0.4, -2.5) away from clip boundaries so all 4 corners contribute.
        """
        eps = 1e-4
        log_z0 = -0.4  # interior point, well away from subsolar (-0.7) and solar (0.0) edges

        def sum_means(log_z: float) -> float:
            means, _ = cloudy_line_priors(log_z=log_z, neb_logU=-2.5)
            return jnp.sum(means)

        g_analytic = float(jax.grad(sum_means)(log_z0))
        g_fd = float((sum_means(log_z0 + eps) - sum_means(log_z0 - eps)) / (2.0 * eps))
        rel_err = abs(g_analytic - g_fd) / (abs(g_fd) + 1e-12)
        assert rel_err < 0.001, (
            f"log_z gradient mismatch: analytic={g_analytic:.6f}, FD={g_fd:.6f}, "
            f"rel_err={rel_err:.4f} — bilinear z_frac term may be missing"
        )

    def test_gradient_matches_finite_difference_neb_logU(self):
        """AD gradient wrt neb_logU matches finite-difference at interior grid point.

        Regression for Phase 6C: verifies the u_frac term in the bilinear
        interpolation is correctly differentiated.
        """
        eps = 1e-4
        logU0 = -2.5  # interior point between grid nodes -3.0 and -2.0

        def sum_means(neb_logU: float) -> float:
            means, _ = cloudy_line_priors(log_z=-0.4, neb_logU=neb_logU)
            return jnp.sum(means)

        g_analytic = float(jax.grad(sum_means)(logU0))
        g_fd = float((sum_means(logU0 + eps) - sum_means(logU0 - eps)) / (2.0 * eps))
        rel_err = abs(g_analytic - g_fd) / (abs(g_fd) + 1e-12)
        assert rel_err < 0.001, (
            f"neb_logU gradient mismatch: analytic={g_analytic:.6f}, FD={g_fd:.6f}, "
            f"rel_err={rel_err:.4f} — bilinear u_frac term may be missing"
        )


# ── marginalize_emission_lines_cloudy integration tests ───────────


class TestMarginalizeEmissionLinesCloudy:
    """Tests for the CLOUDY-prior emission line marginalization wrapper."""

    @pytest.fixture()
    def mock_spectral_data(self):
        """Create mock spectral data for emission line tests."""
        n_pix = 500
        n_lines = 3

        wave = jnp.linspace(4000.0, 7000.0, n_pix)
        noise = jnp.ones(n_pix) * 0.1
        residual = jnp.zeros(n_pix)

        # Simple Gaussian design matrix (3 lines, vacuum wavelengths)
        line_waves = jnp.array([4862.68, 5008.24, 6564.61])
        design_matrix = jnp.zeros((n_pix, n_lines))

        for j in range(n_lines):
            sigma = 2.0  # Angstrom
            col = jnp.exp(-0.5 * ((wave - line_waves[j]) / sigma) ** 2) / (
                jnp.sqrt(2 * jnp.pi) * sigma
            )
            design_matrix = design_matrix.at[:, j].set(col)

        return {
            "residual": residual,
            "noise": noise,
            "design_matrix": design_matrix,
            "line_wavelengths": line_waves,
        }

    def test_returns_three_outputs(self, mock_spectral_data):
        """Should return (ln_L_marg, a_hat, a_cov)."""
        d = mock_spectral_data
        result = marginalize_emission_lines_cloudy(
            d["residual"],
            d["noise"],
            d["design_matrix"],
            line_wavelengths=d["line_wavelengths"],
        )
        assert len(result) == 3
        _ln_l, a_hat, a_cov = result
        chex.assert_shape(a_hat, (3,))
        chex.assert_shape(a_cov, (3, 3))

    def test_finite_output(self, mock_spectral_data):
        """All outputs should be finite."""
        d = mock_spectral_data
        ln_l, a_hat, a_cov = marginalize_emission_lines_cloudy(
            d["residual"],
            d["noise"],
            d["design_matrix"],
            line_wavelengths=d["line_wavelengths"],
        )
        assert jnp.isfinite(ln_l)
        chex.assert_tree_all_finite(a_hat)
        chex.assert_tree_all_finite(a_cov)

    def test_prior_constrains_amplitudes(self, mock_spectral_data):
        """With CLOUDY priors, amplitudes should reflect line ratios."""
        d = mock_spectral_data

        # Add signal: inject Halpha >> Hbeta emission into residual
        # (Halpha at index 2, Hbeta at index 0 in our 3-line setup)
        signal = d["design_matrix"] @ jnp.array([1.0, 1.34, 2.86])
        residual_with_signal = d["residual"] + signal

        _ln_l, a_hat, _a_cov = marginalize_emission_lines_cloudy(
            residual_with_signal,
            d["noise"],
            d["design_matrix"],
            line_wavelengths=d["line_wavelengths"],
            l_hbeta=1.0,
        )

        # Halpha amplitude (index 2) should be larger than Hbeta (index 0)
        assert a_hat[2] > a_hat[0], "Halpha should be stronger than Hbeta"

    def test_l_hbeta_scaling(self, mock_spectral_data):
        """Scaling l_hbeta should scale the prior means linearly.

        To test prior-dominated behavior we need uninformative data (large
        noise), so we override the fixture noise here.  With noise >> signal,
        the posterior mean converges to the prior mean which scales linearly
        with l_hbeta.
        """
        d = mock_spectral_data
        # Use very large noise so the posterior is prior-dominated.
        uninformative_noise = jnp.ones_like(d["noise"]) * 1e6

        _, a_hat_1, _ = marginalize_emission_lines_cloudy(
            d["residual"],
            uninformative_noise,
            d["design_matrix"],
            line_wavelengths=d["line_wavelengths"],
            l_hbeta=1.0,
        )

        _, a_hat_10, _ = marginalize_emission_lines_cloudy(
            d["residual"],
            uninformative_noise,
            d["design_matrix"],
            line_wavelengths=d["line_wavelengths"],
            l_hbeta=10.0,
        )

        # In the prior-dominated regime, a_hat ≈ prior_mean ∝ l_hbeta
        ratio = a_hat_10 / jnp.maximum(jnp.abs(a_hat_1), 1e-30)
        assert jnp.all(ratio > 5.0), "l_hbeta=10 should give ~10x larger amplitudes"

    def test_lnl_varies_with_prior_mean(self, mock_spectral_data):
        """ln_L must change when the CLOUDY prior mean changes (different log_z).

        Regression for NEW-02: the shift-marginalize-unshift trick is mathematically
        exact (the marginal likelihood is shift-invariant), but this test guards against
        any future implementation that breaks that invariance — e.g. by returning the
        zero-mean-prior ln_L without applying the residual shift.
        If the shift is missing, solar and sub-solar will give identical ln_L values.
        """
        d = mock_spectral_data
        # Inject a signal so the prior mean matters
        signal = d["design_matrix"] @ jnp.array([1.0, 1.34, 2.86])
        residual_with_signal = d["residual"] + signal

        ln_l_solar, _, _ = marginalize_emission_lines_cloudy(
            residual_with_signal,
            d["noise"],
            d["design_matrix"],
            log_z=0.0,
            neb_logU=-3.0,
            line_wavelengths=d["line_wavelengths"],
        )
        ln_l_subsolar, _, _ = marginalize_emission_lines_cloudy(
            residual_with_signal,
            d["noise"],
            d["design_matrix"],
            log_z=-0.7,
            neb_logU=-3.0,
            line_wavelengths=d["line_wavelengths"],
        )
        assert abs(float(ln_l_solar - ln_l_subsolar)) > 0.01, (
            "ln_L must differ between solar and sub-solar metallicity priors; "
            "may indicate residual shift (prior mean) is not applied"
        )

    def test_gradient_is_finite(self, mock_spectral_data):
        """Gradient of marginalized ln_L wrt log_z must be finite.

        Regression for NEW-09: ensures log_z gradient is computable and finite.
        """
        d = mock_spectral_data

        def ln_l_fn(log_z):
            result = marginalize_emission_lines_cloudy(
                d["residual"],
                d["noise"],
                d["design_matrix"],
                log_z=log_z,
                neb_logU=-3.0,
                line_wavelengths=d["line_wavelengths"],
            )
            return result[0]

        grad_jax = float(jax.grad(ln_l_fn)(0.0))
        grad_fd = fd_grad(ln_l_fn, 0.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )

    def test_gradient_matches_finite_difference(self, mock_spectral_data):
        """Analytic gradient must match finite-difference to 0.1% precision.

        Regression for NEW-09: verifies that marginalize_emission_lines_cloudy
        correctly handles the Gaussian prior normalization when the prior mean
        is non-zero (i.e., when log_z != 0).
        """
        d = mock_spectral_data

        def ln_l_fn(log_z):
            result = marginalize_emission_lines_cloudy(
                d["residual"],
                d["noise"],
                d["design_matrix"],
                log_z=log_z,
                neb_logU=-3.0,
                line_wavelengths=d["line_wavelengths"],
            )
            return result[0]

        eps = 1e-4
        g_analytic = float(jax.grad(ln_l_fn)(0.0))
        g_fd = float((ln_l_fn(eps) - ln_l_fn(-eps)) / (2 * eps))

        rel_err = abs(g_analytic - g_fd) / (abs(g_fd) + 1e-10)
        assert rel_err < 0.001, (
            f"Gradient mismatch: analytic={g_analytic:.6f}, FD={g_fd:.6f}, "
            f"relative error={rel_err:.4f}"
        )


# ── balmer_decrement_prior tests ──────────────────────────────────


class TestBalmerDecrementPrior:
    """Tests for Calzetti-based Balmer decrement predictions."""

    def test_returns_four_lines(self):
        """Should return wavelengths and ratios for 4 Balmer lines."""
        from tengri.observation.eline_priors import balmer_decrement_prior

        wavs, ratios = balmer_decrement_prior(dust_tau_diff=0.0)
        chex.assert_shape(wavs, (4,))
        chex.assert_shape(ratios, (4,))

    def test_zero_dust_gives_intrinsic_ratios(self):
        """With no dust, ratios should equal intrinsic Case B values."""
        from tengri.observation.eline_priors import balmer_decrement_prior

        _wavs, ratios = balmer_decrement_prior(dust_tau_diff=0.0)
        intrinsic = jnp.array([2.86, 1.0, 0.468, 0.259])
        assert jnp.allclose(ratios, intrinsic, atol=1e-5)

    def test_dust_increases_halpha_hbeta(self):
        """More dust → larger observed Hα/Hβ (Hα is less attenuated than Hβ)."""
        from tengri.observation.eline_priors import balmer_decrement_prior

        _wavs, ratios_lo = balmer_decrement_prior(dust_tau_diff=0.3)
        _wavs, ratios_hi = balmer_decrement_prior(dust_tau_diff=1.0)
        # Hα/Hβ: index 0 / index 1 — Hα is in red branch (less attenuated)
        ratio_lo = ratios_lo[0] / ratios_lo[1]
        ratio_hi = ratios_hi[0] / ratios_hi[1]
        assert ratio_hi > ratio_lo > 2.86

    def test_halpha_uses_red_calzetti_branch(self):
        """Hα (6564.61 Å) must use the red Calzetti piecewise (λ > 6300 Å).

        The red branch k(λ) = 2.659(-1.857 + 1.040/λ_um) + R_V gives a smaller
        k than the blue branch at the same wavelength, so Hα is less reddened.
        We verify by checking k(Hα) < k(Hβ) at the same E(B-V).
        """
        from tengri.observation.eline_priors import balmer_decrement_prior

        # With dust, Hβ (blue) should be more attenuated than Hα (red branch).
        _wavs, ratios = balmer_decrement_prior(dust_tau_diff=1.0)
        # Hα/Hβ > 2.86 means Hα is relatively brighter after dust → less attenuated
        assert ratios[0] / ratios[1] > 2.86

    def test_wavelengths_are_vacuum(self):
        """Returned wavelengths should be vacuum, not air."""
        from tengri.observation.eline_priors import balmer_decrement_prior

        wavs, _ = balmer_decrement_prior(dust_tau_diff=0.0)
        # Hα vacuum = 6564.61, air = 6562.80 — must NOT return the air value
        halpha = float(wavs[0])
        assert halpha > 6563.5, f"Hα should be vacuum (6564.61), got {halpha}"
        # Hβ vacuum = 4862.68, air = 4861.33
        hbeta = float(wavs[1])
        assert hbeta > 4861.8, f"Hβ should be vacuum (4862.68), got {hbeta}"


# ── cloudy_grid_line_priors tests (NEW-07) ────────────────────────


class _MockGridData:
    """Minimal mock of CloudyGridData for testing cloudy_grid_line_priors."""

    def __init__(self):
        import numpy as np

        # Small 3x2x3 grid: n_met=3, n_age=2, n_logU=3, n_lines=5
        self.line_log_met = np.array([-2.0, -1.0, 0.0])  # sub-solar to solar
        self.line_log_age = np.array([6.0, 8.0])  # 1 Myr, 100 Myr
        self.line_log_U = np.array([-3.5, -3.0, -2.0])

        # Vacuum wavelengths for 5 reference lines
        self.line_wavelengths = np.array([4862.68, 4960.30, 5008.24, 6564.61, 6585.28])
        # Hβ, [OIII]4959, [OIII]5007, Hα, [NII]6583

        n_met, n_age, n_logU, n_lines = 3, 2, 3, 5
        # log10 luminosities: Hβ=1.0 (log=0), others relative
        # Hβ index 0: always log10=0 → lin=1.0
        # [OIII]5007 index 2: stronger at higher U and lower Z
        lum = np.zeros((n_met, n_age, n_logU, n_lines))
        for iz in range(n_met):
            for ia in range(n_age):
                for iu in range(n_logU):
                    lum[iz, ia, iu, 0] = 0.0  # Hβ: 1.0 (reference)
                    lum[iz, ia, iu, 1] = -0.35  # [OIII]4959 ~ 0.45
                    # [OIII]5007 increases with logU and decreases with Z
                    lum[iz, ia, iu, 2] = 0.1 + 0.2 * iu - 0.1 * iz
                    lum[iz, ia, iu, 3] = 0.456  # Hα: 2.86 (Case B)
                    # [NII]6583 decreases with lower Z
                    lum[iz, ia, iu, 4] = -1.0 + 0.5 * iz

        self.line_luminosity = lum


class TestCloudyGridLinePriors:
    """Tests for cloudy_grid_line_priors() trilinear interpolation."""

    def test_returns_correct_shape(self):
        """Should return prior means and sigmas for all grid lines by default."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        means, sigmas = cloudy_grid_line_priors(grid, log_z=-1.0, neb_logU=-3.0)
        assert means.shape == (5,), f"Expected (5,), got {means.shape}"
        assert sigmas.shape == (5,), f"Expected (5,), got {sigmas.shape}"

    def test_hbeta_is_reference(self):
        """Hβ (nearest to 4862.68) must have prior mean = 1.0 after normalization."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        means, _ = cloudy_grid_line_priors(grid, log_z=-1.0, neb_logU=-3.0)
        # Hβ is index 0 in our mock (4862.68 Å)
        hbeta_mean = float(means[0])
        assert abs(hbeta_mean - 1.0) < 1e-6, f"Hβ prior mean = {hbeta_mean}, expected 1.0"

    def test_all_positive(self):
        """All means and sigmas must be positive."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        means, sigmas = cloudy_grid_line_priors(grid, log_z=0.0, neb_logU=-2.5)
        assert jnp.all(means > 0), f"Non-positive prior means: {means}"
        assert jnp.all(sigmas > 0), f"Non-positive prior sigmas: {sigmas}"

    def test_target_wavelengths_subset(self):
        """target_wavelengths must return only matched lines."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        target = jnp.array([4862.68, 6564.61])  # Hβ and Hα vacuum
        means, sigmas = cloudy_grid_line_priors(
            grid, log_z=-1.0, neb_logU=-3.0, target_wavelengths=target
        )
        assert means.shape == (2,), f"Expected (2,), got {means.shape}"
        assert sigmas.shape == (2,), f"Expected (2,), got {sigmas.shape}"

    def test_oiii_stronger_at_higher_logU(self):
        """[OIII] 5007 must increase monotonically with ionization parameter."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        means_low_u, _ = cloudy_grid_line_priors(grid, log_z=-1.0, neb_logU=-3.5)
        means_high_u, _ = cloudy_grid_line_priors(grid, log_z=-1.0, neb_logU=-2.0)
        # [OIII]5007 is index 2 in our mock
        assert float(means_high_u[2]) > float(means_low_u[2]), (
            "[OIII]5007 must be stronger at higher logU"
        )

    def test_nii_weaker_at_lower_metallicity(self):
        """[NII] 6583 must be weaker at lower metallicity (less nitrogen)."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        means_solar, _ = cloudy_grid_line_priors(grid, log_z=0.0, neb_logU=-3.0)
        means_subsolar, _ = cloudy_grid_line_priors(grid, log_z=-2.0, neb_logU=-3.0)
        # [NII]6583 is index 4 in our mock
        assert float(means_subsolar[4]) < float(means_solar[4]), (
            "[NII]6583 must be weaker at lower metallicity"
        )

    def test_prior_width_dex_scales_sigmas(self):
        """Wider prior_width_dex must give larger sigmas."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        _, sigmas_narrow = cloudy_grid_line_priors(
            grid, log_z=-1.0, neb_logU=-3.0, prior_width_dex=0.1
        )
        _, sigmas_wide = cloudy_grid_line_priors(
            grid, log_z=-1.0, neb_logU=-3.0, prior_width_dex=0.5
        )
        assert jnp.all(sigmas_wide > sigmas_narrow), "Wider dex scatter must give larger sigmas"

    def test_clamped_to_grid_bounds(self):
        """Out-of-range inputs must clamp without raising."""
        from tengri.observation.eline_priors import cloudy_grid_line_priors

        grid = _MockGridData()
        # Way outside the grid bounds
        means, sigmas = cloudy_grid_line_priors(grid, log_z=5.0, neb_logU=0.0)
        chex.assert_tree_all_finite(means)
        chex.assert_tree_all_finite(sigmas)
