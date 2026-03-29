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

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.models.observation.eline_priors import (
    cloudy_line_priors,
    marginalize_emission_lines_cloudy,
)

# ---------------------------------------------------------------------------
# cloudy_line_priors tests
# ---------------------------------------------------------------------------


class TestCloudyLinePriors:
    """Tests for CLOUDY line ratio priors."""

    def test_returns_correct_number_of_lines(self):
        """Should return priors for all 11 reference lines by default."""
        means, sigmas = cloudy_line_priors()
        assert means.shape == (11,)
        assert sigmas.shape == (11,)

    def test_halpha_hbeta_ratio(self):
        """Halpha/Hbeta should be ~2.86 (Case B recombination)."""
        means, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)

        # H-alpha is index 7, H-beta is index 3 in the reference array
        halpha_idx = 7
        hbeta_idx = 3
        ratio = means[halpha_idx] / means[hbeta_idx]
        assert jnp.allclose(ratio, 2.86, atol=0.01), f"Ha/Hb = {ratio:.3f}, expected 2.86"

    def test_oiii_5007_gt_4959(self):
        """[OIII] 5007 should be ~3x [OIII] 4959."""
        means, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)

        # [OIII] 4959 is index 4, [OIII] 5007 is index 5
        oiii_4959 = means[4]
        oiii_5007 = means[5]
        ratio = oiii_5007 / oiii_4959
        assert ratio > 2.5, f"[OIII] 5007/4959 = {ratio:.2f}, expected ~3"
        assert ratio < 3.5, f"[OIII] 5007/4959 = {ratio:.2f}, expected ~3"

    def test_hbeta_is_reference(self):
        """Hbeta (index 3) should have ratio = 1.0."""
        means, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)
        assert jnp.allclose(means[3], 1.0, atol=0.01)

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

        # [NII] 6583 is index 8
        assert means_lowz[8] < means_solar[8], "Low-Z [NII] should be weaker"

    def test_high_ionization_stronger_oiii(self):
        """At higher logU, [OIII] lines should be stronger."""
        means_lowu, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)
        means_highu, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.0)

        # [OIII] 5007 is index 5
        assert means_highu[5] > means_lowu[5], "Higher logU -> stronger [OIII]"

    def test_specific_wavelengths(self):
        """Requesting specific wavelengths should return matched priors."""
        # Request only Halpha and Hbeta
        lines = jnp.array([6563.0, 4861.33])
        means, sigmas = cloudy_line_priors(line_wavelengths=lines)
        assert means.shape == (2,)
        assert sigmas.shape == (2,)

        # First should be Halpha (~2.86), second should be Hbeta (1.0)
        assert means[0] > 2.0  # Halpha
        assert jnp.allclose(means[1], 1.0, atol=0.1)  # Hbeta

    def test_jit_compatible(self):
        """cloudy_line_priors should be JIT-compilable."""

        @jax.jit
        def _eval():
            return cloudy_line_priors(log_z=0.0, neb_logU=-3.0)

        means, _sigmas = _eval()
        assert means.shape == (11,)
        assert jnp.all(jnp.isfinite(means))


# ---------------------------------------------------------------------------
# marginalize_emission_lines_cloudy integration tests
# ---------------------------------------------------------------------------


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

        # Simple Gaussian design matrix (3 lines)
        line_waves = jnp.array([4861.33, 5007.0, 6563.0])
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
        assert a_hat.shape == (3,)
        assert a_cov.shape == (3, 3)

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
        assert jnp.all(jnp.isfinite(a_hat))
        assert jnp.all(jnp.isfinite(a_cov))

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
        """Scaling l_hbeta should scale the prior means linearly."""
        d = mock_spectral_data

        _, a_hat_1, _ = marginalize_emission_lines_cloudy(
            d["residual"],
            d["noise"],
            d["design_matrix"],
            line_wavelengths=d["line_wavelengths"],
            l_hbeta=1.0,
        )

        _, a_hat_10, _ = marginalize_emission_lines_cloudy(
            d["residual"],
            d["noise"],
            d["design_matrix"],
            line_wavelengths=d["line_wavelengths"],
            l_hbeta=10.0,
        )

        # With zero residual, a_hat should scale with l_hbeta
        # (the prior mean dominates when there's no signal)
        ratio = a_hat_10 / jnp.maximum(a_hat_1, 1e-30)
        assert jnp.all(ratio > 5.0), "l_hbeta=10 should give ~10x larger amplitudes"
