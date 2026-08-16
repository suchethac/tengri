# SPDX-License-Identifier: BSD-3-Clause
"""Tests for emission line design matrix and marginalization."""

from __future__ import annotations

import types

import chex
import pytest

pytestmark = pytest.mark.bounds

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.observation.eline_marginalization import (
    apply_doublet_constraints,
    build_eline_design_matrix,
    expand_constrained_amplitudes,
    marginalize_emission_lines,
)
from tengri.observation.line_list import LineList
from tengri.observation.spectroscopy import Spectroscopy


class TestDesignMatrix:
    def test_shape(self):
        wave = jnp.linspace(4000, 7000, 1000)
        lines = jnp.array([4862.68, 5008.24, 6564.61])  # vacuum Hβ, [OIII]5007, Hα
        G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
        chex.assert_shape(G, (1000, 3))

    def test_normalized_profiles(self):
        """Each column should integrate to approximately 1."""
        wave = jnp.linspace(4000, 7000, 5000)
        lines = jnp.array([4862.68, 5008.24, 6564.61])  # vacuum Hβ, [OIII]5007, Hα
        G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
        dlam = float(jnp.mean(jnp.diff(wave)))
        for i in range(3):
            integral = float(jnp.sum(G[:, i])) * dlam
            assert abs(integral - 1.0) < 0.05

    def test_velocity_broadening_widens_profile(self):
        wave = jnp.linspace(6500, 6620, 500)
        ha = jnp.array([6564.61])  # vacuum Hα
        G_narrow = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_sigma_kms=0.0)
        G_broad = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_sigma_kms=200.0)
        fwhm_narrow = int(jnp.sum(G_narrow[:, 0] > 0.5 * G_narrow[:, 0].max()))
        fwhm_broad = int(jnp.sum(G_broad[:, 0] > 0.5 * G_broad[:, 0].max()))
        assert fwhm_broad > fwhm_narrow

    def test_velocity_offset_shifts_center(self):
        wave = jnp.linspace(6500, 6650, 1000)
        ha = jnp.array([6564.61])  # vacuum Hα
        G_zero = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_delta_v_kms=0.0)
        G_red = build_eline_design_matrix(wave, ha, 2000.0, 0.0, eline_delta_v_kms=300.0)
        peak_zero = float(wave[jnp.argmax(G_zero[:, 0])])
        peak_red = float(wave[jnp.argmax(G_red[:, 0])])
        assert peak_red > peak_zero

    def test_redshift_shifts_lines(self):
        wave = jnp.linspace(9000, 10000, 500)
        ha = jnp.array([6564.61])  # vacuum Hα
        z = 0.5
        G = build_eline_design_matrix(wave, ha, 1000.0, z)
        peak = float(wave[jnp.argmax(G[:, 0])])
        assert abs(peak - 6564.61 * 1.5) < 20.0

    def test_backward_compatible_4arg_call(self):
        """Old 4-argument call must still work."""
        wave = jnp.linspace(4000, 7000, 500)
        lines = jnp.array([6564.61])  # vacuum Hα
        G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
        chex.assert_shape(G, (500, 1))


class TestDoubletConstraints:
    def test_constraint_reduces_columns(self):
        cat = LineList.default_optical()
        C = cat.build_constraint_matrix()
        wave = jnp.linspace(4000, 7000, 1000)
        G_full = build_eline_design_matrix(wave, cat.wavelengths, 2000.0, 0.0)
        G_eff = apply_doublet_constraints(G_full, C)
        assert G_eff.shape[1] == cat.n_independent
        assert G_eff.shape[1] < G_full.shape[1]

    def test_doublet_ratio_enforced(self):
        """After marginalization with constraints, OIII ratio should be 2.98."""
        cat = LineList.default_optical()
        C = cat.build_constraint_matrix()
        wave = jnp.linspace(4800, 5100, 1000)
        G_full = build_eline_design_matrix(wave, cat.wavelengths, 2000.0, 0.0)

        # Inject OIII at correct ratio
        true_amps = jnp.zeros(cat.n_lines)
        i_5007 = list(cat.names).index("OIII_5007")
        i_4959 = list(cat.names).index("OIII_4959")
        true_amps = true_amps.at[i_5007].set(10.0)
        true_amps = true_amps.at[i_4959].set(10.0 / 2.98)

        spectrum = G_full @ true_amps
        noise = jnp.full_like(spectrum, 0.01)
        G_eff = apply_doublet_constraints(G_full, C)
        prior_var = jnp.full(G_eff.shape[1], 1000.0**2)
        _, a_hat, a_cov = marginalize_emission_lines(
            spectrum, noise, G_eff, prior_variance=prior_var
        )
        a_full, _ = expand_constrained_amplitudes(a_hat, a_cov, C)
        ratio = float(a_full[i_5007]) / float(a_full[i_4959])
        assert abs(ratio - 2.98) < 0.01


class TestMarginalization:
    def test_recovers_injected_line(self):
        wave = jnp.linspace(6400, 6700, 500)
        ha = jnp.array([6564.61])  # vacuum Hα
        G = build_eline_design_matrix(wave, ha, 2000.0, 0.0)
        true_flux = jnp.array([5.0])
        continuum = jnp.ones_like(wave) * 10.0
        key = jax.random.PRNGKey(0)
        noise_arr = jnp.full_like(wave, 0.1)
        data = continuum + G @ true_flux + 0.01 * jax.random.normal(key, wave.shape)
        prior_var = jnp.full(1, 100.0**2)
        _, a_hat, _ = marginalize_emission_lines(
            data - continuum, noise_arr, G, prior_variance=prior_var
        )
        assert abs(float(a_hat[0]) - 5.0) < 0.5

    def test_gradient_through_marginalization(self):
        """Gradient of -ln_L w.r.t. continuum level must be finite and correctly signed.

        Physical check: marginalizing over a line amplitude, the continuum-level
        gradient must point toward the true continuum.
        - Under-subtracted (level < truth): residual inflated → increasing level
          reduces -ln_L → d(-ln_L)/d(level) < 0.
        - Over-subtracted (level > truth): residual negative → increasing level
          makes it worse → d(-ln_L)/d(level) > 0.
        """
        wave = jnp.linspace(6400, 6700, 200)
        ha = jnp.array([6564.61])  # vacuum Hα
        noise_arr = jnp.full_like(wave, 0.1)

        G = build_eline_design_matrix(wave, ha, 2000.0, 0.0)
        # Inject a true Hα signal on top of a flat continuum=1.0
        data = G @ jnp.array([5.0]) + 1.0

        def neg_log_like(level):
            residual = data - level * jnp.ones_like(wave)
            prior_var = jnp.full(1, 100.0**2)
            ln_l, _, _ = marginalize_emission_lines(
                residual, noise_arr, G, prior_variance=prior_var
            )
            return -ln_l

        # At correct continuum, gradient is near zero (minimum of -ln_L)
        g_at_true_jax = float(jax.grad(neg_log_like)(1.0))
        g_at_true_fd = fd_grad(neg_log_like, 1.0)
        np.testing.assert_allclose(
            g_at_true_jax,
            g_at_true_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_at_true_jax:.4e}, FD={g_at_true_fd:.4e}",
        )
        assert abs(g_at_true_jax) < 2.0, f"Gradient too large at true continuum: {g_at_true_jax}"

        # Under-subtracted: d(-ln_L)/d(level) < 0
        g_low_jax = float(jax.grad(neg_log_like)(0.0))
        g_low_fd = fd_grad(neg_log_like, 0.0)
        np.testing.assert_allclose(
            g_low_jax,
            g_low_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_low_jax:.4e}, FD={g_low_fd:.4e}",
        )
        assert g_low_jax < 0.0, (
            f"Expected negative gradient when continuum under-subtracted, got {g_low_jax:.4f}"
        )

        # Over-subtracted: d(-ln_L)/d(level) > 0
        g_high_jax = float(jax.grad(neg_log_like)(3.0))
        g_high_fd = fd_grad(neg_log_like, 3.0)
        np.testing.assert_allclose(
            g_high_jax,
            g_high_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_high_jax:.4e}, FD={g_high_fd:.4e}",
        )
        assert g_high_jax > 0.0, (
            f"Expected positive gradient when continuum over-subtracted, got {g_high_jax:.4f}"
        )

    def test_jit_compiles(self):
        wave = jnp.linspace(4000, 7000, 500)
        lines = jnp.array([4862.68, 5008.24, 6564.61])  # vacuum Hβ, [OIII]5007, Hα

        @jax.jit
        def run(data, noise_arr):
            G = build_eline_design_matrix(wave, lines, 2000.0, 0.0)
            prior_var = jnp.full(3, 100.0**2)
            return marginalize_emission_lines(
                data - jnp.ones_like(data), noise_arr, G, prior_variance=prior_var
            )

        data = jnp.ones(500) * 10.0
        noise_arr = jnp.ones(500) * 0.1
        ln_l, a_hat, _a_cov = run(data, noise_arr)
        assert jnp.isfinite(ln_l)
        chex.assert_shape(a_hat, (3,))

    def test_gradient_matches_finite_difference(self):
        """AD gradient of -ln_L w.r.t. continuum level must match finite-difference.

        Regression for NEW-09: isfinite alone does not catch wrong sign, missing
        terms, or off-by-constant errors in the marginalization formula.
        Tolerance 1% is well within JAX float64 accuracy.
        """
        wave = jnp.linspace(6400, 6700, 200)
        ha = jnp.array([6564.61])
        noise_arr = jnp.full_like(wave, 0.1)
        G = build_eline_design_matrix(wave, ha, 2000.0, 0.0)
        data = G @ jnp.array([5.0]) + 1.0

        def neg_log_like(level):
            residual = data - level * jnp.ones_like(wave)
            prior_var = jnp.full(1, 100.0**2)
            ln_l, _, _ = marginalize_emission_lines(
                residual, noise_arr, G, prior_variance=prior_var
            )
            return -ln_l

        eps = 1e-5
        fd_grad = (neg_log_like(1.0 + eps) - neg_log_like(1.0 - eps)) / (2 * eps)
        ad_grad = float(jax.grad(neg_log_like)(1.0))
        fd_val = float(fd_grad)
        denom = max(abs(fd_val), 1e-10)
        assert abs(ad_grad - fd_val) / denom < 0.01, (
            f"AD grad {ad_grad:.6g} vs FD grad {fd_val:.6g} — relative error too large"
        )


class TestFittedMode:
    """Regression tests for eline_mode='fitted' (IMP-03).

    Uses a SimpleNamespace mock model to avoid SSP data dependencies.
    """

    def _wave(self, n: int = 300) -> jnp.ndarray:
        return jnp.linspace(4000.0, 7000.0, n)

    def _make_spec(self):
        """Minimal Parameters with one free SFH param, rest fixed."""
        import warnings

        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed, Uniform

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Parameters(
                mean_sfh_type="dpl",
                sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
                met_logzsol=Fixed(-0.3),
                dust_tau_bc=Fixed(0.1),
                dust_tau_diff=Fixed(0.1),
                redshift=Fixed(0.1),
            )

    def _make_model(self, wave, model_spec, spec_cfg):
        """Minimal model mock as a SimpleNamespace."""
        continuum = jnp.ones(len(wave)) * 10.0
        return types.SimpleNamespace(
            spec=model_spec,
            wave_obs=wave,
            _spectral_resolution=2000.0,
            _spectroscopy_config=spec_cfg,
            predict_spectrum=lambda params, w=None, **kwargs: continuum,
            observation=None,
        )

    # ── Config-level tests (no SSP, no Fitter) ────────────────────

    def test_spectroscopy_fitted_mode_no_error(self):
        """Spectroscopy(eline_mode='fitted') must not raise NotImplementedError."""
        wave = self._wave()
        cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")
        assert cfg.eline_mode == "fitted"

    def test_has_eline_fitting_true_for_fitted(self):
        """has_eline_fitting property must return True for 'fitted' mode."""
        wave = self._wave()
        cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")
        assert cfg.has_eline_fitting is True

    # ── Parameters-level test ──────────────────────────────────────

    def test_merge_observation_params_adds_free_params(self):
        """merge_observation_params must add params to free_params and leave original intact."""
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed, Uniform

        spec = Parameters(mean_sfh_type="dpl", redshift=Fixed(0.1))
        n_before = len(spec.free_params)

        aug = spec.merge_observation_params(eline_amp_Halpha=Uniform(-1000.0, 1000.0))

        assert len(aug.free_params) == n_before + 1
        assert "eline_amp_Halpha" in aug.free_params
        # Original spec must be unmodified
        assert "eline_amp_Halpha" not in spec.free_params
        assert "eline_amp_Halpha" not in spec._valid_param_names

    # ── Fitter-level tests (mock model, no SSP) ───────────────────

    def test_fitter_sets_eline_fitted_flag(self):
        """Fitter must set _eline_fitted=True when eline_mode='fitted'."""
        from tengri.inference.fitter import Fitter

        wave = self._wave()
        cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")
        model = self._make_model(wave, self._make_spec(), cfg)

        fitter = Fitter(
            model, jnp.ones(len(wave)) * 10.0, jnp.ones(len(wave)) * 0.1, data_type="spectroscopy"
        )

        assert fitter._eline_fitted is True
        assert len(fitter._eline_amplitude_names) > 0
        assert all(nm.startswith("eline_amp_") for nm in fitter._eline_amplitude_names)

    def test_amplitude_params_appear_in_free_names(self):
        """Every amplitude param registered in Fitter must appear in _free_names."""
        from tengri.inference.fitter import Fitter

        wave = self._wave()
        cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")
        model = self._make_model(wave, self._make_spec(), cfg)

        fitter = Fitter(
            model, jnp.ones(len(wave)) * 10.0, jnp.ones(len(wave)) * 0.1, data_type="spectroscopy"
        )

        for amp_name in fitter._eline_amplitude_names:
            assert amp_name in fitter._free_names, (
                f"{amp_name!r} missing from _free_names; present: {sorted(fitter._free_names)[:6]}"
            )

    def test_amplitude_count_matches_independent_lines(self):
        """Number of amplitude params must equal n_independent in the catalog."""
        from tengri.inference.fitter import Fitter

        wave = self._wave()
        cat = LineList.default_13()
        cfg = Spectroscopy(
            wave_obs=wave, eline_mode="fitted", eline_catalog=cat, eline_fix_doublets=True
        )
        model = self._make_model(wave, self._make_spec(), cfg)

        fitter = Fitter(
            model, jnp.ones(len(wave)) * 10.0, jnp.ones(len(wave)) * 0.1, data_type="spectroscopy"
        )

        assert len(fitter._eline_amplitude_names) == cat.n_independent

    def test_loss_fn_finite_with_zero_amplitudes(self):
        """Loss function must return a finite scalar when all amplitudes are zero."""
        from tengri.inference.fitter import Fitter

        wave = self._wave()
        cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")
        model = self._make_model(wave, self._make_spec(), cfg)

        n_pix = len(wave)
        data = jnp.ones(n_pix) * 10.0
        noise = jnp.ones(n_pix) * 0.1
        fitter = Fitter(model, data, noise, data_type="spectroscopy")

        loss_fn = fitter._build_loss_fn()
        params_u = {nm: jnp.array(0.0) for nm in fitter._free_names}
        loss_val = loss_fn(params_u, {"data": data, "noise": noise})

        assert jnp.isfinite(loss_val), f"Loss is not finite: {loss_val}"

    def test_loglikelihood_fn_finite(self):
        """Log-likelihood function must return a finite scalar."""
        from tengri.inference.fitter import Fitter

        wave = self._wave()
        cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")
        model = self._make_model(wave, self._make_spec(), cfg)

        n_pix = len(wave)
        data = jnp.ones(n_pix) * 10.0
        noise = jnp.ones(n_pix) * 0.1
        fitter = Fitter(model, data, noise, data_type="spectroscopy")

        ll_fn = fitter._build_loglikelihood_fn()
        params_u = {nm: jnp.array(0.0) for nm in fitter._free_names}
        ll_val = ll_fn(params_u, {"data": data, "noise": noise})

        assert jnp.isfinite(ll_val), f"Log-likelihood is not finite: {ll_val}"

    def test_loss_lower_with_true_amplitude(self):
        """Loss must decrease when the Hα amplitude matches the injected signal.

        Physical check: Fitter with correct line amplitude should fit the data
        better (lower chi2) than Fitter with zero amplitudes when an Hα feature
        is present in the spectrum.

        Uses a fine wavelength grid centered on Hα (0.3 Å/pix) so the line
        profile is well-sampled even at R=2000 (σ≈1.4 Å).
        """
        from tengri.inference.fitter import Fitter
        from tengri.utils.transforms import to_unbounded

        # Fine grid centered on Hα: 0.3 Å/pix → line well-sampled at R=2000
        wave = jnp.linspace(6500.0, 6630.0, 440)
        cat = LineList.default_13()
        cfg = Spectroscopy(
            wave_obs=wave, eline_mode="fitted", eline_catalog=cat, eline_fix_doublets=True
        )

        # Build G_eff to inject Hα into the mock spectrum
        G_full = build_eline_design_matrix(wave, cat.wavelengths, 2000.0, 0.0)
        C = cat.build_constraint_matrix()
        G_eff = apply_doublet_constraints(G_full, C)

        # Locate Hα column in G_eff (independent lines, same order as amplitude names)
        secondary_indices = {dc.secondary_idx for dc in cat.doublets}
        independent_names = [nm for i, nm in enumerate(cat.names) if i not in secondary_indices]
        ha_col = independent_names.index("Halpha")

        true_amp = 50.0
        continuum = jnp.ones(len(wave)) * 10.0
        true_spectrum = continuum + G_eff[:, ha_col] * true_amp
        noise = jnp.ones(len(wave)) * 0.5

        # Mock model returns the flat continuum (lines not included in predict_spectrum)
        # Use redshift=0.0 so the loss function builds G at z=0, matching signal injection above.
        import warnings

        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed, Uniform

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model_spec = Parameters(
                mean_sfh_type="dpl",
                sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
                met_logzsol=Fixed(-0.3),
                dust_tau_bc=Fixed(0.1),
                dust_tau_diff=Fixed(0.1),
                redshift=Fixed(0.0),  # z=0 so Hα stays at 6564 Å, within the wave grid
            )
        model = types.SimpleNamespace(
            spec=model_spec,
            wave_obs=wave,
            _spectral_resolution=2000.0,
            _spectroscopy_config=cfg,
            predict_spectrum=lambda params, w=None, **kwargs: continuum,
            observation=None,
        )

        fitter = Fitter(model, true_spectrum, noise, data_type="spectroscopy")
        loss_fn = fitter._build_loss_fn()
        data_args = {"data": true_spectrum, "noise": noise}

        # Loss with all amplitudes at zero (midpoint of bounded range)
        params_zero = {nm: jnp.array(0.0) for nm in fitter._free_names}
        loss_zero = float(loss_fn(params_zero, data_args))

        # Loss with Hα amplitude set to the injected value (convert to unbounded space)
        ha_amp_name = "eline_amp_Halpha"
        amp_bound = 10.0 * cfg.eline_prior_sigma  # 1000.0
        u_ha = float(to_unbounded(jnp.array(true_amp), -amp_bound, amp_bound))
        params_true = {**params_zero, ha_amp_name: jnp.array(u_ha)}
        loss_true = float(loss_fn(params_true, data_args))

        assert loss_true < loss_zero, (
            f"Expected loss_true ({loss_true:.3f}) < loss_zero ({loss_zero:.3f}) "
            f"when amplitude matches injected Hα signal"
        )
