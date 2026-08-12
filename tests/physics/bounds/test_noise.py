# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.noise — NIFTy-native noise model.

Tests the core noise computation functions, the detection helper,
and the GGN metric-vector product for VariableCovarianceGaussian.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.observation.noise import (
    compute_effective_noise,
    compute_std_inv,
    has_noise_model,
    variable_noise_hamiltonian,
    variable_noise_metric_vec,
)
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── compute_effective_noise ───────────────────────────────────────


class TestComputeEffectiveNoise:
    """Tests for σ_eff = sqrt(σ²_obs + (f_cal · model)²)."""

    def test_zero_fcal_recovers_noise_obs(self):
        """f_cal=0 → σ_eff = σ_obs."""
        noise_obs = jnp.array([0.1, 0.2, 0.3])
        model_flux = jnp.array([10.0, 20.0, 30.0])
        result = compute_effective_noise(noise_obs, model_flux, 0.0)
        npt.assert_allclose(result, noise_obs, atol=1e-12)

    def test_hand_computed_values(self):
        """Check against hand-computed σ_eff."""
        noise_obs = jnp.array([0.1])
        model_flux = jnp.array([100.0])
        f_cal = 0.05
        # σ_eff = sqrt(0.01 + (5.0)^2) = sqrt(25.01) ≈ 5.001
        expected = jnp.sqrt(0.1**2 + (0.05 * 100.0) ** 2)
        result = compute_effective_noise(noise_obs, model_flux, f_cal)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_calibration_dominates_for_bright_sources(self):
        """For bright sources, calibration noise >> observational noise."""
        noise_obs = jnp.array([0.01])  # very small obs noise
        model_flux = jnp.array([1000.0])  # bright source
        f_cal = 0.05
        result = compute_effective_noise(noise_obs, model_flux, f_cal)
        # Calibration term: 0.05 * 1000 = 50 >> 0.01
        assert result[0] > 49.0

    def test_uses_absolute_model_flux(self):
        """Negative model flux should use |model| for calibration term."""
        noise_obs = jnp.array([0.1])
        model_pos = jnp.array([100.0])
        model_neg = jnp.array([-100.0])
        f_cal = 0.05
        result_pos = compute_effective_noise(noise_obs, model_pos, f_cal)
        result_neg = compute_effective_noise(noise_obs, model_neg, f_cal)
        npt.assert_allclose(result_pos, result_neg, atol=1e-12)

    def test_jit_compatible(self):
        """Function compiles under jax.jit."""
        noise_obs = jnp.array([0.1, 0.2])
        model_flux = jnp.array([10.0, 20.0])
        result = assert_jit_matches_eager(compute_effective_noise, noise_obs, model_flux, 0.05)
        chex.assert_shape(result, (2,))

    def test_grad_through_f_cal(self):
        """Gradient through f_cal is well-defined."""

        def loss(f_cal):
            sigma = compute_effective_noise(jnp.array([0.1]), jnp.array([10.0]), f_cal)
            return jnp.sum(sigma**2)

        grad_jax = float(jax.grad(loss)(0.05))
        grad_fd = fd_grad(loss, 0.05)
        npt.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax > 0  # increasing f_cal increases σ_eff

    def test_grad_through_model_flux(self):
        """Gradient through model flux is well-defined."""

        def loss(model_flux):
            sigma = compute_effective_noise(jnp.array([0.1]), model_flux, 0.05)
            return jnp.sum(sigma**2)

        grad_jax = float(jax.grad(loss)(jnp.array([10.0]))[0])
        grad_fd = fd_grad(lambda mf: float(loss(jnp.array([mf]))), 10.0)
        npt.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )


# ── compute_std_inv ───────────────────────────────────────────────


class TestComputeStdInv:
    """Tests for τ = 1/σ_eff."""

    def test_inverse_of_effective_noise(self):
        """τ = 1/σ_eff."""
        noise_obs = jnp.array([0.1, 0.2])
        model_flux = jnp.array([10.0, 20.0])
        f_cal = 0.05
        sigma = compute_effective_noise(noise_obs, model_flux, f_cal)
        tau = compute_std_inv(noise_obs, model_flux, f_cal)
        npt.assert_allclose(tau, 1.0 / sigma, rtol=1e-12)

    def test_zero_fcal(self):
        """f_cal=0 → τ = 1/σ_obs."""
        noise_obs = jnp.array([0.1, 0.5])
        model_flux = jnp.array([10.0, 20.0])
        tau = compute_std_inv(noise_obs, model_flux, 0.0)
        npt.assert_allclose(tau, 1.0 / noise_obs, atol=1e-12)


# ── has_noise_model ───────────────────────────────────────────────


class TestHasNoiseModel:
    """Tests for noise model detection."""

    def test_default_spec_no_noise(self):
        """Default Parameters has noise OFF (Fixed(0.0))."""
        from tengri import Fixed, Parameters, Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            sfh_dpl_tau_gyr=Uniform(1.0, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            redshift=0.1,
        )
        assert has_noise_model(spec) is False

    def test_free_noise_frac_cal(self):
        """noise_frac_cal as Uniform → noise model active."""
        from tengri import Fixed, Parameters, Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            sfh_dpl_tau_gyr=Uniform(1.0, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
            noise_frac_cal=Uniform(0.01, 0.2),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            redshift=0.1,
        )
        assert has_noise_model(spec) is True

    def test_fixed_nonzero_noise(self):
        """noise_frac_cal=Fixed(0.05) → noise model active."""
        from tengri import Fixed, Parameters, Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            sfh_dpl_tau_gyr=Uniform(1.0, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
            noise_frac_cal=Fixed(0.05),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            redshift=0.1,
        )
        assert has_noise_model(spec) is True

    def test_fixed_zero_no_noise(self):
        """noise_frac_cal=Fixed(0.0) → noise model OFF."""
        from tengri import Fixed, Parameters, Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            sfh_dpl_tau_gyr=Uniform(1.0, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
            noise_frac_cal=Fixed(0.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            redshift=0.1,
        )
        assert has_noise_model(spec) is False


# ── variable_noise_hamiltonian ────────────────────────────────────


class TestVariableNoiseHamiltonian:
    """Tests for the energy function E_lh = ½χ²_eff + Σlog(σ_eff)."""

    def test_matches_hand_computation(self):
        """Check against explicit formula."""
        data = jnp.array([1.0, 2.0, 3.0])
        noise_obs = jnp.array([0.1, 0.1, 0.1])
        predicted = jnp.array([1.1, 1.9, 3.2])
        f_cal = 0.05

        result = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal)

        # Manual computation
        sigma_eff = jnp.sqrt(noise_obs**2 + (f_cal * jnp.abs(predicted)) ** 2)
        chi2 = jnp.sum(((data - predicted) / sigma_eff) ** 2)
        logdet = jnp.sum(jnp.log(sigma_eff))
        expected = 0.5 * chi2 + logdet

        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_zero_fcal_matches_standard_chi2(self):
        """f_cal=0 → E = ½χ² + Σlog(σ_obs) (constant logdet)."""
        data = jnp.array([1.0, 2.0])
        noise_obs = jnp.array([0.1, 0.2])
        predicted = jnp.array([1.05, 2.1])

        result = variable_noise_hamiltonian(data, noise_obs, predicted, 0.0)
        chi2 = jnp.sum(((data - predicted) / noise_obs) ** 2)
        logdet = jnp.sum(jnp.log(noise_obs))
        expected = 0.5 * chi2 + logdet

        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_logdet_prevents_sigma_inflation(self):
        """Increasing f_cal increases logdet, preventing σ → ∞.

        At some point, the logdet penalty outweighs the chi2 reduction.
        The energy is NOT monotonically decreasing — there's a minimum.
        """
        data = jnp.array([1.0])
        noise_obs = jnp.array([0.1])
        predicted = jnp.array([1.5])  # large residual

        energies = []
        f_cal_values = [0.01, 0.1, 0.5, 1.0, 5.0, 50.0, 500.0]
        for f_cal in f_cal_values:
            e = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal)
            energies.append(float(e))

        # Energy should NOT be monotonically decreasing.
        # For very large f_cal, logdet dominates and energy rises.
        assert energies[-1] > min(energies), (
            "Energy at f_cal=500 should exceed the minimum — logdet penalty "
            f"must prevent σ→∞. Energies: {list(zip(f_cal_values, energies))}"
        )

    def test_jit_and_grad(self):
        """JIT + gradient through f_cal."""
        data = jnp.array([1.0, 2.0])
        noise_obs = jnp.array([0.1, 0.2])
        predicted = jnp.array([1.1, 1.9])

        @jax.jit
        def energy(f_cal):
            return variable_noise_hamiltonian(data, noise_obs, predicted, f_cal)

        grad_jax = float(jax.grad(energy)(0.05))
        grad_fd = fd_grad(
            lambda f_cal: float(variable_noise_hamiltonian(data, noise_obs, predicted, f_cal)),
            0.05,
        )
        npt.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )


# ── Student-t energy ──────────────────────────────────────────────


class TestStudentTEnergy:
    """Tests for the Student-t likelihood energy."""

    def test_student_t_matches_hand_computation(self):
        """Check Student-t energy against formula."""
        data = jnp.array([1.0, 2.0])
        noise_obs = jnp.array([0.1, 0.2])
        predicted = jnp.array([1.05, 2.1])
        f_cal = 0.05
        dof = 2.0

        result = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=dof)

        # Manual: E = (ν+1)/2 · Σ log(1 + r²/ν) + Σ log(σ_eff)
        sigma_eff = jnp.sqrt(noise_obs**2 + (f_cal * jnp.abs(predicted)) ** 2)
        r = (data - predicted) / sigma_eff
        expected = 0.5 * (dof + 1.0) * jnp.sum(jnp.log(1.0 + r**2 / dof))
        expected += jnp.sum(jnp.log(sigma_eff))
        npt.assert_allclose(float(result), float(expected), rtol=1e-10)

    def test_student_t_converges_to_gaussian_for_large_dof(self):
        """Student-t(ν→∞) → Gaussian energy."""
        data = jnp.array([1.0, 2.0])
        noise_obs = jnp.array([0.1, 0.2])
        predicted = jnp.array([1.05, 2.1])
        f_cal = 0.05

        e_gauss = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=None)
        e_t_large = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=1e6)
        npt.assert_allclose(float(e_t_large), float(e_gauss), rtol=1e-4)

    def test_student_t_downweights_outliers(self):
        """A 5σ outlier should contribute less energy with Student-t."""
        data = jnp.array([1.0, 10.0])  # second point is outlier
        noise_obs = jnp.array([0.1, 0.1])
        predicted = jnp.array([1.0, 1.0])
        f_cal = 0.0

        e_gauss = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=None)
        e_t2 = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=2.0)

        # Gaussian: outlier contributes 0.5 * (9/0.1)^2 = 4050
        # Student-t(2): outlier contributes 1.5 * log(1 + 8100/2) ≈ 12.6
        assert e_t2 < e_gauss, "Student-t should give lower energy for outliers"

    def test_student_t_jit_and_grad(self):
        """JIT + gradient through Student-t energy."""
        data = jnp.array([1.0, 2.0])
        noise_obs = jnp.array([0.1, 0.2])
        predicted = jnp.array([1.1, 1.9])

        @jax.jit
        def energy(f_cal):
            return variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=2.0)

        grad_jax = float(jax.grad(energy)(0.05))
        grad_fd = fd_grad(
            lambda f_cal: float(
                variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=2.0)
            ),
            0.05,
        )
        npt.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )


class TestUsesStudentT:
    """Tests for Student-t detection."""

    def test_default_no_student_t(self):
        """Default spec uses Gaussian."""
        from tengri import Fixed, Parameters, Uniform
        from tengri.observation.noise import uses_student_t

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            sfh_dpl_tau_gyr=Uniform(1.0, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            redshift=0.1,
        )
        assert uses_student_t(spec) is False

    def test_fixed_dof_activates(self):
        """noise_dof=Fixed(2.0) → Student-t active."""
        from tengri import Fixed, Parameters, Uniform
        from tengri.observation.noise import uses_student_t

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            sfh_dpl_tau_gyr=Uniform(1.0, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
            noise_frac_cal=Uniform(0.01, 0.2),
            noise_dof=Fixed(2.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            redshift=0.1,
        )
        assert uses_student_t(spec) is True

    def test_zero_dof_no_student_t(self):
        """noise_dof=Fixed(0.0) → Gaussian (default)."""
        from tengri import Fixed, Parameters, Uniform
        from tengri.observation.noise import uses_student_t

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            sfh_dpl_tau_gyr=Uniform(1.0, 12.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.0),
            noise_dof=Fixed(0.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            redshift=0.1,
        )
        assert uses_student_t(spec) is False


# ── variable_noise_metric_vec ─────────────────────────────────────


class TestVariableNoiseMetricVec:
    """Tests for the GGN metric-vector product."""

    @pytest.fixture()
    def setup(self):
        """Create a simple test case with 2 data points, 3 latent dims."""
        data = jnp.array([1.0, 2.0])
        noise_obs = jnp.array([0.1, 0.2])

        # Simple linear model: f = [w1, w2], tau = 1/sqrt(noise^2 + (w3*f)^2)
        def signal_noise_fn(primals):
            predicted = primals["signal"]
            f_cal = primals["f_cal"]
            sigma_eff = jnp.sqrt(noise_obs**2 + (f_cal * jnp.abs(predicted)) ** 2)
            std_inv = 1.0 / sigma_eff
            return predicted, std_inv

        param_keys = ["f_cal", "signal"]

        def flatten(d):
            return jnp.concatenate([jnp.atleast_1d(d[k]).ravel() for k in param_keys])

        def unflatten(x):
            return {"f_cal": x[0], "signal": x[1:3]}

        xi = flatten({"f_cal": jnp.array(0.05), "signal": jnp.array([1.1, 1.9])})
        v = flatten({"f_cal": jnp.array(0.01), "signal": jnp.array([0.1, -0.1])})

        return {
            "xi": xi,
            "v": v,
            "signal_noise_fn": signal_noise_fn,
            "data": data,
            "unflatten": unflatten,
            "flatten": flatten,
        }

    def test_returns_correct_shape(self, setup):
        """M @ v has same shape as v."""
        result = variable_noise_metric_vec(
            setup["xi"],
            setup["v"],
            setup["signal_noise_fn"],
            setup["data"],
            setup["unflatten"],
            setup["flatten"],
        )
        assert result.shape == setup["v"].shape

    def test_positive_definite(self, setup):
        """v^T M v > 0 for nonzero v (M is positive definite)."""
        Mv = variable_noise_metric_vec(
            setup["xi"],
            setup["v"],
            setup["signal_noise_fn"],
            setup["data"],
            setup["unflatten"],
            setup["flatten"],
        )
        vTMv = jnp.dot(setup["v"], Mv)
        assert vTMv > 0, f"Expected v^T M v > 0, got {vTMv}"

    def test_agrees_with_finite_diff_hessian(self, setup):
        """Compare to numerical Hessian-vector product.

        The metric should approximate the Hessian of the energy
        near the expansion point.
        """
        data = setup["data"]
        noise_obs = jnp.array([0.1, 0.2])
        unflatten = setup["unflatten"]

        def energy(xi_flat):
            primals = unflatten(xi_flat)
            predicted = primals["signal"]
            f_cal = primals["f_cal"]
            sigma_eff = jnp.sqrt(noise_obs**2 + (f_cal * jnp.abs(predicted)) ** 2)
            chi2 = jnp.sum(((data - predicted) / sigma_eff) ** 2)
            logdet = jnp.sum(jnp.log(sigma_eff))
            prior = jnp.sum(xi_flat**2)
            return 0.5 * chi2 + logdet + 0.5 * prior

        xi = setup["xi"]
        v = setup["v"]

        # Hessian-vector product via forward-over-reverse
        hess_vp = jax.jvp(jax.grad(energy), (xi,), (v,))[1]

        # GGN metric-vector product
        ggn_vp = variable_noise_metric_vec(
            xi,
            v,
            setup["signal_noise_fn"],
            data,
            unflatten,
            setup["flatten"],
        )

        # GGN is a positive-definite approximation to the Hessian.
        # They should agree when the model is near the data (small residuals).
        # Check that they're in the same ballpark
        assert jnp.allclose(hess_vp, ggn_vp, atol=0.5), f"Hessian VP: {hess_vp}, GGN VP: {ggn_vp}"

    def test_jit_compatible(self, setup):
        """Compiles under jax.jit."""
        result = assert_jit_matches_eager(
            lambda xi, v: variable_noise_metric_vec(
                xi,
                v,
                setup["signal_noise_fn"],
                setup["data"],
                setup["unflatten"],
                setup["flatten"],
            ),
            setup["xi"],
            setup["v"],
        )
        chex.assert_tree_all_finite(result)
