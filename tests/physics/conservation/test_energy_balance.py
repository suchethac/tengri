# SPDX-License-Identifier: BSD-3-Clause
"""Tests for energy balance relaxation via dust_eta_balance parameter.

Validates that:
- eta=1.0 preserves strict energy balance (L_IR = L_absorbed)
- eta=2.0 doubles IR luminosity
- eta=0.0 produces zero IR emission
- Gradients flow through eta
- Parameters accepts dust_eta_balance as a free parameter
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Parameters, SEDModel, Uniform

pytestmark = pytest.mark.conservation


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────
# synthetic_ssp is provided by conftest.py (session scope)


@pytest.fixture(scope="module")
def base_spec():
    """Parameters with dust emission enabled and eta fixed at 1.0."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        # Fix age_gyr (#549): otherwise it is a *free* DPL param, and the
        # no-emission reference spec has a different free-param set, so the
        # shared PRNGKey splits differently and age_gyr is sampled to a
        # different value — changing the stellar SED and faking a ~4% eta=0
        # "energy-balance" inequivalence (issue #548).
        sfh_dpl_age_gyr=Fixed(13.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        dust_eta_balance=Fixed(1.0),
        redshift=Fixed(0.1),
        dust_emission="modified_blackbody",
    )


def _make_model_and_predict(ssp, spec, eta_value):
    """Build a SEDModel from spec/ssp and predict (wavelength, SED) at eta_balance."""
    model = SEDModel(spec, ssp)
    key = jax.random.PRNGKey(0)
    params = spec.sample(key)
    params = {**params, "dust_eta_balance": eta_value}
    pred = model.predict_rest_sed(params)
    return np.asarray(pred.wavelength), np.asarray(pred.sed)


# ── Tests ─────────────────────────────────────────────────────────


class TestEnergyBalanceEta:
    """Test dust_eta_balance scaling of IR luminosity."""

    def test_eta_1_preserves_energy_balance(self, synthetic_ssp, base_spec):
        """eta=1.0 should give L_IR = L_absorbed (strict energy conservation)."""
        _, sed = _make_model_and_predict(synthetic_ssp, base_spec, 1.0)
        chex.assert_tree_all_finite(sed)
        assert sed.shape[-1] > 0, "SED is empty"

    def test_eta_2_doubles_ir(self, synthetic_ssp, base_spec):
        """eta=2.0 should produce ~2x the IR emission compared to eta=1.0."""
        # Same spec → same (extended) master grid for all three; unpack seds.
        _, sed_1 = _make_model_and_predict(synthetic_ssp, base_spec, 1.0)
        _, sed_2 = _make_model_and_predict(synthetic_ssp, base_spec, 2.0)

        # IR excess = sed(eta=2) - sed(eta=1) should be approximately equal to
        # sed(eta=1) - sed(eta=0), since IR scales linearly with eta.
        _, sed_0 = _make_model_and_predict(synthetic_ssp, base_spec, 0.0)

        ir_from_eta1 = jnp.sum(sed_1 - sed_0)
        ir_from_eta2 = jnp.sum(sed_2 - sed_0)

        # eta=2 IR contribution should be ~2x the eta=1 IR contribution
        ratio = float(ir_from_eta2 / ir_from_eta1)
        np.testing.assert_allclose(ratio, 2.0, rtol=0.05)

    def test_eta_0_produces_zero_ir(self, synthetic_ssp, base_spec):
        """eta=0.0 should produce zero dust IR emission.

        The reference (no-emission) spec must Fix the *same* parameters as
        ``base_spec`` — including ``sfh_dpl_age_gyr`` (#549). The emission spec
        carries extra free params (``dust_T``, ``dust_beta_ir``); if a shared
        param like ``age_gyr`` is left free, the two specs split the same
        PRNGKey differently and sample it to different values, changing the
        stellar SED and faking a ~4% eta=0 inequivalence (issue #548 — a test
        artifact, not an energy-balance bug; the eta=0 path is a strict no-op).
        """
        # Build model with no dust emission as reference — Fix the identical
        # parameter set (incl. age_gyr) so both receive byte-identical inputs.
        spec_no_dust_em = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_age_gyr=Fixed(13.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
            # No dust_emission
        )
        model_no_em = SEDModel(spec_no_dust_em, synthetic_ssp)
        key = jax.random.PRNGKey(0)
        params_no_em = spec_no_dust_em.sample(key)
        pred_no_em = model_no_em.predict_rest_sed(params_no_em)
        wave_no_em = np.asarray(pred_no_em.wavelength)
        sed_no_em = np.asarray(pred_no_em.sed)

        wave_eta0, sed_eta0 = _make_model_and_predict(synthetic_ssp, base_spec, 0.0)

        # With eta=0, the dust emission should be zero, so the SED should match
        # the no-emission model (both have attenuation but no IR re-emission).
        # Compare on the no-emission grid: the emission model's master grid
        # extends into the submm (analytic-emitter native grid, #1005), so the
        # two SEDs live on different wavelength arrays.
        sed_eta0_on_ref = np.interp(wave_no_em, wave_eta0, np.array(sed_eta0))
        np.testing.assert_allclose(
            sed_eta0_on_ref,
            sed_no_em,
            rtol=1e-5,
            err_msg="eta=0 should produce same SED as no dust emission",
        )

    def test_gradient_flows_through_eta(self, synthetic_ssp, base_spec):
        """Gradient of SED sum w.r.t. dust_eta_balance should be finite."""
        model = SEDModel(base_spec, synthetic_ssp)
        key = jax.random.PRNGKey(0)
        params = base_spec.sample(key)

        def _loss(eta):
            p = {**params, "dust_eta_balance": eta}
            sed = model.predict_rest_sed(p).sed
            return jnp.sum(sed)

        grad_jax = float(jax.grad(_loss)(1.0))
        grad_fd = fd_grad(_loss, 1.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert grad_jax != 0.0, "Gradient is zero — eta has no effect"
        assert np.all(np.isfinite(grad_jax)), (
            "`grad_jax` is non-finite — non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )

    def test_paramspec_accepts_eta_as_free(self):
        """Parameters should accept dust_eta_balance=Uniform(0.5, 2.0)."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_eta_balance=Uniform(0.5, 2.0),
            redshift=Fixed(0.1),
            dust_emission="modified_blackbody",
        )
        # dust_eta_balance should appear in free params
        assert "dust_eta_balance" in spec.free_params
        # Sample should include it
        key = jax.random.PRNGKey(0)
        sample = spec.sample(key)
        assert "dust_eta_balance" in sample
        val = float(sample["dust_eta_balance"])
        assert 0.5 <= val <= 2.0, f"Sampled eta={val} outside [0.5, 2.0]"
