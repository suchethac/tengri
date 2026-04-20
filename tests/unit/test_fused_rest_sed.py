"""Tests for the compositional rest-frame SED kernel (Tier 2).

Validates that the Tier 2 path (build_fused_rest_sed) produces results
matching the Tier 3 exact path (compute_sed_components) to <0.1% for
all supported component combinations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.sps.dsps_wrapper import SSPData
from tengri.forward.kernels import (
    observe_photometry_from_rest_sed,
    observe_spectrum_from_rest_sed,
)
from tengri.parameters.parameters import ParamSpec
from tengri.parameters.priors import Uniform

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Synthetic SSP data (no real data file dependency) ─────────────

_N_MET = 5
_N_AGE = 94
_N_WAVE = 300


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Create minimal synthetic SSP data for unit tests."""
    key = jax.random.PRNGKey(42)
    k1, _k2 = jax.random.split(key)

    wave = jnp.linspace(1000.0, 30000.0, _N_WAVE)
    lg_age_gyr = jnp.linspace(-2.0, 1.14, _N_AGE)
    lgmet = jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0])

    # Synthetic flux: smooth power-law SED with age/met variation
    # Shape: (n_met, n_age, n_wave)
    base_sed = (wave / 5500.0) ** (-0.5)  # (n_wave,)
    age_factor = 10.0 ** (-0.3 * lg_age_gyr)  # (n_age,)
    met_factor = jnp.linspace(0.5, 1.5, _N_MET)  # (n_met,)
    flux = met_factor[:, None, None] * age_factor[None, :, None] * base_sed[None, None, :]
    # Add small noise to break degeneracies
    flux = flux + 0.01 * jnp.abs(jax.random.normal(k1, flux.shape))

    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=lg_age_gyr,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def simple_spec():
    """Basic ParamSpec with DPL SFH and two-component dust."""
    return ParamSpec(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
    )


@pytest.fixture(scope="module")
def simple_params():
    """Fixed parameter set for reproducibility."""
    return {
        "sfh_dpl_alpha": 1.5,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 4.0,
        "sfh_dpl_log_peak_sfr": 1.0,
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.8,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }


# ── Test: Tier 2 compatibility check ──────────────────────────────


class TestTier2Compatibility:
    def test_tier2_kernel_builds(self, synthetic_ssp, simple_spec):
        from tengri.forward.sed_model import SEDModel

        model = SEDModel(simple_spec, synthetic_ssp)
        assert model._compositional.rest_sed is not None


# ── Test: Tier 2 vs Tier 3 agreement (core) ───────────────────────


class TestTier2VsTier3:
    """Validate Tier 2 produces results matching Tier 3 exact path."""

    def test_rest_sed_matches_exact(self, synthetic_ssp, simple_spec, simple_params):
        """Rest-frame SED from Tier 2 matches compute_sed_components."""
        from tengri.forward.sed_model import Model

        model = Model(simple_spec, synthetic_ssp)

        # Tier 3: exact path
        tier3_result = model._compute_sed_components(simple_params)
        sed_tier3 = tier3_result["sed_total"]

        # Tier 2: compositional kernel
        sed_tier2 = model._compute_rest_sed_compositional(simple_params)

        assert_allclose(sed_tier2, sed_tier3, rtol=1e-3, atol=1e-30)

    def test_photometry_matches_exact(self, synthetic_ssp, simple_params):
        """Tier 2 photometry matches Tier 3 photometry."""
        from tengri.forward.sed_model import Model
        from tengri.observation.filters import FilterCurve
        from tengri.observation.photometry import compute_flux_density

        spec = ParamSpec(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=0.3,
            dust_slope=-0.7,
            redshift=0.1,
        )

        # Create simple synthetic filter curves as FilterCurve namedtuples
        filter_curves = []
        filter_waves = []
        filter_trans = []
        centers = [4000.0, 6000.0, 8000.0]
        for i, c in enumerate(centers):
            filt_w = jnp.linspace(c - 500.0, c + 500.0, 50)
            filt_t = jnp.exp(-0.5 * ((filt_w - c) / 200.0) ** 2)
            filter_waves.append(filt_w)
            filter_trans.append(filt_t)
            filter_curves.append(FilterCurve(name=f"test_{i}", wave=filt_w, trans=filt_t))

        model = Model(spec, synthetic_ssp, filters=filter_curves)

        # Disable Tier 1 (fused photometry) to test Tier 2 path
        # Get Tier 3 reference by calling predict_sed + manual integration
        sed_tier3 = model.predict_rest_sed(simple_params).sed
        z = model._get_redshift(simple_params)
        dl_cm = model._get_dl_cm(simple_params)

        wave_rest = synthetic_ssp.ssp_wave
        phot_tier3 = jnp.array(
            [
                compute_flux_density(sed_tier3, wave_rest, fw, ft, z, dl_cm)
                for fw, ft in zip(filter_waves, filter_trans)
            ]
        )

        # Tier 2 path
        phot_tier2 = model._predict_photometry_compositional(simple_params)

        assert_allclose(phot_tier2, phot_tier3, rtol=1e-3, atol=1e-30)

    def test_spectrum_matches_exact(self, synthetic_ssp, simple_spec, simple_params):
        """Tier 2 spectrum matches Tier 3 spectrum."""
        from tengri.forward.sed_model import Model
        from tengri.observation.spectrum import compute_spectrum

        model = Model(simple_spec, synthetic_ssp)

        # Wavelength grid in observed frame
        wave_obs = jnp.linspace(4000.0, 9000.0, 100)

        # Tier 3: exact path
        sed_tier3 = model.predict_rest_sed(simple_params).sed
        z = model._get_redshift(simple_params)
        dl_cm = model._get_dl_cm(simple_params)
        spec_tier3 = compute_spectrum(sed_tier3, model.ssp_data.ssp_wave, wave_obs, z, dl_cm)

        # Tier 2 path
        spec_tier2 = model._predict_spectrum_compositional(simple_params, wave_obs)

        assert_allclose(spec_tier2, spec_tier3, rtol=1e-3, atol=1e-30)


# ── Test: component combinations ──────────────────────────────────


class TestComponentCombinations:
    """Test that enabling/disabling components doesn't break Tier 2."""

    def test_single_component_dust(self, synthetic_ssp):
        """Single-component dust model works with Tier 2."""
        from tengri.forward.sed_model import Model

        spec = ParamSpec(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_model="single_component",
            dust_tau_v=Uniform(0.0, 3.0),
            dust_slope=-0.7,
            redshift=0.1,
        )
        model = Model(spec, synthetic_ssp)
        assert model._compositional.rest_sed is not None

        params = {
            "sfh_dpl_alpha": 1.5,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            "sfh_dpl_log_peak_sfr": 1.0,
            "met_logzsol": -0.3,
            "dust_tau_v": 1.0,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }

        # Tier 2 should produce the same as Tier 3
        tier3 = model._compute_sed_components(params)["sed_total"]
        tier2 = model._compute_rest_sed_compositional(params)

        assert_allclose(tier2, tier3, rtol=1e-3, atol=1e-30)

    def test_dust_emission_mbb(self, synthetic_ssp, simple_params):
        """Modified blackbody dust emission works with Tier 2."""
        from tengri.forward.sed_model import Model

        spec = ParamSpec(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=0.3,
            dust_slope=-0.7,
            dust_emission="modified_blackbody",
            redshift=0.1,
        )
        model = Model(spec, synthetic_ssp)
        assert model._compositional.rest_sed is not None

        tier3 = model._compute_sed_components(simple_params)["sed_total"]
        tier2 = model._compute_rest_sed_compositional(simple_params)

        assert_allclose(tier2, tier3, rtol=1e-3, atol=1e-30)

    def test_stochastic_sfh(self, synthetic_ssp):
        """Stochastic (GP-modulated) SFH works with Tier 2."""
        from tengri.forward.sed_model import Model

        spec = ParamSpec(
            mean_sfh_type="dpl",
            stochastic=True,
            n_grid=32,
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            sfh_field_psd_sigma=Uniform(0.01, 1.0),
            sfh_field_psd_tau_myr=Uniform(10, 500),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=0.3,
            dust_slope=-0.7,
            redshift=0.1,
        )
        model = Model(spec, synthetic_ssp)
        assert model._compositional.rest_sed is not None

        # Sample params including GP latent vector
        params = spec.sample(jax.random.PRNGKey(7))

        tier3 = model._compute_sed_components(params)["sed_total"]
        tier2 = model._compute_rest_sed_compositional(params)

        assert_allclose(tier2, tier3, rtol=1e-3, atol=1e-30)


# ── Test: observation wrappers ────────────────────────────────────


class TestObservationWrappers:
    def test_observe_photometry(self, synthetic_ssp, simple_spec, simple_params):
        """observe_photometry_from_rest_sed matches manual integration."""
        from tengri.forward.sed_model import Model
        from tengri.observation.photometry import compute_flux_density

        model = Model(simple_spec, synthetic_ssp)
        rest_sed = model._compute_rest_sed_compositional(simple_params)
        wave_rest = model.ssp_data.ssp_wave
        z = model._get_redshift(simple_params)
        dl_cm = model._get_dl_cm(simple_params)

        # Manual filter curves
        fw1 = jnp.linspace(4000.0, 5000.0, 40)
        ft1 = jnp.ones(40)
        fw2 = jnp.linspace(7000.0, 8000.0, 40)
        ft2 = jnp.ones(40)

        expected = jnp.array(
            [
                compute_flux_density(rest_sed, wave_rest, fw1, ft1, z, dl_cm),
                compute_flux_density(rest_sed, wave_rest, fw2, ft2, z, dl_cm),
            ]
        )
        result = observe_photometry_from_rest_sed(
            rest_sed, wave_rest, z, dl_cm, [fw1, fw2], [ft1, ft2]
        )
        assert_allclose(result, expected, rtol=1e-10)

    def test_observe_spectrum(self, synthetic_ssp, simple_spec, simple_params):
        """observe_spectrum_from_rest_sed matches compute_spectrum."""
        from tengri.forward.sed_model import Model
        from tengri.observation.spectrum import compute_spectrum

        model = Model(simple_spec, synthetic_ssp)
        rest_sed = model._compute_rest_sed_compositional(simple_params)
        wave_rest = model.ssp_data.ssp_wave
        wave_obs = jnp.linspace(5000.0, 8000.0, 50)
        z = model._get_redshift(simple_params)
        dl_cm = model._get_dl_cm(simple_params)

        expected = compute_spectrum(rest_sed, wave_rest, wave_obs, z, dl_cm)
        result = observe_spectrum_from_rest_sed(rest_sed, wave_rest, wave_obs, z, dl_cm)
        assert_allclose(result, expected, rtol=1e-10)


# ── Test: edge cases and fallbacks ────────────────────────────────


class TestFallbacks:
    def test_tabulated_sfh_falls_back_to_tier3(self, synthetic_ssp, simple_spec, simple_params):
        """Tabulated SFH bypasses Tier 2 and uses Tier 3."""
        from tengri.forward.sed_model import Model

        model = Model(simple_spec, synthetic_ssp)
        assert model._compositional.rest_sed is not None

        # Add tabulated SFH params — should skip Tier 2
        params_tab = {
            **simple_params,
            "sfh_t_gyr": jnp.linspace(0.1, 13.7, 50),
            "sfh_sfr": jnp.ones(50) * 5.0,
        }

        # This should work (falls back to Tier 3)
        sed = model.predict_rest_sed(params_tab).sed
        assert sed.shape == (len(model.ssp_data.ssp_wave),)
        assert jnp.all(jnp.isfinite(sed))

    def test_tier2_jit_traces(self, synthetic_ssp, simple_spec, simple_params):
        """Tier 2 kernel can be JIT-compiled and re-called."""
        from tengri.forward.sed_model import Model

        model = Model(simple_spec, synthetic_ssp)
        assert model._compositional.rest_sed is not None

        # Call twice — second should use cached JIT
        sed1 = model._compute_rest_sed_compositional(simple_params)
        sed2 = model._compute_rest_sed_compositional(simple_params)

        assert_allclose(sed1, sed2, rtol=0.0, atol=0.0)

    def test_tier2_different_params(self, synthetic_ssp, simple_spec):
        """Tier 2 produces different SEDs for different params."""
        from tengri.forward.sed_model import Model

        model = Model(simple_spec, synthetic_ssp)

        params1 = {
            "sfh_dpl_alpha": 1.5,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            "sfh_dpl_log_peak_sfr": 1.0,
            "met_logzsol": -0.3,
            "dust_tau_bc": 0.5,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }
        params2 = {
            **params1,
            "dust_tau_bc": 2.0,
            "met_logzsol": 0.0,
        }

        sed1 = model._compute_rest_sed_compositional(params1)
        sed2 = model._compute_rest_sed_compositional(params2)

        # Different params should give different SEDs
        assert not jnp.allclose(sed1, sed2)


# ── Test: gradient flows through Tier 2 ───────────────────────────


class TestGradients:
    def test_gradient_through_tier2(self, synthetic_ssp, simple_spec):
        """Gradients through Tier 2 rest SED kernel match FD."""
        from tengri.forward.sed_model import Model

        model = Model(simple_spec, synthetic_ssp)

        def loss_fn(dust_tau_bc):
            params = {
                "sfh_dpl_alpha": 1.5,
                "sfh_dpl_beta": 1.0,
                "sfh_dpl_tau_gyr": 4.0,
                "sfh_dpl_log_peak_sfr": 1.0,
                "met_logzsol": -0.3,
                "dust_tau_bc": dust_tau_bc,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 0.1,
            }
            sed = model._compute_rest_sed_compositional(params)
            return jnp.sum(sed)

        x0 = 1.0
        grad_jax = float(jax.grad(loss_fn)(x0))
        grad_fd = fd_grad(loss_fn, x0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=5e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        # More dust absorption → lower total flux → non-positive gradient
        assert grad_jax <= 0.0


# ── Test: Fused Tier 2 end-to-end kernels ─────────────────────────


class TestFusedTier2Photometry:
    """Validate the fused end-to-end params → photometry JIT kernel."""

    def test_fused_tier2_phot_builds(self, synthetic_ssp, simple_spec):
        """Fused Tier 2 photometry kernel builds for fixed-z + filters."""
        from tengri.forward.sed_model import Model
        from tengri.observation.filters import FilterCurve

        filters = [
            FilterCurve(
                name=f"f{i}",
                wave=jnp.linspace(c - 500, c + 500, 50),
                trans=jnp.exp(-0.5 * ((jnp.linspace(c - 500, c + 500, 50) - c) / 200) ** 2),
            )
            for i, c in enumerate([4000.0, 6000.0, 8000.0])
        ]
        model = Model(simple_spec, synthetic_ssp, filters=filters)
        assert model._compositional.photometry is not None

    def test_fused_tier2_phot_matches_unfused(self, synthetic_ssp, simple_spec, simple_params):
        """Fused Tier 2 photometry matches unfused path."""
        from tengri.forward.sed_model import Model
        from tengri.observation.filters import FilterCurve

        filters = [
            FilterCurve(
                name=f"f{i}",
                wave=jnp.linspace(c - 500, c + 500, 50),
                trans=jnp.exp(-0.5 * ((jnp.linspace(c - 500, c + 500, 50) - c) / 200) ** 2),
            )
            for i, c in enumerate([4000.0, 6000.0, 8000.0])
        ]
        model = Model(simple_spec, synthetic_ssp, filters=filters)

        # Fused path (sfr_on_ssp computed outside JIT, passed as traced arg)
        phot_fused = model._predict_photometry_compositional(simple_params)

        # Unfused: force through _compute_rest_sed_compositional + filter loop
        saved_phot = model._compositional.photometry
        model._compositional.photometry = None
        phot_unfused = model._predict_photometry_compositional(simple_params)
        model._compositional.photometry = saved_phot

        assert_allclose(phot_fused, phot_unfused, rtol=1e-10)

    def test_fused_tier2_phot_gradient(self, synthetic_ssp, simple_spec):
        """Gradients through fused Tier 2 photometry match FD."""
        from tengri.forward.sed_model import Model
        from tengri.observation.filters import FilterCurve

        filters = [
            FilterCurve(
                name="r",
                wave=jnp.linspace(5500, 7000, 50),
                trans=jnp.ones(50),
            )
        ]
        model = Model(simple_spec, synthetic_ssp, filters=filters)
        assert model._compositional.photometry is not None

        def loss(dust_tau_bc):
            params = {
                "sfh_dpl_alpha": 1.5,
                "sfh_dpl_beta": 1.0,
                "sfh_dpl_tau_gyr": 4.0,
                "sfh_dpl_log_peak_sfr": 1.0,
                "met_logzsol": -0.3,
                "dust_tau_bc": dust_tau_bc,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 0.1,
            }
            return jnp.sum(model._predict_photometry_compositional(params))

        x0 = 1.0
        grad_jax = float(jax.grad(loss)(x0))
        grad_fd = fd_grad(loss, x0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=5e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_free_z_builds(self, synthetic_ssp):
        """Fused Tier 2 photometry builds even with free redshift."""
        from tengri.forward.sed_model import Model
        from tengri.observation.filters import FilterCurve

        spec = ParamSpec(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=0.3,
            dust_slope=-0.7,
            redshift=Uniform(0.01, 0.5),  # FREE
        )
        filters = [
            FilterCurve(
                name="r",
                wave=jnp.linspace(5500, 7000, 50),
                trans=jnp.ones(50),
            )
        ]
        model = Model(spec, synthetic_ssp, filters=filters)
        assert model._compositional.photometry is not None

        params = spec.sample(jax.random.PRNGKey(42))
        phot = model._predict_photometry_compositional(params)
        assert phot.shape == (1,)
        assert jnp.all(jnp.isfinite(phot))
