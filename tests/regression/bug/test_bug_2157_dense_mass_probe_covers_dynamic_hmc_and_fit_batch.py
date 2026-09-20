# SPDX-License-Identifier: BSD-3-Clause
"""#2157: dense-mass step-size stability probe covers dynamic-HMC and fit_batch.

The post-adaptation density probe (_stabilize_dense_mass_step) was initially
wired into only NUTS and HMC single-galaxy fresh-adaptation branches. This test
suite validates that the probe now also runs after adaptation in the dynamic-HMC
backend and in fit_batch's shared window adaptation.

Taxonomy: regression_bug
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    SEDModel,
    Uniform,
    builders,
)

pytestmark = pytest.mark.regression_bug

_SSP_PATH = Path(__file__).resolve().parents[2] / "data" / "ssp_bc03_miles_chabrier.h5"
_SSP_AVAILABLE = _SSP_PATH.exists()


def _build_small_model():
    """Build a D ≤ 30 model for fast testing."""
    import tengri

    ssp = tengri.load_ssp("bc03_miles_chabrier")

    obs = Observation(
        photometry={
            "sdss_u": np.array([1.0]),
            "sdss_g": np.array([1.0]),
            "sdss_r": np.array([1.0]),
            "sdss_i": np.array([1.0]),
        }
    )

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=builders.sfh.dpl(all_params=Fixed(DEFAULT), log_total_mass=FREE, alpha=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=Fixed(DEFAULT),
            law="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )
    return model


@pytest.mark.skipif(not _SSP_AVAILABLE, reason="SSP data not available")
class TestDenseProbeWiringDynamicHMC:
    """Test that the dense-mass probe is wired into dynamic-HMC adaptation."""

    @pytest.mark.integration
    def test_dynamic_hmc_probe_called_with_dense_mass_true(self, monkeypatch):
        """Probe is called exactly once during dynamic-HMC with dense_mass_matrix=True."""
        from tengri import Data
        from tengri.inference.backends.mcmc import dynamic_hmc

        model = _build_small_model()

        # Create synthetic data
        truth = model.spec.sample(jax.random.PRNGKey(0))
        truth = {
            **truth,
            "sfh_dpl_log_total_mass": jnp.array(10.0),
            "sfh_dpl_alpha": jnp.array(0.8),
            "dust_tau_bc": jnp.array(0.2),
            "dust_tau_diff": jnp.array(0.1),
        }
        truth_full = {**model.spec.get_fixed_values(), **{k: float(v) for k, v in truth.items()}}
        flux = np.asarray(model.predict_photometry(truth_full))
        noise = flux / 10.0

        # Recorder to track probe calls
        call_log = []

        def recorder(
            kernel,
            state,
            ld_fn,
            data_args,
            step_size,
            inv_mass_matrix,
            max_doublings,
            sampler_name="NUTS",
        ):
            call_log.append({"sampler_name": sampler_name})
            # Return stable step (no backoff needed)
            return step_size, 0

        monkeypatch.setattr(dynamic_hmc, "_stabilize_dense_mass_step", recorder)

        from tengri import ForwardModel

        forward = ForwardModel.build(sed=model)
        posterior = forward.fit(
            Data(photometry=flux, noise_photometry=noise),
            key=jax.random.PRNGKey(1),
            method="mcmc_hmc",
            n_warmup=100,
            n_samples=50,
            dense_mass_matrix=True,
            precondition=False,
        )

        assert len(call_log) == 1, (
            f"probe should be called exactly once, was called {len(call_log)} times"
        )
        assert call_log[0]["sampler_name"] == "HMC", "sampler name should be HMC"

    @pytest.mark.integration
    def test_dynamic_hmc_probe_not_called_with_dense_mass_false(self, monkeypatch):
        """Probe is never called during dynamic-HMC with dense_mass_matrix=False."""
        from tengri import Data
        from tengri.inference.backends.mcmc import dynamic_hmc

        model = _build_small_model()

        # Create synthetic data
        truth = model.spec.sample(jax.random.PRNGKey(0))
        truth = {
            **truth,
            "sfh_dpl_log_total_mass": jnp.array(10.0),
            "sfh_dpl_alpha": jnp.array(0.8),
            "dust_tau_bc": jnp.array(0.2),
            "dust_tau_diff": jnp.array(0.1),
        }
        truth_full = {**model.spec.get_fixed_values(), **{k: float(v) for k, v in truth.items()}}
        flux = np.asarray(model.predict_photometry(truth_full))
        noise = flux / 10.0

        # Recorder to track probe calls
        call_log = []

        def recorder(
            kernel,
            state,
            ld_fn,
            data_args,
            step_size,
            inv_mass_matrix,
            max_doublings,
            sampler_name="NUTS",
        ):
            call_log.append(True)
            return step_size, 0

        monkeypatch.setattr(dynamic_hmc, "_stabilize_dense_mass_step", recorder)

        from tengri import ForwardModel

        forward = ForwardModel.build(sed=model)
        posterior = forward.fit(
            Data(photometry=flux, noise_photometry=noise),
            key=jax.random.PRNGKey(1),
            method="mcmc_hmc",
            n_warmup=100,
            n_samples=50,
            dense_mass_matrix=False,
            precondition=False,
        )

        assert len(call_log) == 0, (
            f"probe should never be called, was called {len(call_log)} times"
        )


@pytest.mark.skipif(not _SSP_AVAILABLE, reason="SSP data not available")
class TestDenseProbeWiringFitBatch:
    """Test that the dense-mass probe is wired into fit_batch adaptation."""

    @pytest.mark.integration
    def test_fit_batch_probe_called_with_dense_mass_true(self, monkeypatch):
        """Probe called once in fit_batch's shared adaptation with dense_mass_matrix=True."""
        from tengri import CatalogFitter, Data
        from tengri.inference import fitter

        model = _build_small_model()

        # Create 2-galaxy batch
        truth = model.spec.sample(jax.random.PRNGKey(0))
        truth = {
            **truth,
            "sfh_dpl_log_total_mass": jnp.array(10.0),
            "sfh_dpl_alpha": jnp.array(0.8),
            "dust_tau_bc": jnp.array(0.2),
            "dust_tau_diff": jnp.array(0.1),
        }
        truth_full = {**model.spec.get_fixed_values(), **{k: float(v) for k, v in truth.items()}}
        flux_1 = np.asarray(model.predict_photometry(truth_full))
        flux_2 = np.asarray(model.predict_photometry(truth_full))
        noise = flux_1 / 10.0

        # Recorder to track probe calls with backoff count
        call_log = []
        backoff_count = [2]  # Will be returned by recorder

        def recorder(
            kernel,
            state,
            ld_fn,
            data_args,
            step_size,
            inv_mass_matrix,
            max_doublings,
            sampler_name="NUTS",
        ):
            call_log.append({"sampler_name": sampler_name})
            # Return stable step with backoff count = 2 to prove the plumbing
            return step_size, backoff_count[0]

        monkeypatch.setattr(fitter, "_stabilize_dense_mass_step", recorder)

        # Create batch data
        batch_flux = np.stack([flux_1, flux_2])
        batch_noise = np.stack([noise, noise])

        cat_fitter = CatalogFitter(model)
        posteriors = cat_fitter.fit_batch(
            Data(photometry=batch_flux, noise_photometry=batch_noise),
            key=jax.random.PRNGKey(1),
            method="mcmc_hmc",
            n_warmup=100,
            n_samples=50,
            dense_mass_matrix=True,
            precondition=False,
        )

        # Shared adaptation means probe is called once
        assert len(call_log) == 1, (
            f"probe called {len(call_log)} times, expected 1 in shared adaptation"
        )
        assert call_log[0]["sampler_name"] == "HMC", "sampler name should be HMC"

        # Check first posterior's diagnostics carry the backoff count
        diag = posteriors[0].diagnostics
        assert "dense_mass_backoffs" in diag, "diagnostics should carry dense_mass_backoffs"
        assert diag["dense_mass_backoffs"] == 2, (
            f"backoff count should be 2, got {diag['dense_mass_backoffs']}"
        )

    @pytest.mark.integration
    def test_fit_batch_probe_not_called_with_dense_mass_false(self, monkeypatch):
        """Probe is never called in fit_batch with dense_mass_matrix=False."""
        from tengri import CatalogFitter, Data
        from tengri.inference import fitter

        model = _build_small_model()

        # Create 2-galaxy batch
        truth = model.spec.sample(jax.random.PRNGKey(0))
        truth = {
            **truth,
            "sfh_dpl_log_total_mass": jnp.array(10.0),
            "sfh_dpl_alpha": jnp.array(0.8),
            "dust_tau_bc": jnp.array(0.2),
            "dust_tau_diff": jnp.array(0.1),
        }
        truth_full = {**model.spec.get_fixed_values(), **{k: float(v) for k, v in truth.items()}}
        flux_1 = np.asarray(model.predict_photometry(truth_full))
        flux_2 = np.asarray(model.predict_photometry(truth_full))
        noise = flux_1 / 10.0

        # Recorder to track probe calls
        call_log = []

        def recorder(
            kernel,
            state,
            ld_fn,
            data_args,
            step_size,
            inv_mass_matrix,
            max_doublings,
            sampler_name="NUTS",
        ):
            call_log.append(True)
            return step_size, 0

        monkeypatch.setattr(fitter, "_stabilize_dense_mass_step", recorder)

        # Create batch data
        batch_flux = np.stack([flux_1, flux_2])
        batch_noise = np.stack([noise, noise])

        cat_fitter = CatalogFitter(model)
        posteriors = cat_fitter.fit_batch(
            Data(photometry=batch_flux, noise_photometry=batch_noise),
            key=jax.random.PRNGKey(1),
            method="mcmc_hmc",
            n_warmup=100,
            n_samples=50,
            dense_mass_matrix=False,
            precondition=False,
        )

        assert len(call_log) == 0, (
            f"probe should never be called, was called {len(call_log)} times"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
