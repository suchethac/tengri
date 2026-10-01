# SPDX-License-Identifier: BSD-3-Clause
"""#2157: dense-mass step-size stability probe covers dynamic-HMC and fit_batch.

The post-adaptation stability probe (``_stabilize_dense_mass_step``) was
initially wired into only the NUTS and HMC single-galaxy fresh-adaptation
branches. These tests pin the two seams this fix adds: ``run_dynamic_hmc``
and ``Fitter._fit_batch_vmap_mcmc`` (the shared window adaptation reached
through ``fit_batch``).

Wiring notes the first draft got wrong, kept here so they stay wrong-proof:

- ``dynamic_hmc.py`` imports the probe at module top, so the recorder must be
  patched into the ``dynamic_hmc`` NAMESPACE — patching ``_shared`` leaves the
  already-imported name untouched and the recorder never fires (which is why
  the draft could only assert ``posterior is not None``).
- ``fitter.py`` imports the probe INSIDE ``_fit_batch_vmap_mcmc``, so there
  patching ``_shared`` is exactly right.
- Both positive tests assert the recorder actually FIRED with the seam's own
  sampler name; a wiring test that passes with the recorder silent is theater.
- The vmap seam is only reached through ``fit_batch`` with an MCMC method,
  >=2 same-shape galaxies and a fixed-z photometry precompute; the recorder
  assertion doubles as proof the batch actually took the vmap route.

Taxonomy: regression_bug
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SSPData,
    Uniform,
    builders,
)

pytestmark = pytest.mark.regression_bug


def _build_small_model(ssp=None):
    """Build a D <= 30 model for fast testing."""
    if ssp is None:
        # Build a synthetic SSP suitable for testing (no disk I/O)
        n_met, n_age = 3, 20
        wave = jnp.logspace(2.0, 5.5, 400)  # 100 Å – 300 μm
        ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
        lgmet = jnp.array([-4.0, -2.65, -1.3])
        base = (5000.0 / wave) ** 2
        flux = (
            base[None, None, :]
            * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
            * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
        )
        flux = jnp.abs(flux) + 1e-12
        ssp = SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)

    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]))

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


def _free_truth(model):
    """A free-only truth dict (#2296: params dicts must not carry Fixed keys)."""
    truth = dict(model.spec.sample(jax.random.PRNGKey(0)))
    truth.update(
        {
            "sfh_dpl_log_total_mass": jnp.array(10.0),
            "sfh_dpl_alpha": jnp.array(0.8),
            "dust_tau_bc": jnp.array(0.2),
            "dust_tau_diff": jnp.array(0.1),
        }
    )
    return {k: float(v) for k, v in truth.items()}


def _mock_flux_noise(model):
    flux = np.asarray(model.predict_photometry(_free_truth(model)))
    return flux, flux / 10.0


def _make_recorder(call_log):
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
        return step_size, 0

    return recorder


_FIT_KW = dict(n_warmup=60, n_samples=20, precondition=False)


class TestDenseProbeWiringDynamicHMC:
    """The probe runs after dynamic-HMC's fresh adaptation iff the mass is dense."""

    def _run(self, monkeypatch, dense):
        from tengri import Data, ForwardModel
        from tengri.inference.backends.mcmc import dynamic_hmc as dynamic_hmc_mod

        model = _build_small_model()
        flux, noise = _mock_flux_noise(model)

        call_log = []
        monkeypatch.setattr(
            dynamic_hmc_mod, "_stabilize_dense_mass_step", _make_recorder(call_log)
        )

        forward = ForwardModel.build(sed=model)
        posterior = forward.fit(
            Data(photometry=(flux, noise)),
            key=jax.random.PRNGKey(1),
            method="mcmc_dynamic_hmc",
            dense_mass_matrix=dense,
            **_FIT_KW,
        )
        assert posterior is not None
        return call_log

    def test_probe_fires_with_dense_mass(self, monkeypatch):
        call_log = self._run(monkeypatch, dense=True)
        assert call_log == [{"sampler_name": "Dynamic HMC"}], (
            f"the dynamic-HMC seam must call the probe exactly once, got {call_log}"
        )

    def test_probe_silent_with_diagonal_mass(self, monkeypatch):
        call_log = self._run(monkeypatch, dense=False)
        assert call_log == [], f"diagonal mass must skip the probe, got {call_log}"


class TestDenseProbeWiringFitBatchGHMC:
    """GHMC is excluded from the batch probe even under a dense mass.

    ``_stabilize_dense_mass_step`` drives an HMC/NUTS-shaped kernel; the GHMC
    kernel takes an extra positional (delta), so probing it raises TypeError
    mid-fit. The contract is a deliberate skip: the fit completes and the
    probe never fires.
    """

    def test_probe_skipped_for_ghmc_dense(self, monkeypatch):
        from tengri.inference.backends.mcmc import _shared
        from tengri.inference.fitter import Fitter

        model = _build_small_model()
        flux, noise = _mock_flux_noise(model)

        call_log = []
        monkeypatch.setattr(_shared, "_stabilize_dense_mass_step", _make_recorder(call_log))

        fitter = Fitter(model, data=jnp.asarray(flux), noise=jnp.asarray(noise))
        batch = [
            {"flux_obs": jnp.asarray(flux), "noise": jnp.asarray(noise)},
            {"flux_obs": jnp.asarray(flux) * 1.1, "noise": jnp.asarray(noise)},
        ]
        posteriors = fitter.fit_batch(
            batch,
            method="mcmc_ghmc",
            key=jax.random.PRNGKey(1),
            verbose=False,
            dense_mass_matrix=True,
            **_FIT_KW,
        )
        assert len(posteriors) == 2
        assert call_log == [], (
            f"GHMC must not be probed (kernel signature mismatch), got {call_log}"
        )


class TestDenseProbeWiringFitBatch:
    """The probe runs in fit_batch's shared vmap adaptation iff the mass is dense."""

    def _run(self, monkeypatch, dense):
        from tengri.inference.backends.mcmc import _shared
        from tengri.inference.fitter import Fitter

        model = _build_small_model()
        flux, noise = _mock_flux_noise(model)

        call_log = []
        # _fit_batch_vmap_mcmc imports the probe inside the method, so the
        # _shared attribute is resolved at call time — patching it there works.
        monkeypatch.setattr(_shared, "_stabilize_dense_mass_step", _make_recorder(call_log))

        fitter = Fitter(model, data=jnp.asarray(flux), noise=jnp.asarray(noise))
        batch = [
            {"flux_obs": jnp.asarray(flux), "noise": jnp.asarray(noise)},
            {"flux_obs": jnp.asarray(flux) * 1.1, "noise": jnp.asarray(noise)},
        ]
        posteriors = fitter.fit_batch(
            batch,
            method="mcmc_hmc",
            key=jax.random.PRNGKey(1),
            verbose=False,
            dense_mass_matrix=dense,
            **_FIT_KW,
        )
        assert len(posteriors) == 2
        return call_log

    def test_probe_fires_with_dense_mass(self, monkeypatch):
        call_log = self._run(monkeypatch, dense=True)
        assert call_log == [{"sampler_name": "MCMC-HMC"}], (
            "the fit_batch vmap seam must call the probe exactly once (shared "
            f"adaptation, not per galaxy), got {call_log}"
        )

    def test_probe_silent_with_diagonal_mass(self, monkeypatch):
        call_log = self._run(monkeypatch, dense=False)
        assert call_log == [], f"diagonal mass must skip the probe, got {call_log}"
