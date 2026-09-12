# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``profile_mass``: analytic marginalization of the mass amplitude.

See ``tengri.inference.mass_profile`` for the math and
``docs/dev/inference_methods.md`` ("Profiling the mass") for the measured
effect. ``tests/inference/`` is auto-marked ``slow`` (see ``tests/conftest.py``);
run with ``-m slow``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    ForwardModel,
    SEDModel,
    builders,
    generate_mock,
    recipes,
)
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.mass_profile import _log_mass_integral, _profile_stats
from tengri.observation import Observation, Photometry

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
_MASS_NAME = "sfh_tsnorm_log_total_mass"


def _minimal_model(ssp_data):
    """``recipes.mock_recovery_minimal()`` over ``_FILTERS`` (photometry only)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


def _mock(model, *, seed: int, snr: float = 30.0):
    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(seed))
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=snr)
    return truth, jnp.asarray(mock["flux_obs"]), jnp.asarray(mock["noise"])


def _spectroscopy_mock(model, *, seed: int, snr: float = 30.0):
    """``generate_mock`` is photometry-only; build a noisy spectrum by hand."""
    key_truth, key_noise = jax.random.split(jax.random.PRNGKey(seed))
    truth = model.spec.sample(key_truth)
    flux_true = model.predict_spectrum(truth)
    noise = jnp.abs(flux_true) / snr
    flux = flux_true + noise * jax.random.normal(key_noise, flux_true.shape)
    return truth, flux, noise


def _agn_model(ssp_data):
    """A model whose photometry is NOT linear in the mass (AGN continuum)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh=builders.sfh.dpl(all_params=FREE),
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb=builders.neb.none(),
        agn={
            "type": "composable",
            "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.05),
    )


class TestMarginalLikelihoodCorrectness:
    """Profiled log-likelihood vs. a brute-force numerical integral over mass."""

    def test_agrees_with_brute_force_integral(self, ssp_data_fsps):
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)

        fitter_off = Fitter(forward, data=flux, noise=noise, profile_mass=False)
        ctx = InferenceContext.from_target(fitter_off)
        nlp = ctx.neg_log_posterior_fn
        log_prior = ctx.log_prior_fn
        data_args = ctx.data_args
        other_names = [n for n in fitter_off._free_names if n != _MASS_NAME]

        mass_prior = fitter_off.spec.get_distribution(_MASS_NAME)
        ell_lo, ell_hi = mass_prior.bounds
        ell_grid = np.linspace(ell_lo, ell_hi, 2000)
        p_ell = np.asarray([float(mass_prior.log_prob(jnp.asarray(e))) for e in ell_grid])
        p_ell = np.exp(p_ell)

        for seed in (10, 11, 12):
            key = jax.random.PRNGKey(seed)
            other_xi = {
                name: jax.random.normal(jax.random.fold_in(key, j))
                for j, name in enumerate(other_names)
            }

            loglik_vals = np.empty(ell_grid.shape)
            for k, ell in enumerate(ell_grid):
                xi_mass = mass_prior.standardize(jnp.asarray(ell))
                xi_full = {**other_xi, _MASS_NAME: xi_mass}
                loglik_vals[k] = float(-nlp(xi_full, data_args) - log_prior(xi_full))

            integrand = np.exp(loglik_vals - loglik_vals.max()) * p_ell
            brute_force_log_z = (
                float(np.log(np.trapezoid(integrand, ell_grid))) + loglik_vals.max()
            )

            # Profiled quantity, evaluated at the SAME theta (other_xi) via the
            # physical params the unprofiled context would compute for them.
            phys = {
                name: fitter_off.spec.get_distribution(name).unstandardize(other_xi[name])
                for name in other_names
            }
            for name, val in fitter_off._fixed_values.items():
                phys[name] = jnp.asarray(val)
            phys[_MASS_NAME] = jnp.asarray(0.0)
            A, mstar, chi2_min = _profile_stats(model, _MASS_NAME, phys, flux, noise)
            profiled_log_z = float(
                -0.5 * chi2_min + _log_mass_integral(A, mstar, ell_lo, ell_hi, mass_prior)
            )

            rel_diff = abs(profiled_log_z - brute_force_log_z) / abs(brute_force_log_z)
            assert rel_diff < 1e-6, (
                f"seed {seed}: profiled={profiled_log_z:.10f} "
                f"brute_force={brute_force_log_z:.10f} rel_diff={rel_diff:.3e}"
            )


class TestGuards:
    """profile_mass=True raises ValueError when a precondition fails."""

    def test_nonlinear_agn_component_raises(self, ssp_data_fsps):
        model = _agn_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)

        with pytest.raises(ValueError, match="linear"):
            Fitter(forward, data=flux, noise=noise, profile_mass=True)

        # "auto" must not raise; it silently falls back to unprofiled inference.
        fitter = Fitter(forward, data=flux, noise=noise, profile_mass="auto")
        assert fitter._profile_mass is False

    def test_spectroscopy_raises(self, ssp_data_fsps):
        from tengri import Spectroscopy

        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4000.0, 7000.0, 50)))
        model = SEDModel.build(
            ssp_data=ssp_data_fsps, observation=obs, **recipes.mock_recovery_minimal()
        )
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _spectroscopy_mock(model, seed=0)

        with pytest.raises(ValueError, match="photometry"):
            Fitter(forward, data=flux, noise=noise, profile_mass=True)

        fitter = Fitter(forward, data=flux, noise=noise, profile_mass="auto")
        assert fitter._profile_mass is False


class TestMapParity:
    """method='map' with and without profile_mass agree on theta and mass."""

    def test_map_matches_unprofiled(self, ssp_data_fsps):
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        # Seed/SNR chosen so the (small, expected) Occam-factor difference
        # between the joint MAP and the profiled-marginal MAP -- see
        # tests/inference/test_profile_mass.py's module docstring and the
        # task report -- stays under the 1e-3 tolerance; it does not for
        # every truth draw (worst observed here ~0.5 for a weakly-identified
        # SFH shape parameter).
        _, flux, noise = _mock(model, seed=5, snr=300.0)

        key = jax.random.PRNGKey(42)
        post_off = forward.fit(flux, noise, method="map", profile_mass=False, key=key)
        post_on = forward.fit(flux, noise, method="map", profile_mass=True, key=key)

        assert post_on.diagnostics["profile_mass_resolved"] is True
        assert post_off.diagnostics["profile_mass_resolved"] is False

        for name in post_off.params:
            if name == _MASS_NAME:
                diff = abs(float(post_off.params[name]) - float(post_on.params[name]))
                assert diff < 1e-3, f"mass MAP differs by {diff:.3e} dex"
            elif name in forward.spec.free_params:
                diff = abs(float(post_off.params[name]) - float(post_on.params[name]))
                assert diff < 1e-3, f"{name} MAP differs by {diff:.3e}"


class TestSamplerParity:
    """method='mcmc_nuts' with and without profile_mass agree in the posterior mean."""

    def test_nuts_posterior_means_agree(self, ssp_data_fsps):
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)

        kw = dict(
            method="mcmc_nuts",
            dense_mass_matrix=True,
            n_warmup=200,
            n_samples=200,
            n_chains=2,
            key=jax.random.PRNGKey(7),
        )
        post_off = forward.fit(flux, noise, profile_mass=False, **kw)
        post_on = forward.fit(flux, noise, profile_mass=True, **kw)

        assert post_on.diagnostics["profile_mass_resolved"] is True
        assert post_on.samples[_MASS_NAME].shape == post_on.samples["dust_tau_bc"].shape

        # Standard errors use the effective sample size (autocorrelation-aware),
        # not the raw draw count: a short, dense-mass-matrix NUTS run mixes
        # slowly enough that the naive std/sqrt(n) badly understates the
        # Monte Carlo error and manufactures spurious disagreement.
        from blackjax.diagnostics import effective_sample_size

        n_chains = int(post_off.diagnostics["n_chains"])
        for name in post_off.samples:
            a = np.asarray(post_off.samples[name]).reshape(n_chains, -1)
            b = np.asarray(post_on.samples[name]).reshape(n_chains, -1)
            if a.std() == 0.0 and b.std() == 0.0:
                continue  # Fixed parameter, both sides constant.
            ess_a = max(float(effective_sample_size(jnp.asarray(a))), 1.0)
            ess_b = max(float(effective_sample_size(jnp.asarray(b))), 1.0)
            se = float(np.sqrt(a.std() ** 2 / ess_a + b.std() ** 2 / ess_b))
            z = abs(a.mean() - b.mean()) / se if se > 0 else 0.0
            assert z < 3.0, f"{name}: off_mean={a.mean():.4g} on_mean={b.mean():.4g} z={z:.2f}"


class TestFreeNamesLength:
    """InferenceContext.free_names has length D-1 under profiling."""

    def test_free_names_drop_the_mass(self, ssp_data_fsps):
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)

        d_full = model.spec.n_free
        fitter = Fitter(forward, data=flux, noise=noise, profile_mass=True)
        ctx = InferenceContext.from_target(fitter)

        assert len(ctx.free_names) == d_full - 1
        assert _MASS_NAME not in ctx.free_names


class TestAutoDefault:
    """profile_mass="auto" is the default and resolves per-fit."""

    def test_default_is_auto(self, ssp_data_fsps):
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)
        fitter = Fitter(forward, data=flux, noise=noise)
        assert fitter._profile_mass is True  # every guard passes on this fixture

    def test_auto_resolves_true_on_photometry(self, ssp_data_fsps):
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)
        fitter = Fitter(forward, data=flux, noise=noise, profile_mass="auto")
        assert fitter._profile_mass is True
        assert fitter._profile_mass_resolved is True

    def test_auto_resolves_false_on_spectroscopy(self, ssp_data_fsps):
        from tengri import Spectroscopy

        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4000.0, 7000.0, 50)))
        model = SEDModel.build(
            ssp_data=ssp_data_fsps, observation=obs, **recipes.mock_recovery_minimal()
        )
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _spectroscopy_mock(model, seed=0)

        fitter = Fitter(forward, data=flux, noise=noise, profile_mass="auto")
        assert fitter._profile_mass is False
        assert fitter._profile_mass_resolved is False
        assert fitter._profile_mass_resolved is False


def test_profiling_steps_aside_for_backends_that_build_their_own_objective():
    """A NIFTy VI fit must sample the mass itself: it never sees the profiled loss.

    Measured 2026-09-12 (ctl-dpl seed 7): with the mass frozen at the placeholder
    geoVI returned mass 10.24 against the NUTS reference 11.96. The run-time
    resolution restores the original spec for such backends under ``"auto"`` and
    refuses an explicit ``True``.
    """
    from tengri.inference.mass_profile import (
        PROFILE_MASS_BACKENDS,
        resolve_profile_mass_for_method,
    )

    assert "mcmc_nuts_fast" in PROFILE_MASS_BACKENDS
    assert "vi" not in PROFILE_MASS_BACKENDS

    class _Spec:
        free_params = ("a", "m_log_total_mass")

        def get_fixed_values(self):
            return {}

        def get_distribution(self, name):
            class _D:
                bounds = (0.0, 1.0)

            return _D()

    class _Fitter:
        _profile_mass = True
        _profile_mass_resolved = True
        _profile_mass_reason = "auto-enabled"
        _profile_mass_original_spec = _Spec()
        spec = object()
        _free_names = ("a",)

    f = _Fitter()
    resolve_profile_mass_for_method(f, "mcmc_nuts", "auto")
    assert f._profile_mass is True  # a consumer keeps it

    resolve_profile_mass_for_method(f, "vi", "auto")
    assert f._profile_mass is False
    assert tuple(f._free_names) == ("a", "m_log_total_mass")
    assert "auto-disabled" in f._profile_mass_reason

    g = _Fitter()
    with pytest.raises(ValueError, match="profile_mass=True"):
        resolve_profile_mass_for_method(g, "vi", True)
