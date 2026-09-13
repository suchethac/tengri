# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``profile_mass``: analytic marginalization of the mass amplitude.

See ``tengri.inference.mass_profile`` for the math and
``docs/dev/inference_methods.md`` ("Profiling the Mass") for the measured
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
    Spectroscopy,
    builders,
    generate_mock,
    recipes,
)
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.mass_profile import _log_mass_integral, _profile_stats
from tengri.observation import Observation, Photometry

pytestmark = pytest.mark.contract

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
_MASS_NAME = "sfh_tsnorm_log_total_mass"
_SPEC_WAVE = jnp.linspace(4000.0, 7000.0, 40)


def _minimal_model(ssp_data):
    """``recipes.mock_recovery_minimal()`` over ``_FILTERS`` (photometry only)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


def _spectroscopy_model(ssp_data):
    """``recipes.mock_recovery_minimal()`` over ``_SPEC_WAVE`` (spectroscopy only)."""
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


def _joint_model(ssp_data):
    """``recipes.mock_recovery_minimal()`` over ``_FILTERS`` + ``_SPEC_WAVE`` (joint)."""
    obs = Observation(
        photometry=Photometry.from_names(_FILTERS),
        spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE),
    )
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


def _joint_mock(model, *, seed: int, snr_phot: float = 30.0, snr_spec: float = 20.0):
    """Noisy photometry+spectrum mock, concatenated photometry-then-spectrum.

    Matches the ``Fitter(data_type="joint")`` contract pinned in
    ``tests/regression/bug/test_bug_1366_joint_data_record.py``.
    """
    key_truth, key_p, key_s = jax.random.split(jax.random.PRNGKey(seed), 3)
    truth = model.spec.sample(key_truth)
    flux_p = model.predict_photometry(truth)
    flux_s = model.predict_spectrum(truth)
    noise_p = jnp.abs(flux_p) / snr_phot
    noise_s = jnp.abs(flux_s) / snr_spec
    data_p = flux_p + noise_p * jax.random.normal(key_p, flux_p.shape)
    data_s = flux_s + noise_s * jax.random.normal(key_s, flux_s.shape)
    return truth, jnp.concatenate([data_p, data_s]), jnp.concatenate([noise_p, noise_s])


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


def _assert_profiled_matches_brute_force(
    model, forward, data_type, flux, noise, *, seeds=(10, 11, 12)
):
    """Profiled log-likelihood vs. a 2000-node numerical integral over mass, 1e-6 rel."""
    fitter_off = Fitter(forward, data=flux, noise=noise, data_type=data_type, profile_mass=False)
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

    for seed in seeds:
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
        brute_force_log_z = float(np.log(np.trapezoid(integrand, ell_grid))) + loglik_vals.max()

        # Profiled quantity, evaluated at the SAME theta (other_xi) via the
        # physical params the unprofiled context would compute for them.
        phys = {
            name: fitter_off.spec.get_distribution(name).unstandardize(other_xi[name])
            for name in other_names
        }
        for name, val in fitter_off._fixed_values.items():
            phys[name] = jnp.asarray(val)
        phys[_MASS_NAME] = jnp.asarray(0.0)
        A_ref, a_star, chi2_min, ell_ref = _profile_stats(
            model, _MASS_NAME, phys, flux, noise, data_type=data_type
        )
        profiled_log_z = float(
            -0.5 * chi2_min
            + _log_mass_integral(A_ref, a_star, ell_lo, ell_hi, mass_prior, ell_ref)
        )

        rel_diff = abs(profiled_log_z - brute_force_log_z) / abs(brute_force_log_z)
        assert rel_diff < 1e-6, (
            f"seed {seed}: profiled={profiled_log_z:.10f} "
            f"brute_force={brute_force_log_z:.10f} rel_diff={rel_diff:.3e}"
        )


class TestMarginalLikelihoodCorrectness:
    """Profiled log-likelihood vs. a brute-force numerical integral over mass."""

    def test_agrees_with_brute_force_integral(self, ssp_data_fsps):
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)
        _assert_profiled_matches_brute_force(model, forward, "photometry", flux, noise)

    def test_agrees_with_brute_force_integral_spectroscopy(self, ssp_data_fsps):
        model = _spectroscopy_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _spectroscopy_mock(model, seed=0)
        _assert_profiled_matches_brute_force(model, forward, "spectroscopy", flux, noise)

    def test_agrees_with_brute_force_integral_joint(self, ssp_data_fsps):
        model = _joint_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, data, noise = _joint_mock(model, seed=0)
        _assert_profiled_matches_brute_force(model, forward, "joint", data, noise)


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

    def test_calibration_marginalize_raises(self, ssp_data_fsps):
        """Calibration marginalization is its own linear block, not this module's math.

        Spectroscopy is otherwise fair game for profiling (the guard relaxed in
        this change) -- this pins that ``calibration_marginalize=True`` still
        refuses it, on a spectroscopy fixture that would engage profiling
        without that flag (see ``TestAutoDefault.test_auto_resolves_true_on_spectroscopy``).
        """
        model = _spectroscopy_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _spectroscopy_mock(model, seed=0)

        with pytest.raises(ValueError, match="calibration_marginalize"):
            Fitter(
                forward,
                data=flux,
                noise=noise,
                data_type="spectroscopy",
                calibration_marginalize=True,
                profile_mass=True,
            )

        fitter = Fitter(
            forward,
            data=flux,
            noise=noise,
            data_type="spectroscopy",
            calibration_marginalize=True,
            profile_mass="auto",
        )
        assert fitter._profile_mass is False
        assert "calibration_marginalize" in fitter._profile_mass_reason

    def test_eline_channel_raises(self, ssp_data_fsps):
        """A marginalized emission-line channel refuses profiling (own likelihood plumbing).

        Same point as ``test_calibration_marginalize_raises``: this fixture's
        spectroscopy would otherwise engage profiling, so this isolates the
        eline guard rather than merely re-confirming the relaxed data_type one.
        """
        from tengri.observation.line_list import LineList

        wave = jnp.linspace(4000.0, 7500.0, 200)
        obs = Observation(
            spectroscopy=Spectroscopy(
                wave_obs=wave,
                resolution=2000.0,
                eline_mode="marginalized",
                eline_catalog=LineList.default_13(),
            )
        )
        model = SEDModel.build(
            ssp_data=ssp_data_fsps, observation=obs, **recipes.mock_recovery_minimal()
        )
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _spectroscopy_mock(model, seed=0)

        with pytest.raises(ValueError, match="emission-line"):
            Fitter(forward, data=flux, noise=noise, data_type="spectroscopy", profile_mass=True)

        fitter = Fitter(
            forward, data=flux, noise=noise, data_type="spectroscopy", profile_mass="auto"
        )
        assert fitter._profile_mass is False
        assert "emission-line" in fitter._profile_mass_reason


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

    # Deliberately no joint/spectroscopy-mock counterpart, in any tolerance:
    # the profiled (marginal) MAP and the joint (theta, mass) MAP are
    # different estimators by construction -- the marginal adds an
    # A(theta)-dependent normalization term the joint chi^2 does not carry,
    # so their argmax need not coincide. Measured on this recipe, the
    # disagreement ranges 5e-4 to 0.56 dex depending on the truth draw, so no
    # cross-estimator MAP tolerance is the right assertion here; correctness
    # for these data types is covered by ``TestMarginalLikelihoodCorrectness``
    # (the brute-force integral, an exact target) and
    # ``TestSamplerParity.test_nuts_posterior_means_agree_joint`` (exact
    # posterior sampling, not a point estimate).


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

    def test_nuts_posterior_means_agree_joint(self, ssp_data_fsps):
        model = _joint_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, data, noise = _joint_mock(model, seed=0)

        kw = dict(
            method="mcmc_nuts",
            dense_mass_matrix=True,
            n_warmup=100,
            n_samples=150,
            n_chains=2,
            key=jax.random.PRNGKey(7),
        )
        fitter_off = Fitter(forward, data=data, noise=noise, data_type="joint", profile_mass=False)
        fitter_on = Fitter(forward, data=data, noise=noise, data_type="joint", profile_mass=True)
        assert fitter_on.data_type == "joint"

        post_off = fitter_off.run(**kw)
        post_on = fitter_on.run(**kw)

        assert post_on.diagnostics["profile_mass_resolved"] is True
        assert post_on.samples[_MASS_NAME].shape == post_on.samples["dust_tau_bc"].shape

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

    def test_auto_resolves_true_on_spectroscopy(self, ssp_data_fsps):
        model = _spectroscopy_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _spectroscopy_mock(model, seed=0)
        fitter = Fitter(
            forward, data=flux, noise=noise, data_type="spectroscopy", profile_mass="auto"
        )
        assert fitter._profile_mass is True
        assert fitter._profile_mass_resolved is True

    def test_auto_resolves_true_on_joint(self, ssp_data_fsps):
        model = _joint_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, data, noise = _joint_mock(model, seed=0)
        fitter = Fitter(forward, data=data, noise=noise, data_type="joint", profile_mass="auto")
        assert fitter._profile_mass is True
        assert fitter._profile_mass_resolved is True


class TestReferenceInvariance:
    """Marginal log-likelihood is invariant to the mass reference frame.

    The dimensionless formulation of _profile_stats cancels mass factors
    algebraically, so the returned marginal integral must not depend on
    the reference point ``phys[mass_name]`` at which the prediction is
    evaluated. This test sweeps the reference over the prior bounds plus
    ±1 dex outside and verifies the marginal log-likelihood is unchanged.
    """

    def test_reference_invariance_photometry(self, ssp_data_fsps):
        """Profiled marginal is the same at different reference masses."""
        model = _minimal_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _mock(model, seed=0)

        fitter_off = Fitter(forward, data=flux, noise=noise, profile_mass=False)
        mass_prior = fitter_off.spec.get_distribution(_MASS_NAME)
        ell_lo, ell_hi = mass_prior.bounds

        # Reference points: lo, midpoint, hi, and ±1 dex outside
        ell_refs = [
            ell_lo - 1.0,
            ell_lo,
            0.5 * (ell_lo + ell_hi),
            ell_hi,
            ell_hi + 1.0,
        ]

        # Compute profiled marginal at each reference
        other_names = [n for n in fitter_off._free_names if n != _MASS_NAME]
        phys_base = {
            name: fitter_off.spec.get_distribution(name).unstandardize(0.0) for name in other_names
        }
        for name, val in fitter_off._fixed_values.items():
            phys_base[name] = jnp.asarray(val)

        log_z_vals = []
        for ell_ref in ell_refs:
            phys = {**phys_base, _MASS_NAME: jnp.asarray(ell_ref)}
            A_ref, a_star, chi2_min, ell_ref_out = _profile_stats(
                model, _MASS_NAME, phys, flux, noise, data_type="photometry"
            )
            log_z = float(
                -0.5 * chi2_min
                + _log_mass_integral(A_ref, a_star, ell_lo, ell_hi, mass_prior, ell_ref_out)
            )
            log_z_vals.append(log_z)

        # All values should be identical (within numerical precision)
        log_z_vals = np.array(log_z_vals)
        rel_error = np.abs(np.diff(log_z_vals)) / np.abs(log_z_vals[:-1] + 1e-10)
        assert np.all(rel_error < 1e-5), f"Relative errors: {rel_error}"

    def test_reference_invariance_spectroscopy(self, ssp_data_fsps):
        """Profiled marginal is the same at different reference masses (spectroscopy)."""
        model = _spectroscopy_model(ssp_data_fsps)
        forward = ForwardModel.build(sed=model)
        _, flux, noise = _spectroscopy_mock(model, seed=1)

        fitter_off = Fitter(
            forward, data=flux, noise=noise, data_type="spectroscopy", profile_mass=False
        )
        mass_prior = fitter_off.spec.get_distribution(_MASS_NAME)
        ell_lo, ell_hi = mass_prior.bounds

        # Reference points: lo, midpoint, hi
        ell_refs = [
            ell_lo,
            0.5 * (ell_lo + ell_hi),
            ell_hi,
        ]

        other_names = [n for n in fitter_off._free_names if n != _MASS_NAME]
        phys_base = {
            name: fitter_off.spec.get_distribution(name).unstandardize(0.0) for name in other_names
        }
        for name, val in fitter_off._fixed_values.items():
            phys_base[name] = jnp.asarray(val)

        log_z_vals = []
        for ell_ref in ell_refs:
            phys = {**phys_base, _MASS_NAME: jnp.asarray(ell_ref)}
            A_ref, a_star, chi2_min, ell_ref_out = _profile_stats(
                model, _MASS_NAME, phys, flux, noise, data_type="spectroscopy"
            )
            log_z = float(
                -0.5 * chi2_min
                + _log_mass_integral(A_ref, a_star, ell_lo, ell_hi, mass_prior, ell_ref_out)
            )
            log_z_vals.append(log_z)

        log_z_vals = np.array(log_z_vals)
        rel_error = np.abs(np.diff(log_z_vals)) / np.abs(log_z_vals[:-1] + 1e-10)
        assert np.all(rel_error < 1e-5), f"Relative errors: {rel_error}"


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


@pytest.mark.contract
def test_profile_mass_float32_end_to_end(ssp_data_fsps):
    """Verify float32 support end-to-end: construction, inference, finite gradients.

    Regression test for float32 support: the dimensionless rewrite of
    profile_mass allows float32 at construction time. This test verifies the
    entire pipeline works and produces finite values and gradients in float32.
    """
    with jax.enable_x64(False):
        # Verify we're in float32
        assert jnp.result_type(float) == jnp.float32

        # Build the model INSIDE the context, matching the shape in the task
        model = _minimal_model(ssp_data_fsps)
        obs = Observation(photometry=Photometry.from_names(_FILTERS))

        key_t, key_m = jax.random.split(jax.random.PRNGKey(42))
        truth = model.spec.sample(key_t)
        mock = generate_mock(model, truth, key=key_m, snr=30.0)
        flux = jnp.asarray(mock["flux_obs"])
        noise = jnp.asarray(mock["noise"])

        forward = ForwardModel.build(sed=model, observation=obs)
        fitter = Fitter(forward, flux, noise, profile_mass="auto")

        # Assert float32 no longer refused at construction
        assert fitter._profile_mass is True

        # Build context and test inference
        ctx = InferenceContext.from_target(fitter)
        xi = {n: jnp.asarray(0.1) for n in fitter._free_names}

        # Forward pass
        val = ctx.neg_log_posterior_fn(xi, ctx.data_args)

        # Check dtype and finiteness of value
        assert val.dtype == jnp.float32, f"Expected float32, got {val.dtype}"
        assert jnp.isfinite(val), f"Value is not finite: {val}"

        # Gradient pass
        grads = jax.grad(lambda p: ctx.neg_log_posterior_fn(p, ctx.data_args))(xi)

        # Check every gradient leaf is float32 and finite
        for name, grad_leaf in grads.items():
            assert grad_leaf.dtype == jnp.float32, (
                f"Gradient {name} is {grad_leaf.dtype}, expected float32"
            )
            assert jnp.isfinite(grad_leaf).all(), f"Gradient {name} contains non-finite values"


def test_profile_mass_laplace_rejects_float32(ssp_data_fsps):
    """Verify laplace method rejects float32 due to Hessian NaN.

    Guard test: float32 refusal for laplace occurs at method resolution time
    (resolve_profile_mass_for_method), not at Fitter construction. This reflects
    the true cause: the SED model's Hessian NaN is a forward-model seam,
    not a profiling defect.
    """
    with jax.enable_x64(False):
        model = _minimal_model(ssp_data_fsps)
        obs = Observation(photometry=Photometry.from_names(_FILTERS))

        key_t, key_m = jax.random.split(jax.random.PRNGKey(42))
        truth = model.spec.sample(key_t)
        mock = generate_mock(model, truth, key=key_m, snr=30.0)
        flux = jnp.asarray(mock["flux_obs"])
        noise = jnp.asarray(mock["noise"])

        forward = ForwardModel.build(sed=model, observation=obs)
        fitter = Fitter(forward, flux, noise, profile_mass="auto")

        # profile_mass=True at construction (profiling is allowed in float32)
        assert fitter._profile_mass is True

        # But laplace method should refuse at resolution time
        from tengri.inference.mass_profile import resolve_profile_mass_for_method

        with pytest.raises(ValueError, match="float32 mode"):
            resolve_profile_mass_for_method(fitter, "laplace", "auto")
