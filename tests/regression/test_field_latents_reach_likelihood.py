# SPDX-License-Identifier: BSD-3-Clause
"""The GP field latents must reach the likelihood, not just the predict path.

Regression for a silent-failure bug: the sampler keys the stochastic-SFH latents
``psd_xi``, but :class:`StellarSEDComponent` reads ``sfh_field_xi`` and falls back
to ``jnp.zeros(n_grid)`` when that key is absent
(``components/stellar/component.py``). ``_unstandardize_parameters`` published only
``psd_xi``, so every non-hierarchical field fit ran with the GP field pinned to
zero -- ``exp(0 - K0/2)`` is a constant, so the burstiness degrees of freedom were
sampled from their prior and never touched the SED.

It failed open: no exception, no warning, a physically plausible SED, and a
posterior whose SFH bands looked wide-and-covering because ``predict_sfh`` reads
the latents through a *different* path (``get_internal_params``, which accepts
both spellings). The only visible signature was the likelihood gradient w.r.t.
the latents being exactly zero.

The existing field tests all drive ``predict_*`` with ``sfh_field_xi`` directly,
so none of them could see it. This one drives the inference path.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri import (
    FREE,
    Fitter,
    Fixed,
    ForwardModel,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    builders,
    load_ssp_data,
)

_SSP = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)


def _model(ssp_data, observation):
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={"type": ["dpl", "field"], "*": FREE},
        met={"logzsol": Fixed(-0.3)},
        dust=builders.dust.two_component(defaults=FREE, law_bc="calzetti"),
        neb=builders.neb.ssp(),
        redshift=Fixed(0.1),
        apply_igm=False,
        n_grid=16,
        approx=None,
    )


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
def test_field_latents_receive_likelihood_gradient():
    """d(neg_log_posterior)/d(psd_xi) must have a nonzero LIKELIHOOD component.

    Measured by subtracting the prior-only gradient (obtained by inflating the
    data errors) rather than by eyeballing the total, because the latents carry
    an N(0, I) prior whose gradient is large enough to hide a dead likelihood.
    """
    from tengri.inference.context import InferenceContext

    ssp = load_ssp_data(str(_SSP))
    phot = Photometry.from_names(["galex_fuv", "galex_nuv", "sdss_g", "sdss_r", "2mass_ks"])
    noise_model = NoiseModel(calibration_floor=0.01, student_t_dof=None)
    observation = Observation(photometry=phot, noise=noise_model)
    model = _model(ssp, observation)

    params = {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(0))}
    mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(1))
    flux, err = np.asarray(mock.flux_obs), np.asarray(mock.noise)

    def grad_psd_xi(noise_scale):
        forward = ForwardModel.build(sed=_model(ssp, observation), observation=observation)
        fitter = Fitter(forward, flux, err * noise_scale, approx=None)
        ctx = InferenceContext.from_target(fitter)
        x0 = ctx.initial_params(jax.random.PRNGKey(2))
        _, grad = ctx.grad_fn(x0, ctx.data_args)
        return np.asarray(grad["psd_xi"])

    g_full = grad_psd_xi(1.0)
    g_prior = grad_psd_xi(1e8)  # likelihood switched off, prior untouched
    g_like = float(np.linalg.norm(g_full - g_prior))

    assert np.all(np.isfinite(g_full)), "field-latent gradient must be finite"
    assert g_like > 1e-6, (
        "likelihood gradient w.r.t. psd_xi is zero: the GP field latents are not "
        "reaching the forward model, so burstiness is pinned at its prior "
        f"(|g_like| = {g_like:.3e})"
    )


def test_unstandardize_publishes_both_field_spellings():
    """``_unstandardize_parameters`` must publish the latents under both names.

    ``psd_xi`` is what the sampler and the prior term use; ``sfh_field_xi`` is what
    the stellar component reads. Publishing only one silently zeroes the field.
    """
    from tengri.inference.loss_functions import _unstandardize_parameters

    class _Spec:
        stochastic = True

        def get_distribution(self, name):  # pragma: no cover - not reached
            raise AssertionError("no free scalars in this stub")

        def resolve_mirrors(self, params):
            return params

    xi = jnp.arange(4.0)
    out = _unstandardize_parameters(
        {"psd_xi": xi}, _Spec(), free_names=(), fixed_values={}, stochastic=True
    )
    assert "psd_xi" in out, "sampler/prior key must be published"
    assert "sfh_field_xi" in out, (
        "StellarSEDComponent reads sfh_field_xi and defaults to zeros when it is "
        "missing — publishing only psd_xi pins the GP field to zero"
    )
    np.testing.assert_array_equal(np.asarray(out["psd_xi"]), np.asarray(out["sfh_field_xi"]))


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
def test_posterior_params_evaluate_to_the_fitted_model():
    """``model.predict_*(posterior.params)`` must reproduce the fit, not the smooth model.

    The user-facing half of the same bug: ``Fitter._to_physical`` published the
    GP latents only as ``psd_xi``, so every prediction made from a returned
    ``Posterior`` hit the ``sfh_field_xi`` lookup, got zeros, and silently
    evaluated the SMOOTH SFH. Measured on one realization, a MAP fit whose true
    photometric chi2/N was 0.34 read back as 9.00 -- good enough to look like an
    ordinary bad fit, which is why it survived.

    The invariant asserted here is public and cheap: predicting from the
    returned posterior must differ from predicting with the field switched off.
    """
    ssp = load_ssp_data(str(_SSP))
    phot = Photometry.from_names(["galex_fuv", "galex_nuv", "sdss_g", "sdss_r", "2mass_ks"])
    observation = Observation(
        photometry=phot, noise=NoiseModel(calibration_floor=0.01, student_t_dof=None)
    )
    model = _model(ssp, observation)

    params = {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(0))}
    mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(1))
    flux, err = np.asarray(mock.flux_obs), np.asarray(mock.noise)

    forward = ForwardModel.build(sed=_model(ssp, observation), observation=observation)
    res = forward.fit(
        flux,
        err,
        method="map",
        approx=None,
        n_steps=200,
        n_restarts=1,
        key=jax.random.PRNGKey(2),
        verbose=False,
    )

    assert "sfh_field_xi" in res.params, (
        "Posterior.params must carry the spelling the forward model reads; "
        "publishing only psd_xi makes every predict_* call score the smooth model"
    )
    xi = np.asarray(res.params["psd_xi"])
    assert np.linalg.norm(xi) > 1e-8, "fit returned a degenerate all-zero field; test is vacuous"

    fixed = model.spec.get_fixed_values()
    with_field = np.asarray(model.predict_photometry({**fixed, **res.params}))
    off = {**fixed, **res.params}
    off["sfh_field_xi"] = jnp.zeros_like(jnp.asarray(xi))
    off["psd_xi"] = jnp.zeros_like(jnp.asarray(xi))
    without_field = np.asarray(model.predict_photometry(off))

    assert np.all(np.isfinite(with_field))
    rel = float(np.max(np.abs(with_field - without_field) / np.abs(without_field)))
    assert rel > 1e-6, (
        "predicting from Posterior.params gives the SAME fluxes as predicting with the "
        f"GP field zeroed (max rel. diff {rel:.2e}): the returned latents are not reaching "
        "the forward model"
    )


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
def test_predict_sfh_native_grid_is_reachable_and_unresampled():
    """``grid="native"`` must expose the SFH on the model's own log-age nodes.

    Without it there is no public way to score an SFH residual on the grid the
    model is parameterized on: ``predict_sfh`` only returned a uniform
    LINEAR-time resampling whose step is ``age_max / n_linear`` (13.8 Myr at the
    defaults). Five of the sixteen log-age nodes lie below 15 Myr, and the linear
    grid puts 2 of 1000 samples there — so a residual scored on it weights the
    young bins at ~5% instead of ~50%.

    That is not academic: it turned a real +54% improvement in recovery from
    adding emission-line fluxes into an apparent 0%, because nearly all of the
    line information sits below 15 Myr. Scoring code must be able to ask for the
    native grid, and it must be the unresampled values.
    """
    ssp = load_ssp_data(str(_SSP))
    phot = Photometry.from_names(["galex_fuv", "sdss_g", "sdss_r", "2mass_ks"])
    observation = Observation(
        photometry=phot, noise=NoiseModel(calibration_floor=0.01, student_t_dof=None)
    )
    model = _model(ssp, observation)
    params = {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(5))}

    native = model.predict_sfh(params, grid="native")
    linear = model.predict_sfh(params)

    n_grid = int(np.asarray(model.log_age_grid).shape[0])
    assert np.asarray(native["sfr_full"]).shape == (n_grid,)
    np.testing.assert_allclose(
        np.asarray(native["t_gyr"]), 10.0 ** np.asarray(model.log_age_grid) / 1e9, rtol=1e-12
    )

    # The native values must be the model's own, not a round-trip through the
    # lossy linear grid -- that is the entire point of the parameter.
    internal = model._compute_sfr_mean_and_full(model._get_internal_params(params))[1]
    np.testing.assert_allclose(
        np.asarray(native["sfr_full"]),
        np.asarray(internal),
        rtol=1e-12,
        err_msg="grid='native' must return the unresampled internal SFH",
    )

    # The sampling asymmetry that motivated the parameter.
    t_nat, t_lin = np.asarray(native["t_gyr"]), np.asarray(linear["t_gyr"])
    assert (t_nat < 0.015).sum() > (t_lin[t_lin < 0.5] < 0.015).sum(), (
        "the native grid must sample the young ages better than the linear resampling"
    )

    # Default stays backward compatible.
    assert np.asarray(linear["t_gyr"]).shape == (1000,)
    with pytest.raises(ValueError, match="grid must be"):
        model.predict_sfh(params, grid="bogus")


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
def test_sampler_spelling_alone_does_not_reach_the_flux_path():
    """Pin the asymmetry that makes publishing ``sfh_field_xi`` mandatory.

    A dict carrying only the sampler's ``psd_xi`` produces the CORRECT SFH --
    ``_get_internal_params`` accepts both spellings -- but the forward pipeline
    filters params down to declared names, so the latents never reach
    ``_apply_gp_field`` and the photometry is computed from the smooth history.
    Right SFH, wrong flux: strictly worse than being uniformly wrong, because
    the SFH plot looks right while the fluxes silently do not match it.

    This is why the fix lives at the PRODUCERS (``_to_physical``,
    ``_unstandardize_parameters``), which publish both spellings, rather than at
    the consumer. The test documents the asymmetry so that anyone who later
    makes ``psd_xi`` work end-to-end sees this fail and removes it deliberately.
    """
    ssp = load_ssp_data(str(_SSP))
    phot = Photometry.from_names(["galex_fuv", "sdss_g", "sdss_r", "2mass_ks"])
    observation = Observation(
        photometry=phot, noise=NoiseModel(calibration_floor=0.01, student_t_dof=None)
    )
    model = _model(ssp, observation)

    base = {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(5))}
    xi = np.asarray(base["sfh_field_xi"])
    assert np.linalg.norm(xi) > 1e-8, "sampled field is degenerate; test would be vacuous"

    renamed = {k: v for k, v in base.items() if k != "sfh_field_xi"}
    renamed["psd_xi"] = jnp.asarray(xi)
    off = {**base, "sfh_field_xi": jnp.zeros_like(jnp.asarray(xi))}

    sfh_canonical = np.asarray(model.predict_sfh(base)["sfr_full"])
    sfh_renamed = np.asarray(model.predict_sfh(renamed)["sfr_full"])
    np.testing.assert_allclose(
        sfh_canonical,
        sfh_renamed,
        rtol=1e-12,
        err_msg="predict_sfh accepts both spellings via _get_internal_params",
    )

    f_renamed = np.asarray(model.predict_photometry(renamed))
    f_off = np.asarray(model.predict_photometry(off))
    np.testing.assert_allclose(
        f_renamed,
        f_off,
        rtol=1e-12,
        err_msg=(
            "psd_xi alone now reaches the flux path. That is an improvement — "
            "delete this test and the producer-side duplication it justifies."
        ),
    )


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
def test_forward_model_supports_line_flux_fits():
    """A line-flux fit must work through the CANONICAL ForwardModel surface.

    ``loss_functions`` calls ``model._has_line_catalog()`` whenever the
    observation carries line fluxes, to decide between predicting lines (Cue /
    CloudyGrid) and measuring them off the spectrum (wNE / shock).
    ``ForwardModel`` has no ``__getattr__`` fall-through and the method was never
    added to its explicit delegation list, so this raised ``AttributeError`` --
    while the *deprecated* ``Fitter(sed_model, ...)`` path worked fine.

    The recommended API being the broken one is the failure mode worth guarding:
    every existing line-flux test drove the deprecated surface, so nothing caught
    it, and the notebook that exercised this path silenced the deprecation
    warning that would have pointed at the mismatch.
    """
    from tengri.observation import LineFluxData

    ssp = load_ssp_data(str(_SSP))
    phot = Photometry.from_names(["galex_fuv", "sdss_g", "sdss_r", "2mass_ks"])
    names = ("Halpha", "Hbeta", "OIII_5007")
    lines = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in names})
    observation = Observation(
        photometry=phot,
        line_fluxes=lines,
        noise=NoiseModel(calibration_floor=0.01, student_t_dof=None),
    )
    model = _model(ssp, observation)
    forward = ForwardModel.build(sed=model, observation=observation)

    assert forward._has_line_catalog() == model._has_line_catalog(), (
        "ForwardModel must delegate _has_line_catalog to the inner SED"
    )

    params = {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(0))}
    mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(1))
    res = forward.fit(
        np.asarray(mock.flux_obs),
        np.asarray(mock.noise),
        method="map",
        approx=None,
        n_steps=50,
        n_restarts=1,
        key=jax.random.PRNGKey(2),
        verbose=False,
    )
    assert res.params, "line-flux fit through ForwardModel returned no parameters"
