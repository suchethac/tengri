# SPDX-License-Identifier: BSD-3-Clause
"""Contract for #2296: params dicts carry free parameters only.

A ``params`` dict key the spec declared ``Fixed`` is refused, loudly, at
every entry point that takes a params dict: :class:`ParameterError` naming
the key, the pinned value, and the remedy (rebuild with ``FREE`` or a
different ``Fixed`` value). One rule on every path; no path honors the
override, no path silently ignores it.

``Fitter(params_override=...)`` (and ``CatalogFitter``'s per-galaxy redshift
override) is a different, still-sanctioned channel: a validated
construction-time re-pin, checked to name only Fixed parameters, not a
``params`` dict handed to a predict surface. It is not covered here; see
``tests/inference/test_fitter.py`` / ``tests/inference/test_catalog_mcmc_vmap.py``
for its own coverage.

https://github.com/suchethac/tengri/issues/2296
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.config.exceptions import ParameterError
from tengri.inference.catalog import Catalog

pytestmark = pytest.mark.contract

REMEDY_PHRASE = "rebuild the model"


def _assert_is_a_fixed_key_refusal(exc, key, pinned_value):
    """The message names the key, the pinned value, and the remedy."""
    msg = str(exc)
    assert key in msg, f"error should name the Fixed key {key!r} (got: {msg})"
    assert f"{pinned_value}" in msg or f"{float(pinned_value)}" in msg, (
        f"error should contain the pinned value {pinned_value!r} (got: {msg})"
    )
    assert REMEDY_PHRASE in msg, f"error should state the remedy (got: {msg})"


@pytest.fixture(scope="module")
def model_with_fixed_redshift(synthetic_ssp_wide, synthetic_tophat_obs):
    """DPL SFH + Cue nebular (a photoionized backend, so predict_line_fluxes
    has something to predict) + Fixed redshift, so a params dict legitimately
    omits it.

    ``synthetic_ssp_wide`` is unphysical by construction (young-bin log Q_H
    far above the physical band for a bare-stellar population), which trips
    CueBackend's wNE sanity check; ``tests/conftest.py`` sets
    ``TENGRI_ALLOW_WNE_CUE=1`` suite-wide to downgrade that to a warning for
    exactly this kind of structural test.
    """
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "alpha": FREE, "beta": FREE},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )


@pytest.fixture
def free_params_dict(model_with_fixed_redshift):
    """Build a dict with only free parameters."""
    return {name: 0.5 for name in model_with_fixed_redshift.spec.free_params}


@pytest.fixture
def params_with_fixed_override(free_params_dict):
    """Build a dict with a Fixed key override (the pinned value itself --
    the rule is on key presence, not value comparison)."""
    return {**free_params_dict, "redshift": 0.05}


# ── Direct predict_* surfaces (no deprecation warning) ──────────────────

DIRECT_METHODS = [
    "predict",
    "predict_photometry",
    "predict_properties",
    "predict_line_fluxes",
    "predict_state",
    "predict_observables",
    "predict_observables_jit",
]


def _call(model, method_name, params):
    method = getattr(model, method_name)
    if method_name == "predict_properties":
        return method(params, names=["stellar_mass"])
    return method(params)


@pytest.mark.parametrize("method_name", DIRECT_METHODS)
def test_fixed_key_refused_by_entry_points(
    model_with_fixed_redshift, free_params_dict, params_with_fixed_override, method_name
):
    """Each entry point refuses Fixed keys in params; free-only params work."""
    model = model_with_fixed_redshift

    result = _call(model, method_name, free_params_dict)
    assert result is not None, f"{method_name} with free-only params should succeed"

    with pytest.raises(ParameterError) as exc_info:
        _call(model, method_name, params_with_fixed_override)

    _assert_is_a_fixed_key_refusal(exc_info.value, "redshift", 0.05)


# ── Deprecated aliases: same refusal, plus their own DeprecationWarning ──

DEPRECATED_ALIASES = [
    "predict_rest_sed",
    "predict_obs_sed",
    "predict_derived",
    "predict_magnitudes",
]


@pytest.mark.parametrize("method_name", DEPRECATED_ALIASES)
def test_deprecated_alias_refuses_fixed_key(
    model_with_fixed_redshift, free_params_dict, params_with_fixed_override, method_name
):
    """Deprecated predict_* aliases still refuse a Fixed key (#2296), under
    their own DeprecationWarning."""
    model = model_with_fixed_redshift
    method = getattr(model, method_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = method(free_params_dict)
        assert result is not None, f"{method_name} with free-only params should succeed"

        with pytest.raises(ParameterError) as exc_info:
            method(params_with_fixed_override)

    _assert_is_a_fixed_key_refusal(exc_info.value, "redshift", 0.05)


def test_catalog_predict_refuses_fixed_keys(
    model_with_fixed_redshift, free_params_dict, params_with_fixed_override
):
    """Catalog.predict refuses Fixed keys in param_table.

    ``table=`` (construction) carries the observed photometry the catalog was
    built from; ``predict(param_table=...)`` is a SEPARATE dict of model
    parameter columns -- the two are not the same table (see
    ``Catalog.predict``'s docstring example).
    """
    model = model_with_fixed_redshift
    n_filters = len(model.observation.photometry.filters)
    flux_cols = [f"flux_{i}" for i in range(n_filters)]
    err_cols = [f"flux_{i}_err" for i in range(n_filters)]

    data_table = {col: np.full((10,), 1.0) for col in flux_cols + err_cols}
    cat = Catalog(
        model,
        data_table,
        flux_unit="cgs_fnu",
        flux_cols=flux_cols,
        err_cols=err_cols,
    )

    free_table = {name: np.full((10,), val) for name, val in free_params_dict.items()}
    result = cat.predict(free_table)
    assert result is not None
    assert result.shape == (10, n_filters)

    fixed_table = {name: np.full((10,), val) for name, val in params_with_fixed_override.items()}
    with pytest.raises(ParameterError) as exc_info:
        cat.predict(fixed_table)

    _assert_is_a_fixed_key_refusal(exc_info.value, "redshift", 0.05)


def test_fixed_key_omitted_works(model_with_fixed_redshift, free_params_dict):
    """Params dict without the Fixed key works fine."""
    model = model_with_fixed_redshift
    output = model.predict(free_params_dict)
    assert output is not None


def test_spec_sample_returns_only_free_keys(model_with_fixed_redshift):
    """spec.sample() returns only free parameters."""
    model = model_with_fixed_redshift

    sampled = model.spec.sample(jr.PRNGKey(0))

    for key in sampled:
        assert key in model.spec.free_params, f"Sampled key {key} is not free"
    for free_key in model.spec.free_params:
        assert free_key in sampled, f"Free param {free_key} missing from sample"

    output = model.predict(sampled)
    assert output is not None


def test_posterior_round_trip_is_free_only(model_with_fixed_redshift, free_params_dict):
    """A fitted Posterior's params feed straight back into predict(), and its
    fixed_values match the spec exactly (#2296)."""
    model = model_with_fixed_redshift
    truth = dict(free_params_dict)
    mock = model.mock(truth, snr=10.0, key=jr.PRNGKey(0))

    posterior = model.fit(
        mock.flux_obs, mock.noise, method="map", n_steps=30, verbose=False, key=jr.PRNGKey(1)
    )

    for key in posterior.params:
        assert key in model.spec.free_params, (
            f"Posterior.params carries {key!r}, which is not free -- params dicts "
            "are free-only (#2296)"
        )

    assert posterior.fixed_values == model.spec.get_fixed_values()

    result = model.predict(dict(posterior.params))
    assert result is not None


def test_fixed_key_in_params_override_channel_still_works(
    model_with_fixed_redshift, free_params_dict
):
    """Fitter(params_override=...) is the sanctioned way to re-pin a Fixed
    value for one fit; that channel is untouched by the params-dict refusal.
    """
    from tengri.inference.fitter import Fitter

    model = model_with_fixed_redshift
    truth = dict(free_params_dict)
    mock = model.mock(truth, snr=10.0, key=jr.PRNGKey(2))

    fitter = Fitter(
        model,
        mock.flux_obs,
        mock.noise,
        params_override={"redshift": 0.2},
    )
    result = fitter.run("map", n_steps=30, verbose=False, key=jr.PRNGKey(3))
    assert result is not None
    assert result.fixed_values["redshift"] == pytest.approx(0.2)


def test_fitter_params_override_refuses_a_free_key(model_with_fixed_redshift):
    """Fitter(params_override=...) is a re-pin channel for FIXED parameters
    only. Naming a FREE parameter would corrupt inference (silently taking it
    out of the fit while the returned params still claim it was sampled), so
    construction refuses it -- the mirror image of the params-dict refusal,
    checked at a different seam (construction-time key validation against
    ``self._free_names``, not ``refuse_fixed_overrides``).
    """
    from tengri.inference.fitter import Fitter

    model = model_with_fixed_redshift
    free_name = next(iter(model.spec.free_params))

    with pytest.raises(ValueError, match=free_name):
        Fitter(
            model,
            np.ones(len(model.observation.photometry.filters)),
            np.ones(len(model.observation.photometry.filters)),
            params_override={free_name: 0.5},
        )


# ── Batch and fast-nebular surfaces ──────────────────────────────────────


def test_predict_photometry_batch_refuses_fixed_keys(
    model_with_fixed_redshift, free_params_dict, params_with_fixed_override
):
    """``predict_photometry_batch`` (vmap over ``predict_photometry``) refuses
    a Fixed key the same way the scalar surface does -- the key-set check is a
    static Python-level membership test, safe (and cheap) even though the call
    it guards runs under ``jax.vmap`` (#2296)."""
    model = model_with_fixed_redshift
    n = 4
    free_batch = {name: jnp.full((n,), val) for name, val in free_params_dict.items()}
    result = model.predict_photometry_batch(free_batch)
    n_filters = len(model.observation.photometry.filters)
    assert result.shape == (n, n_filters)

    fixed_batch = {name: jnp.full((n,), val) for name, val in params_with_fixed_override.items()}
    with pytest.raises(ParameterError) as exc_info:
        model.predict_photometry_batch(fixed_batch)

    _assert_is_a_fixed_key_refusal(exc_info.value, "redshift", 0.05)


@pytest.fixture
def fast_nebular_model(synthetic_ssp_wide, synthetic_tophat_obs):
    """Same recipe as ``model_with_fixed_redshift``, but with the per-Q_H fast
    line grid enabled (#950), so ``predict_line_fluxes`` takes the FAST
    (grid-reconstruction) branch instead of the exact ``predict_state`` one.
    Function-scoped and built fresh (not reusing the module-scoped fixture)
    because ``enable_fast_nebular`` mutates the model in place.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "alpha": FREE, "beta": FREE},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )
    target_wavelengths = np.asarray([6564.61, 4862.68])  # Halpha, Hbeta (vacuum)
    model.enable_fast_nebular(target_wavelengths, n_grid=8)
    return model


def test_predict_line_fluxes_fast_branch_refuses_fixed_keys(fast_nebular_model):
    """``predict_line_fluxes``' FAST grid-reconstruction branch (#950) merges
    Fixed values via ``merge_fixed_params`` exactly like the exact branch
    (sed_model.py, the ``if grid is not None:`` arm) -- verify the fast branch
    specifically, not just whichever branch the other parametrized tests
    happen to exercise."""
    model = fast_nebular_model
    free = {name: 0.5 for name in model.spec.free_params}

    result = model.predict_line_fluxes(free)
    assert result is not None

    with pytest.raises(ParameterError) as exc_info:
        model.predict_line_fluxes({**free, "redshift": 0.05})

    _assert_is_a_fixed_key_refusal(exc_info.value, "redshift", 0.05)


def test_model_mock_refuses_fixed_keys(
    model_with_fixed_redshift, free_params_dict, params_with_fixed_override
):
    """``model.mock`` (mock photometric observation generation) refuses a
    Fixed key the same way every other params-dict entry point does (#2296):
    it is auto-discovered by ``test_fixed_params_reach_every_entry_point.py``'s
    sweep too (its second positional arg is named ``params``), but that sweep
    only ever checks the ``redshift`` axis against one model; pin it here
    explicitly alongside the rest of this file's surfaces."""
    model = model_with_fixed_redshift

    mock = model.mock(free_params_dict, snr=10.0, key=jr.PRNGKey(4))
    assert mock is not None

    with pytest.raises(ParameterError) as exc_info:
        model.mock(params_with_fixed_override, snr=10.0, key=jr.PRNGKey(4))

    _assert_is_a_fixed_key_refusal(exc_info.value, "redshift", 0.05)


@pytest.fixture(scope="module")
def profile_mass_model(synthetic_ssp_wide, synthetic_tophat_obs):
    """A model shaped for profile_mass to engage: exactly one free
    ``*log_total_mass``, a second free shape parameter, and dust/redshift
    Fixed so the mass enters the photometry linearly (transparent dust)."""
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": FREE,
            "alpha": FREE,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def test_posterior_fixed_values_excludes_the_profiled_mass(profile_mass_model):
    """Under profile_mass, fixed_values must reflect the USER's model, not
    the internal working spec (#2296).

    ``configure_profile_mass`` pins the mass parameter Fixed on the fitter's
    WORKING spec at an analytic placeholder so the sampler explores one
    fewer dimension; ``finalize_profile_mass`` then writes the real,
    analytically profiled mass into ``posterior.params``. The mass is FREE
    on ``model.spec`` throughout -- profiling is an internal re-pin, never a
    user Fixed override -- so it must never appear in
    ``posterior.fixed_values`` (which would otherwise report the stale
    placeholder for a parameter ``posterior.params`` already carries the
    real value of).
    """
    model = profile_mass_model
    mass_name = next(n for n in model.spec.free_params if n.endswith("log_total_mass"))
    truth = {mass_name: 10.0, "sfh_dpl_alpha": 1.5}
    mock = model.mock(truth, snr=20.0, key=jr.PRNGKey(5))

    posterior = model.fit(
        mock.flux_obs,
        mock.noise,
        method="map",
        profile_mass=True,
        n_steps=40,
        verbose=False,
        key=jr.PRNGKey(6),
    )

    assert posterior.diagnostics["profile_mass"] is True, "profile_mass must have engaged"
    assert mass_name in posterior.params, "the real profiled mass belongs in params"
    assert mass_name not in posterior.fixed_values, (
        f"{mass_name!r} is free on model.spec; fixed_values must not report "
        "the internal profile_mass placeholder for it"
    )
    # A genuine Fixed parameter is unaffected by the filter.
    assert posterior.fixed_values["redshift"] == pytest.approx(0.05)
