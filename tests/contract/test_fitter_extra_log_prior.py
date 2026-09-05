# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for the ``Fitter(..., extra_log_prior=...)`` hook.

Surface protected: ``Fitter.__init__``'s ``extra_log_prior`` parameter and
its wiring into ``tengri.inference.loss_functions.build_logprior_fn`` (the
task-7/D7 fix: the informative AGN priors in ``tengri.agn.priors`` were
"unreachable from the public API" -- this is the fitter-side half of making
them reachable). The physics-correctness of any specific prior (e.g.
``prior_energy_balance``) is covered in
``tests/crossval/test_agn_priors_vs_agnfitter.py`` and
``tests/contract/test_agn_priors.py``; this file only checks that a supplied
``extra_log_prior`` callable is actually invoked, with the documented
``(params, state)`` signature, and its return value folded into the
log-prior by exact addition -- not silently dropped.

Uses the session-scoped ``synthetic_ssp`` fixture (no ``data/ssp_*.h5``
needed, #613) so this runs in the default fast tier.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_logprior_fn
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


_PHOT = Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))


def _build_model(ssp):
    obs = Observation(photometry=_PHOT)
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8, 12)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    ), obs


def _extra_term(params, state):
    """A deliberately simple, non-physical extra term.

    Uses BOTH ``params`` (a free parameter) and ``state`` (a core
    ``ForwardState`` field guaranteed present regardless of which components
    are configured) so the test exercises the full documented contract
    without depending on any specific component's ``state.derived`` keys.
    """
    log_total_mass = params["sfh_dpl_log_total_mass"]
    return -0.5 * (log_total_mass - 10.0) ** 2 - 1e-6 * jnp.sum(state.sed_intrinsic**2)


@pytest.fixture(scope="module")
def _fitters(synthetic_ssp):
    model_a, _obs = _build_model(synthetic_ssp)
    model_b, _ = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    plain = Fitter(model_a, data, noise, data_type="photometry")
    hooked = Fitter(model_b, data, noise, data_type="photometry", extra_log_prior=_extra_term)
    return plain, hooked


def test_extra_log_prior_changes_logprior_by_exactly_the_extra_term(_fitters):
    plain, hooked = _fitters
    free_name = next(iter(plain._free_names))
    assert free_name == "sfh_dpl_log_total_mass"
    free_params = {free_name: 9.3}

    lp_plain = float(build_logprior_fn(plain)(free_params))
    lp_hooked = float(build_logprior_fn(hooked)(free_params))

    # Independently reconstruct what the extra term SHOULD be: merge fixed
    # values exactly as build_logprior_fn's hook path does, predict_state,
    # and evaluate _extra_term directly.
    params = dict(free_params)
    for name, val in hooked._fixed_values.items():
        params[name] = val
    params = hooked.spec.resolve_mirrors(params)
    state = hooked.model.predict_state(params)
    expected_extra = float(_extra_term(params, state))

    assert (lp_hooked - lp_plain) == pytest.approx(expected_extra, rel=1e-9, abs=1e-9)
    # And the extra term must be genuinely nonzero for this input, otherwise
    # the assertion above would pass vacuously even if the hook were ignored.
    assert abs(expected_extra) > 1e-6


def test_extra_log_prior_none_is_bit_identical_to_no_hook(synthetic_ssp):
    """Default ``extra_log_prior=None`` must not perturb the existing log-prior."""
    model_a, _obs = _build_model(synthetic_ssp)
    model_b, _ = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    plain = Fitter(model_a, data, noise, data_type="photometry")
    explicit_none = Fitter(model_b, data, noise, data_type="photometry", extra_log_prior=None)
    free_params = {"sfh_dpl_log_total_mass": 9.3}
    lp_plain = float(build_logprior_fn(plain)(free_params))
    lp_explicit_none = float(build_logprior_fn(explicit_none)(free_params))
    assert lp_plain == lp_explicit_none


def test_hooked_logprior_fn_is_jit_compiled(_fitters):
    _, hooked = _fitters
    logprior_fn = build_logprior_fn(hooked)
    free_params = {"sfh_dpl_log_total_mass": 9.3}
    eager = float(logprior_fn(free_params))
    jitted = float(jax.jit(logprior_fn)(free_params))
    assert jitted == pytest.approx(eager, rel=1e-9, abs=1e-9)


def test_non_callable_extra_log_prior_raises_type_error(synthetic_ssp):
    model, _obs = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    with pytest.raises(TypeError, match="extra_log_prior must be callable"):
        Fitter(model, data, noise, data_type="photometry", extra_log_prior="not callable")
