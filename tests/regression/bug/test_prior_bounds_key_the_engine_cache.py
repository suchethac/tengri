# SPDX-License-Identifier: BSD-3-Clause
"""A free parameter's prior must key the inference engine cache.

``_build_signal_response`` deliberately leaves the priors *baked*: it threads
only ``data``/``noise`` through ``data_args`` (for cross-galaxy reuse) and
converts latents with ``dist.unstandardize(xi)``, which reads the
distribution's Python floats at trace time. ``Uniform.unstandardize`` computes
``theta = lo + (hi - lo) * Phi(xi)``, so ``lo``/``hi`` enter the graph as
constants.

Baked is fine — *if* the cache key distinguishes it. It did not. Two models
differing only in a prior's bounds produced an equal ``_engine_cache_key()``,
so the module-level ``_SHARED_ENGINE_CACHE`` and
``_SHARED_SIGNAL_RESPONSE_CACHE`` handed model A's compiled engine to model B,
and B's latent was decoded through A's interval:

    fit Uniform(9.6, 11.1) -> 11.0139   =>  Phi(xi) = 0.9426
    then fit Uniform(7, 13) on the SAME data -> 12.6554
    but 7.0 + 6.0 * 0.9426 = 12.6556    <- A's latent, B's bounds
    B fitted alone                      -> 11.0682

This is the same hazard the ``params_override`` entry of
``_engine_cache_key`` already guards ("without this, fit #2 silently reuses
fit #1's baked override"), applied to prior configuration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fitter, Gaussian, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference.jit_engine import clear_shared_caches, get_or_build_signal_response
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def ssp():
    """Return a minimal SSPData: small enough to fit fast, real enough to compile."""
    n_met, n_age, n_wave = 8, 15, 200
    rng = np.random.default_rng(0)
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.asarray(rng.uniform(0.5, 1.5, (n_met, n_age, n_wave)), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


@pytest.fixture
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))


def _fitter(ssp, obs, prior):
    """Build a Fitter whose ONLY varying ingredient is ``prior``."""
    spec = Parameters(redshift=0.1, sfh_dpl_alpha=prior, sfh_dpl_beta=Uniform(0.3, 3.0))
    model = SEDModel(spec, ssp, observation=obs)
    return Fitter(model, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")


# ── The key must separate priors that bake different constants ──────────────


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        (Uniform(0.5, 4.0), Uniform(0.3, 9.0), "different Uniform bounds"),
        (Uniform(0.5, 4.0), Uniform(0.5, 4.5), "one bound moved"),
        (
            Gaussian(2.0, 0.3, lo=0.5, hi=4.0),
            Gaussian(2.0, 0.9, lo=0.5, hi=4.0),
            "identical bounds, different sigma",
        ),
        (
            Gaussian(2.0, 0.3, lo=0.5, hi=4.0),
            Gaussian(2.5, 0.3, lo=0.5, hi=4.0),
            "identical bounds and sigma, different mu",
        ),
        (
            Uniform(0.5, 4.0),
            Gaussian(2.0, 0.3, lo=0.5, hi=4.0),
            "identical bounds, different family",
        ),
    ],
)
def test_differing_priors_get_distinct_engine_cache_keys(ssp, obs, left, right, why):
    """Two Fitters differing only in one free parameter's prior must not collide.

    ``bounds`` alone is not enough to separate them — the last three cases hold
    ``(lo, hi)`` fixed and vary only ``mu``/``sigma``/family, which
    ``unstandardize`` bakes just as firmly as ``lo``/``hi``.
    """
    a = _fitter(ssp, obs, left)._engine_cache_key()
    b = _fitter(ssp, obs, right)._engine_cache_key()

    assert a != b, f"engine cache key collides across {why}: {a!r}"
    assert hash(a) != hash(b), f"engine cache key hash collides across {why}"


def test_identical_priors_still_share_one_engine_cache_key(ssp, obs):
    """The fix must not over-key: equal priors must still reuse one engine.

    Cross-galaxy reuse is the whole point of the shared cache. Two Fitters with
    equal specs must remain cache-identical, or catalog fits recompile per row.
    """
    a = _fitter(ssp, obs, Uniform(0.5, 4.0))._engine_cache_key()
    b = _fitter(ssp, obs, Uniform(0.5, 4.0))._engine_cache_key()

    assert a == b, "equal priors must produce equal keys (else catalog fits recompile)"
    assert hash(a) == hash(b)


def test_engine_cache_key_is_hashable(ssp, obs):
    """The key is used as a dict key in the module-level caches."""
    key = _fitter(ssp, obs, Uniform(0.5, 4.0))._engine_cache_key()
    assert {key: "engine"}[key] == "engine"


# ── End-to-end: the result must not depend on what was fitted first ─────────


BANDS = ("galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z")

MASS = "sfh_tsnorm_log_total_mass"


def _mass_model(ssp_real, mass_prior):
    """Build the reproducer's model: only ``mass_prior`` varies between calls."""
    import tengri

    return tengri.SEDModel.build(
        ssp_real,
        observation=Observation(photometry=Photometry.from_names(list(BANDS))),
        sfh={
            "type": "tsnorm",
            "all_params": tengri.FIXED,
            "log_total_mass": mass_prior,
            "peak_lbt_gyr": Uniform(1.0, 6.0),
            "width_gyr": Uniform(1.0, 3.0),
            "skew": 0.3,
            "trunc": 3.0,
        },
        dust={
            "type": "two_component",
            # #1989: attenuation laws are explicit — 'law' applies one curve
            # to both screens, matching this test's pre-#1989 implicit default.
            "law": "calzetti",
            "all_params": tengri.FIXED,
            "tau_diff": Uniform(0.0, 0.6),
            "tau_bc": Uniform(0.0, 0.8),
        },
        redshift=tengri.Fixed(0.1),
    )


def _map_mass(ssp_real, mass_prior, data, noise):
    """MAP-fit ``mass_prior`` on ``data``; return recovered log_total_mass."""
    model = _mass_model(ssp_real, mass_prior)
    result = model.fit(data, noise, method="map", verbose=False)
    params = result.params if hasattr(result, "params") else result
    return float(np.asarray(params[MASS]).ravel()[0])


def test_prior_only_siblings_do_not_share_a_signal_response(ssp, obs):
    """The leak site itself: ``_SHARED_SIGNAL_RESPONSE_CACHE`` must not alias them.

    ``get_or_build_signal_response`` is keyed by ``_engine_cache_key()`` and the
    closure it caches wraps ``_primals_to_params`` — the function that reads
    ``dist._lo``/``dist._hi`` at trace time. Handing the narrow model's closure
    to the wide model is precisely how the latent gets decoded through the wrong
    interval, so assert the two never receive the same object.

    Deterministic and fit-free: unlike an end-to-end value comparison, this
    cannot go vacuous when the optimizer happens to land on the same latent.
    """
    clear_shared_caches()

    narrow = _fitter(ssp, obs, Uniform(0.5, 4.0))
    wide = _fitter(ssp, obs, Uniform(0.3, 9.0))

    narrow_response, _ = get_or_build_signal_response(narrow)
    wide_response, _ = get_or_build_signal_response(wide)

    assert narrow_response is not wide_response, (
        "prior-only siblings received the SAME cached signal_response closure; "
        "the wide model will decode its latent through the narrow model's bounds"
    )


def test_identical_specs_still_share_one_signal_response(ssp, obs):
    """Guard the other direction: equal specs must still hit the cache.

    This is the property that makes catalog fits compile once. If the fix
    over-keys, this test fails and the performance regression is caught here
    rather than in a user's 10^4-galaxy run.
    """
    clear_shared_caches()

    first = _fitter(ssp, obs, Uniform(0.5, 4.0))
    second = _fitter(ssp, obs, Uniform(0.5, 4.0))

    first_response, _ = get_or_build_signal_response(first)
    second_response, _ = get_or_build_signal_response(second)

    assert first_response is second_response, (
        "equal specs must reuse one cached signal_response, or every catalog "
        "row recompiles the physics stack"
    )


@pytest.mark.slow
def test_mass_is_independent_of_which_prior_was_fitted_first(ssp_data_fsps):
    """The reproducer, on a real SSP: a prior-only sibling must not move mass.

    Uses ``log_total_mass`` on a real SSP because the latent optimum has to be
    genuinely prior-dependent for a collision to be *observable*. On a
    degenerate fixture both models converge to the same latent, and then a
    collided decode and an honest one agree to 15 digits — the check goes
    vacuous, which is why the fit-free closure-identity test above carries the
    default-run protection.

    Measured before the fix (gallery prior fitted first, then the wide prior on
    identical data)::

        wide fitted alone   -> 11.0682
        wide after gallery  -> 12.6554   = 7.0 + 6.0 * Phi(xi_gallery)

    a 1.59 dex mass deviation produced purely by fit order.
    """
    gallery = Uniform(9.6, 11.1)  # the docs gallery's tight mass prior
    wide = Uniform(7.0, 13.0)

    truth_model = _mass_model(ssp_data_fsps, wide)
    truth = dict(truth_model.spec.sample(jax.random.PRNGKey(7)))
    truth[MASS] = 11.0
    truth["redshift"] = 0.1
    data = np.asarray(truth_model.predict_photometry(truth))
    noise = data / 100.0

    clear_shared_caches()
    alone = _map_mass(ssp_data_fsps, wide, data, noise)

    clear_shared_caches()
    _map_mass(ssp_data_fsps, gallery, data, noise)  # poison the shared cache
    after_sibling = _map_mass(ssp_data_fsps, wide, data, noise)

    assert after_sibling == pytest.approx(alone, rel=1e-6, abs=1e-8), (
        f"fitting a prior-only sibling first moved the recovered mass by "
        f"{after_sibling - alone:+.4f} dex ({alone!r} alone vs "
        f"{after_sibling!r} after). The shared engine cache decoded this fit's "
        f"latent through the sibling's prior bounds."
    )
