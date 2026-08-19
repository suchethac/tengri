# SPDX-License-Identifier: BSD-3-Clause
"""#1972: baked spec/model state that the inference cache key did not separate.

Follow-up to #1971, which fixed the first of five instances of one class:
a value baked into the compiled inference closure while the cache key is blind
to it, so the second of two models silently runs the first's physics.
Order-dependent, so each model fitted *alone* is correct.

Covered here:

* **instance 2** — spec ``Fixed`` VALUES. ``_primals_to_params`` bakes
  ``fitter._fixed_values``. Measured -0.18 dex on mass for ``dust_slope``
  -0.7 -> 0.4. ``SEDModel.compile_signature`` dropped these on 2026-05-20
  believing them threaded; that holds for the forward observables path, not for
  this closure.
* **instance 3** — the mirror map. ``spec.resolve_mirrors`` bakes
  ``target -> source``; two specs sharing every name and prior but tying to
  different sources collided, so the second tied to the wrong source silently.
* **instance 4** — the MODEL signature was missing from
  ``_SHARED_SIGNAL_RESPONSE_CACHE`` entirely. It keyed on
  ``_engine_cache_key()``, whose docstring claimed to capture "model structure"
  but carries only data_type, spec flags, free names, feature channels and the
  override. Two models differing solely in an attenuation law shared one
  compiled physics closure.

Instance 1 lives in ``test_prior_bounds_key_the_engine_cache.py``.
Instance 5 (SSP grid values baked, +0.9962 dex) is tracked separately — it
needs a memoized content hash, since hashing a 67 MB grid costs 17.7 ms and
``compile_signature`` runs once per Fitter.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import Fitter, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference.jit_engine import clear_shared_caches, get_or_build_signal_response
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression


@pytest.fixture
def ssp():
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
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))


def _fitter(model):
    return Fitter(model, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")


def _spec_fitter(ssp, obs, **spec_kwargs):
    base = dict(redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0))
    base.update(spec_kwargs)
    return _fitter(SEDModel(Parameters(**base), ssp, observation=obs))


def _share_a_closure(fa, fb):
    clear_shared_caches()
    ra, _ = get_or_build_signal_response(fa)
    rb, _ = get_or_build_signal_response(fb)
    return ra is rb


# ── instance 2: spec Fixed VALUES ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        ({"dust_slope": -0.7}, {"dust_slope": 0.4}, "dust_slope (measured -0.18 dex)"),
        ({"met_logzsol": 0.0}, {"met_logzsol": -1.0}, "met_logzsol"),
        ({"dust_Rv": 3.1}, {"dust_Rv": 5.5}, "dust_Rv"),
    ],
)
def test_spec_fixed_values_key_the_engine_cache(ssp, obs, left, right, why):
    """Two models differing only in one baked fixed scalar must not collide."""
    fa = _spec_fitter(ssp, obs, **left)
    fb = _spec_fitter(ssp, obs, **right)

    assert fa._engine_cache_key() != fb._engine_cache_key(), (
        f"engine cache key collides across fixed value {why}"
    )
    assert not _share_a_closure(fa, fb), (
        f"models differing only in fixed value {why} received the SAME cached "
        f"closure; the second runs the first's physics"
    )


def test_equal_fixed_values_still_share_one_closure(ssp, obs):
    """Do not over-key: identical fixed values must still reuse one compile."""
    fa = _spec_fitter(ssp, obs, dust_slope=-0.7)
    fb = _spec_fitter(ssp, obs, dust_slope=-0.7)
    assert fa._engine_cache_key() == fb._engine_cache_key()
    assert _share_a_closure(fa, fb), "equal specs must share one compiled closure"


# ── instance 3: the mirror map ──────────────────────────────────────────────


def test_mirror_map_keys_the_engine_cache(ssp, obs):
    """Same names, same priors, different tie source — must not collide.

    Before the fix both specs got ``{'dust_tau_bc': 'dust_tau_diff'}``, so the
    second silently tied ``dust_tau_bc`` to the wrong parameter.
    """
    common = dict(dust_tau_diff=Uniform(0.0, 0.6), dust_slope=Uniform(0.0, 0.6))
    fa = _spec_fitter(ssp, obs, **common, dust_tau_bc="dust_tau_diff")
    fb = _spec_fitter(ssp, obs, **common, dust_tau_bc="dust_slope")

    assert fa.spec.mirrors != fb.spec.mirrors, "fixture must differ in the mirror map"
    assert fa._engine_cache_key() != fb._engine_cache_key(), "mirror map must key the cache"
    assert not _share_a_closure(fa, fb), (
        "specs tying the same target to different sources shared one closure"
    )


# ── instance 4: the model signature must reach the closure cache ────────────


def test_model_structure_keys_the_signal_response_cache(ssp, obs):
    """Two models differing only in an attenuation law must not share a closure.

    ``_SHARED_SIGNAL_RESPONSE_CACHE`` keyed on ``_engine_cache_key()``, which
    carries no model physics config, so this collided even though
    ``SEDModel.compile_signature`` distinguishes the law by name.
    """

    def build(law):
        return tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "all_params": tengri.FIXED,
                "alpha": Uniform(0.5, 4.0),
                "beta": Uniform(0.3, 3.0),
            },
            dust={"type": "two_component", "all_params": tengri.FIXED, "law_bc": law},
            redshift=0.1,
        )

    ma, mb = build("calzetti"), build("cardelli")
    assert ma.compile_signature() != mb.compile_signature(), (
        "fixture must differ in the model signature"
    )
    assert not _share_a_closure(_fitter(ma), _fitter(mb)), (
        "models differing only in attenuation law shared one compiled closure; "
        "the signal_response cache key must include the model signature"
    )


def test_same_model_still_shares_one_closure(ssp, obs):
    """The cross-galaxy property instance 4's fix must preserve."""

    def build():
        return tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "all_params": tengri.FIXED,
                "alpha": Uniform(0.5, 4.0),
                "beta": Uniform(0.3, 3.0),
            },
            dust={"type": "two_component", "all_params": tengri.FIXED, "law_bc": "calzetti"},
            redshift=0.1,
        )

    assert _share_a_closure(_fitter(build()), _fitter(build())), (
        "two Fitters on equivalent models must share one closure, or catalog "
        "fits recompile the physics stack per row"
    )
