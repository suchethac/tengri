# SPDX-License-Identifier: BSD-3-Clause
"""The keyed-or-threaded contract for the inference closure (#1972).

Anything ``_build_signal_response`` closes over is baked into the compiled
graph at trace time. ``Fitter._engine_cache_key()`` must distinguish it, or the
six module-level caches (``_SHARED_ENGINE_CACHE``,
``_SHARED_SIGNAL_RESPONSE_CACHE``, and the loss / grad / logdensity / loglik
caches, all keyed off that tuple) hand one model's compiled program to another
and the second model silently runs the first model's physics.

The contract is stated in ``SEDModel.compile_signature``'s docstring —
"Changes ... that affect JIT graph shape MUST be added to this method to avoid
silent miscompilation" — but nothing enforced it, and it failed three times:

* free-parameter priors        -> 1.53 dex on ``log_total_mass``  (#1971)
* spec fixed VALUES           -> -0.18 dex via ``dust_slope``     (#1972)
* the mirror map              -> silently ties to the wrong source (#1972)

Each is order-dependent and silent: every model fitted *alone* is correct, so a
suite that builds one model per test never sees it.

This module is the structural guard. Rather than testing the three known
instances one more time, it pins **what the closure closes over at all**, so a
newly baked ingredient fails here — by name — until it is either keyed or
genuinely threaded through ``data_args``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fitter, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference.jit_engine import _build_signal_response
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.contract


# Every free variable ``_primals_to_params`` closes over, and how each is made
# safe. Adding a name here is a deliberate act that must come with either a
# ``_engine_cache_key()`` entry or runtime threading.
BAKED_INGREDIENTS: dict[str, str] = {
    "free_names": "keyed: field 5, tuple(sorted(self._free_names))",
    "spec": (
        "keyed: priors via _free_prior_key (#1971), fixed values via "
        "_fixed_value_key, mirror map via _mirror_key (#1972)"
    ),
    "fixed_values": "keyed: _fixed_value_key + the params_override entry",
    "stochastic": "keyed: field 2, self.spec.stochastic",
}


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
def fitter(ssp):
    spec = Parameters(
        redshift=0.1,
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )
    model = SEDModel(
        spec,
        ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
    )
    return Fitter(model, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")


def test_primals_to_params_bakes_only_accounted_ingredients(fitter):
    """Pin the closure's free variables against the keyed-or-threaded ledger.

    If this fails with an ADDED name, that value is now baked into every
    compiled inference graph while the cache key may be blind to it. Fix it by
    keying the value in ``Fitter._engine_cache_key()`` (cheap; correct for
    per-model values) or by threading it through ``data_args`` the way a
    runtime-routed redshift is (#1316) — then record it in
    ``BAKED_INGREDIENTS``.

    If it fails with a REMOVED name, the value stopped being baked; drop it
    from the ledger and consider whether its cache-key entry is now dead weight
    forcing needless recompiles.
    """
    _, primals_to_params = _build_signal_response(fitter)

    actual = set(primals_to_params.__code__.co_freevars)
    expected = set(BAKED_INGREDIENTS)

    added = sorted(actual - expected)
    removed = sorted(expected - actual)

    assert not added, (
        f"_primals_to_params now bakes {added}, which is not in the "
        f"keyed-or-threaded ledger. Every baked value must be distinguished by "
        f"Fitter._engine_cache_key() or threaded through data_args, else two "
        f"models differing only in it share one compiled program and the "
        f"second silently runs the first's physics (#1971, #1972). "
        f"Key it or thread it, then add it to BAKED_INGREDIENTS with which."
    )
    assert not removed, (
        f"_primals_to_params no longer bakes {removed}. Remove it from "
        f"BAKED_INGREDIENTS, and check whether its cache-key entry is now "
        f"forcing recompiles for nothing."
    )


def test_every_ledger_entry_names_its_protection(fitter):
    """The ledger must say *how* each ingredient is protected, not merely list it.

    A bare name would let a future edit record an ingredient as accounted-for
    without anything actually keying it.
    """
    for name, how in BAKED_INGREDIENTS.items():
        assert any(word in how for word in ("keyed", "threaded")), (
            f"BAKED_INGREDIENTS[{name!r}] must state 'keyed' or 'threaded'; got {how!r}"
        )


def test_signal_response_closes_over_no_unaccounted_spec_state(fitter):
    """The outer ``signal_response`` must add no baked spec state of its own.

    It legitimately closes over the model and the data_type; anything else that
    varies per spec belongs in the ledger above.
    """
    signal_response, _ = _build_signal_response(fitter)

    allowed = {"model", "data_type", "use_components", "_primals_to_params"}
    unexpected = sorted(set(signal_response.__code__.co_freevars) - allowed)

    assert not unexpected, (
        f"signal_response now closes over {unexpected}. If any of these vary "
        f"with the spec, they must be keyed or threaded — see the module "
        f"docstring and BAKED_INGREDIENTS."
    )
