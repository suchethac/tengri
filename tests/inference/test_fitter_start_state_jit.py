# SPDX-License-Identifier: BSD-3-Clause
"""The NUTS chain starting state is built under jit, with draws unchanged.

``blackjax.mcmc.nuts.init`` evaluates the log-density and its gradient at the
chain's starting position. Called outside a traced function it runs eagerly,
one primitive dispatch at a time. These tests pin two things on a small
``Fitter.run("mcmc_nuts", ...)``:

* the starting-state initialisation is never called with concrete (untraced)
  values, so it runs inside a compiled function;
* the draws for a fixed key match the values recorded before the starting
  state was jitted.
"""

from __future__ import annotations

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.fitter import Fitter
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

_RUN_KW = dict(n_warmup=15, n_burnin=5, n_samples=12, n_chains=1, dense_mass_matrix=False)

# Draws of ``log_total_mass`` and ``tau_bc`` for key PRNGKey(7), recorded before the
# starting state was jitted. Shape (n_samples, 2).
_REFERENCE_FIRST_DRAWS = [
    [11.789154629953973, 0.8343496409775282],
    [11.959086183546873, 0.8659864702582116],
    [11.879432795545736, 0.48113121587916163],
]


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def fitter_and_data(synthetic_ssp_wide):
    obs = Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
    )
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    rng = np.random.default_rng(0)
    flux = np.abs(rng.normal(1.0, 0.1, size=3))
    flux_obs, noise = jnp.asarray(flux), jnp.asarray(0.1 * flux)
    return model, flux_obs, noise


def _record_start_state_calls(monkeypatch):
    """Record, per call of ``blackjax.mcmc.nuts.init``, whether it saw traced values."""
    calls = []
    original_init = blackjax.mcmc.nuts.init

    def recording_init(position, logdensity_fn, *args, **kwargs):
        calls.append(isinstance(position, jax.core.Tracer))
        return original_init(position, logdensity_fn, *args, **kwargs)

    monkeypatch.setattr(blackjax.mcmc.nuts, "init", recording_init)
    return calls


def test_nuts_start_state_is_built_under_jit(fitter_and_data, monkeypatch):
    model, flux_obs, noise = fitter_and_data
    calls = _record_start_state_calls(monkeypatch)
    Fitter(model, flux_obs, noise, data_type="photometry").run(
        "mcmc_nuts", key=jax.random.PRNGKey(7), verbose=False, **_RUN_KW
    )
    assert calls, "the single-chain starting state was never built"
    eager = [traced for traced in calls if not traced]
    assert not eager, f"{len(eager)} of {len(calls)} starting-state calls ran eagerly"


def test_nuts_draws_match_pre_jit_reference(fitter_and_data):
    model, flux_obs, noise = fitter_and_data
    post = Fitter(model, flux_obs, noise, data_type="photometry").run(
        "mcmc_nuts", key=jax.random.PRNGKey(7), verbose=False, **_RUN_KW
    )
    draws = np.stack(
        [
            np.asarray(post.samples["sfh_dpl_log_total_mass"]),
            np.asarray(post.samples["dust_tau_bc"]),
        ],
        axis=-1,
    )
    assert _REFERENCE_FIRST_DRAWS is not None
    np.testing.assert_array_equal(draws[:3], np.asarray(_REFERENCE_FIRST_DRAWS))
