# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end check that ``ForwardModel.predict`` returns batched output
for hierarchical fits.

After PR #224 (composable batching), the contract is:
``ForwardModel.predict(params)`` returns a shape-consistent dict
regardless of SubModel. Single-galaxy returns ``(n_filters,)``;
hierarchical returns ``(N_gal, n_filters)``.

This test exercises the end-to-end path on a real SED template +
real photometry observation: SED template → vmap'd state →
vmap'd observation → batched prediction dict.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel

pytestmark = pytest.mark.contract


def _template(synthetic_ssp, simple_observation):
    """Minimal real SEDModel template."""
    from tengri import FIXED, SEDModel, Uniform

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(-1.0, 3.0)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=0.05,
    )


def test_forward_predict_batched_for_population(synthetic_ssp, simple_observation) -> None:
    """forward.predict_observables returns (N_gal, n_filters) for PopulationSEDModel."""
    template = _template(synthetic_ssp, simple_observation)
    N = 3
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[{"flux_obs": jnp.zeros(5), "noise": jnp.ones(5)} for _ in range(N)],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)

    # Per-galaxy params — each free param is a length-N array.
    params: dict[str, jnp.ndarray] = {
        name: jnp.array([0.5] * N) for name in template.spec.free_params
    }

    pred = forward.predict_observables(params)
    assert isinstance(pred, dict)
    # Must have a photometric channel
    phot_key = next((k for k in ("phot_fnu", "fnu_obs") if k in pred), None)
    assert phot_key is not None, f"prediction dict missing photometric key: {list(pred)}"
    phot = pred[phot_key]
    # Leading axis is the galaxy axis
    assert phot.shape[0] == N, (
        f"expected leading axis of size {N} (one per galaxy); got shape {phot.shape}"
    )
    # All values finite
    assert jnp.all(jnp.isfinite(phot)), "non-finite values in batched prediction"


def test_forward_predict_single_galaxy_unchanged(synthetic_ssp, simple_observation) -> None:
    """forward.predict_observables for a single-galaxy SEDModel still returns (n_filters,).

    The standardization contract: regardless of SubModel, the prediction
    dict's per-channel shapes line up with the data the likelihood is fed.
    """
    template = _template(synthetic_ssp, simple_observation)
    forward = ForwardModel.build(sed=template, observation=simple_observation)
    params = {name: jnp.float64(0.5) for name in template.spec.free_params}
    pred = forward.predict_observables(params)
    phot_key = next((k for k in ("phot_fnu", "fnu_obs") if k in pred), None)
    assert phot_key is not None
    phot = pred[phot_key]
    # No leading galaxy axis for single-galaxy SubModel
    assert phot.ndim == 1, f"single-galaxy prediction should be 1-D, got shape {phot.shape}"


def test_forward_predict_batched_matches_per_galaxy_predict_one(
    synthetic_ssp, simple_observation
) -> None:
    """Batched forward.predict_observables == calling predict_one + observation.predict per galaxy.

    Numerical-equivalence gate: the vmap'd batched path should produce
    the same numbers as iterating predict_one over each galaxy and
    calling observation.predict separately. rtol=1e-10.
    """
    from tengri.protocols.component import ForwardState

    template = _template(synthetic_ssp, simple_observation)
    N = 2
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[{"flux_obs": jnp.zeros(5), "noise": jnp.ones(5)} for _ in range(N)],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)
    params_batched = {name: jnp.array([0.5, 0.7]) for name in template.spec.free_params}
    pred_batched = forward.predict_observables(params_batched)

    # Per-galaxy reference: call predict_one + observation.predict directly
    state = ForwardState(wave=jnp.zeros(1))
    per_galaxy = []
    for i in range(N):
        params_one = {name: jnp.array([0.5, 0.7])[i] for name in template.spec.free_params}
        state_one = pop.predict_one(state, params_one)
        # Merge fixed values (which ForwardModel.predict does internally)
        full_params = dict(template.spec.get_fixed_values())
        full_params.update(params_one)
        per_galaxy.append(simple_observation.predict(state_one, full_params))

    phot_key = next((k for k in ("phot_fnu", "fnu_obs") if k in pred_batched), None)
    assert phot_key is not None
    for i in range(N):
        ref = per_galaxy[i].get(phot_key)
        if ref is None:
            continue
        assert jnp.allclose(pred_batched[phot_key][i], ref, rtol=1e-10, atol=0.0), (
            f"batched prediction galaxy {i} differs from per-galaxy reference"
        )
