# SPDX-License-Identifier: BSD-3-Clause
"""PopulationSEDModel.run evaluates the population in galaxy chunks (#3441).

``run`` used to vmap the template over all N galaxies at once, so under the
per-age contraction each galaxy's ``(n_age, n_wave)`` cube was live for every
galaxy simultaneously. ``run`` now takes ``chunk_size`` and evaluates through
:func:`tengri.utils.batching.vmap_chunked`. Chunking is a memory strategy: the
returned :class:`ForwardState` must agree with a single chunk to round-off on every
array leaf, and the compiled chunk program must not hold a full-population cube.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.protocols.component import ForwardState
from tengri.utils.batching import GALAXY_CHUNK_SIZE

pytestmark = pytest.mark.contract

_N = 7
_RTOL = 1e-12
_HLO_N = 16
_HLO_CHUNK = 4


def _template(synthetic_ssp, simple_observation):
    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(-1.0, 3.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=0.05,
    )


def _population(template, n_galaxies: int) -> PopulationSEDModel:
    galaxy = {"flux_obs": jnp.zeros(3), "noise": jnp.ones(3)}
    return PopulationSEDModel(sed=template, galaxies=[galaxy] * n_galaxies)


def _per_galaxy_params(template, n_galaxies: int) -> dict:
    """Every free parameter drawn per galaxy: each value has leading axis ``n_galaxies``."""
    keys = jax.random.split(jax.random.PRNGKey(11), n_galaxies)
    draws = jax.vmap(template.spec.sample)(keys)
    return {name: draws[name] for name in template.spec.free_params}


def _root_state(synthetic_ssp) -> ForwardState:
    return ForwardState(wave=jnp.asarray(synthetic_ssp.ssp_wave))


def _assert_states_close(got: ForwardState, ref: ForwardState) -> None:
    got_leaves = jax.tree_util.tree_leaves(got)
    ref_leaves = jax.tree_util.tree_leaves(ref)
    assert len(got_leaves) == len(ref_leaves) > 0
    for g, r in zip(got_leaves, ref_leaves, strict=True):
        assert np.shape(g) == np.shape(r)
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=_RTOL, atol=0)


def test_default_chunk_size_is_galaxy_chunk_size():
    default = inspect.signature(PopulationSEDModel.run).parameters["chunk_size"].default
    assert default == GALAXY_CHUNK_SIZE == 32


@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_chunked_run_matches_single_chunk_on_every_leaf(synthetic_ssp, simple_observation, chunk):
    template = _template(synthetic_ssp, simple_observation)
    pop = _population(template, _N)
    params = _per_galaxy_params(template, _N)
    state = _root_state(synthetic_ssp)

    reference = pop.run(state, params, chunk_size=_N)
    got = pop.run(state, params, chunk_size=chunk)
    _assert_states_close(got, reference)


def test_run_returns_galaxy_axis_on_every_per_galaxy_leaf(synthetic_ssp, simple_observation):
    template = _template(synthetic_ssp, simple_observation)
    pop = _population(template, _N)
    params = _per_galaxy_params(template, _N)
    out = pop.run(_root_state(synthetic_ssp), params, chunk_size=3)
    assert out.sed_observed.shape[0] == _N


def test_chunked_program_holds_no_full_population_cube(synthetic_ssp, simple_observation):
    template = _template(synthetic_ssp, simple_observation)
    pop = _population(template, _HLO_N)
    params = _per_galaxy_params(template, _HLO_N)
    state = _root_state(synthetic_ssp)

    n_age = synthetic_ssp.ssp_lg_age_gyr.shape[0]
    n_wave = synthetic_ssp.ssp_wave.shape[0]
    full_shape = f"f64[{_HLO_N},{n_age},{n_wave}]"

    def _program(chunk):
        # The returned state itself carries an (N, n_age, n_wave) leaf (a per-galaxy
        # cube in ``derived``), so the output cannot be what is inspected: reduce it
        # to a scalar, which forces every intermediate to exist in the compiled program.
        def _reduced(p):
            out = pop.run(state, p, chunk_size=chunk)
            return sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(out))

        return jax.jit(_reduced).lower(params).compile().as_text()

    # The detector must see the full-population cube when the population is one chunk ...
    unchunked = _program(_HLO_N)
    assert unchunked.count(full_shape) > 0
    # ... and must not see it once the population is chunked, while the chunk shape is there.
    chunked = _program(_HLO_CHUNK)
    assert chunked.count(full_shape) == 0
    assert chunked.count(f"f64[{_HLO_CHUNK},{n_age},{n_wave}]") > 0


def test_padding_does_not_change_results_when_not_divisible(synthetic_ssp, simple_observation):
    """N=5 with chunk 4 pads to 8 rows of real galaxies and trims back to 5."""
    template = _template(synthetic_ssp, simple_observation)
    n = 5
    pop = _population(template, n)
    params = _per_galaxy_params(template, n)
    state = _root_state(synthetic_ssp)

    padded = pop.run(state, params, chunk_size=4)
    reference = pop.run(state, params, chunk_size=n)
    assert padded.sed_observed.shape[0] == n
    _assert_states_close(padded, reference)


def test_traced_chunk_loop_is_one_loop_not_unrolled(synthetic_ssp, simple_observation):
    """The chunks run as one ``lax.map`` loop: the compiled program has a while loop."""
    template = _template(synthetic_ssp, simple_observation)
    pop = _population(template, _HLO_N)
    params = _per_galaxy_params(template, _HLO_N)
    state = _root_state(synthetic_ssp)

    def _reduced(p):
        out = pop.run(state, p, chunk_size=_HLO_CHUNK)
        return sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(out))

    text = jax.jit(_reduced).lower(params).compile().as_text()
    assert "while(" in text


@pytest.mark.parametrize("bad", [None, 2.5, "8", True])
def test_run_rejects_non_int_chunk_size(synthetic_ssp, simple_observation, bad):
    template = _template(synthetic_ssp, simple_observation)
    pop = _population(template, 2)
    params = _per_galaxy_params(template, 2)
    with pytest.raises(ValueError, match="chunk_size"):
        pop.run(_root_state(synthetic_ssp), params, chunk_size=bad)


def test_run_rejects_nonpositive_chunk_size(synthetic_ssp, simple_observation):
    template = _template(synthetic_ssp, simple_observation)
    pop = _population(template, 2)
    params = _per_galaxy_params(template, 2)
    with pytest.raises(ValueError, match="chunk_size"):
        pop.run(_root_state(synthetic_ssp), params, chunk_size=0)
