# SPDX-License-Identifier: BSD-3-Clause
"""Galaxy batches run in chunks of galaxies, with one user-settable size (#3399).

A bare ``jax.vmap`` over B galaxies materializes each galaxy's ``(n_age, n_wave)``
per-age cube for all B at once. The batch paths now evaluate ``chunk_size``
galaxies per compiled call. Chunking is a memory strategy: the result must agree
with a single chunk to round-off, and the compiled chunk program must not hold a
``(B, n_age, n_wave)`` buffer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.convenience import predict_photometry_batch
from tengri.utils.batching import GALAXY_CHUNK_SIZE, vmap_chunked

pytestmark = pytest.mark.contract

_N_AGE = 20
_N_WAVE = 100
_B = 64
_CHUNK = 8
_RTOL = 1e-12


def _model(ssp, observation):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8, 12)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _batch(model, n: int):
    p = model.spec.sample(jax.random.PRNGKey(3))
    base = {k: jnp.asarray(v) for k, v in p.items()}
    keys = jax.random.split(jax.random.PRNGKey(7), n)
    draws = jax.vmap(model.spec.sample)(keys)
    return {k: draws[k] if k in draws else jnp.tile(v[None], (n,)) for k, v in base.items()}


def test_default_chunk_is_32():
    assert GALAXY_CHUNK_SIZE == 32


def test_photometry_batch_chunks_match_one_chunk(synthetic_ssp, synthetic_tophat_obs):
    model = _model(synthetic_ssp, synthetic_tophat_obs)
    batch = _batch(model, 7)
    one_chunk = np.asarray(predict_photometry_batch(model, batch, chunk_size=7))
    for chunk in (1, 3, 64):
        got = np.asarray(predict_photometry_batch(model, batch, chunk_size=chunk))
        np.testing.assert_allclose(got, one_chunk, rtol=_RTOL, atol=0)


def test_photometry_batch_rejects_nonpositive_chunk(synthetic_ssp, synthetic_tophat_obs):
    model = _model(synthetic_ssp, synthetic_tophat_obs)
    with pytest.raises(ValueError, match="chunk_size"):
        predict_photometry_batch(model, _batch(model, 2), chunk_size=0)


def test_shared_argument_is_broadcast_not_mapped():
    xs = jnp.arange(10.0)
    table = jnp.linspace(1.0, 2.0, 5)

    def one(x, shared):
        return x * jnp.sum(shared)

    got = vmap_chunked(one, chunk_size=3, shared=table)(xs)
    np.testing.assert_allclose(np.asarray(got), np.asarray(xs) * float(table.sum()), rtol=_RTOL)


def _per_galaxy_cube(p):
    """One galaxy: a materialized (n_age, n_wave) per-age cube, reduced to a scalar."""
    ages = jnp.linspace(0.1, 1.0, _N_AGE)[:, None]
    waves = jnp.linspace(0.5, 1.5, _N_WAVE)[None, :]
    cube = jnp.exp(-p["s"] * ages) * jnp.cos(waves * p["w"] * ages)
    return jnp.sum(cube, axis=(0, 1))


def _shape_count(text: str, shape: str) -> int:
    return text.count(shape)


def test_chunked_program_holds_no_full_batch_cube():
    params = {"s": jnp.linspace(0.5, 2.0, _B), "w": jnp.linspace(1.0, 3.0, _B)}
    full_shape = f"f64[{_B},{_N_AGE},{_N_WAVE}]"
    chunk_shape = f"f64[{_CHUNK},{_N_AGE},{_N_WAVE}]"

    unchunked = jax.jit(jax.vmap(_per_galaxy_cube)).lower(params).compile().as_text()
    chunked = jax.jit(lambda b: vmap_chunked(_per_galaxy_cube, chunk_size=_CHUNK)(b))
    chunked_text = chunked.lower(params).compile().as_text()

    # The detector must see the full-batch cube when it is there ...
    assert _shape_count(unchunked, full_shape) > 0
    # ... and must not see it once the batch is chunked.
    assert _shape_count(chunked_text, full_shape) == 0
    assert _shape_count(chunked_text, chunk_shape) > 0


def test_chunked_values_match_unchunked_cube():
    params = {"s": jnp.linspace(0.5, 2.0, _B), "w": jnp.linspace(1.0, 3.0, _B)}
    ref = np.asarray(jax.vmap(_per_galaxy_cube)(params))
    got = np.asarray(vmap_chunked(_per_galaxy_cube, chunk_size=_CHUNK)(params))
    np.testing.assert_allclose(got, ref, rtol=_RTOL, atol=0)
