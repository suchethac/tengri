# SPDX-License-Identifier: BSD-3-Clause
"""The one banded Cholesky: agreement with the dense factor, gradients, jit, and cost.

``banded_cholesky`` is jit-compatible and is the only banded factor of the package. These
cells compare it with ``numpy.linalg.cholesky`` on random banded positive-definite matrices,
differentiate it against finite differences, check that a calibration floor on a banded
covariance stays banded and equals the dense route, and record what the banded route saves
over the dense one in time per evaluation and in peak memory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.likelihoods.terms import floored_noise
from tengri.observation.banded import (
    BandedMatrix,
    banded_cholesky,
    banded_gram,
    banded_to_dense,
    dense_to_banded,
)

pytestmark = pytest.mark.contract

#: Largest ``max |L - L_numpy| / max |L_numpy|`` tolerated per dtype: about 4 times (float32) and
#: 20 times (float64) the largest value measured over the cells below (float32 1.3e-7,
#: float64 4.7e-16).
TOLERANCE = {np.dtype("float64"): 1e-14, np.dtype("float32"): 5e-7}


def _banded_spd(seed, n, p):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    i, j = np.indices((n, n))
    a = np.where(np.abs(i - j) <= p, 0.5 * (a + a.T), 0.0)
    return a + (np.abs(a).sum(axis=1).max() + 1.0) * np.eye(n)


def _lower_dense(factor, n):
    out = np.zeros((n, n))
    for k in range(factor.data.shape[0]):
        out[np.arange(k, n), np.arange(0, n - k)] = np.asarray(factor.data[k, k:])
    return out


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize("half", (1, 3, 8, 20))
def test_the_factor_matches_numpy_cholesky_on_random_banded_matrices(half, dtype):
    n = 120
    a = _banded_spd(half, n, half)
    cov = dense_to_banded(jnp.asarray(a, dtype), half=half)
    factor = banded_cholesky(cov)
    reference = np.linalg.cholesky(a)
    error = np.abs(_lower_dense(factor, n) - reference).max() / np.abs(reference).max()
    print(f"CHOL half={half} {np.dtype(dtype).name} relative error {error:.2e}")
    assert factor.data.dtype == np.dtype(dtype)
    assert error < TOLERANCE[np.dtype(dtype)]


def test_a_dense_array_is_reduced_to_its_bandwidth_and_a_wide_one_works():
    for half in (2, 11):
        a = _banded_spd(5, 40, half)
        factor = banded_cholesky(a)
        assert factor.data.shape[0] == half + 1
        np.testing.assert_allclose(_lower_dense(factor, 40), np.linalg.cholesky(a), atol=1e-12)


def test_the_factor_is_the_same_under_jit_and_with_a_mask():
    n, half = 90, 4
    a = _banded_spd(1, n, half)
    cov = dense_to_banded(a, half=half)
    mask = np.ones(n)
    mask[[5, 6, 40]] = 0.0
    run = jax.jit(lambda data, m: banded_cholesky(BandedMatrix(cov.offsets, data), m).data)
    eager = banded_cholesky(cov, mask).data
    np.testing.assert_allclose(np.asarray(run(cov.data, mask)), np.asarray(eager), atol=1e-13)
    masked = a * np.outer(mask, mask) + np.diag(1.0 - mask)
    np.testing.assert_allclose(
        _lower_dense(banded_cholesky(cov, mask), n), np.linalg.cholesky(masked), atol=1e-12
    )


def test_a_matrix_that_is_not_positive_definite_is_refused_when_concrete():
    a = _banded_spd(2, 30, 2)
    a[10, 10] = -5.0
    with pytest.raises(ValueError, match="positive definite"):
        banded_cholesky(dense_to_banded(a, half=2))


def test_the_gradient_through_the_factor_matches_finite_differences():
    n, half = 25, 3
    a = _banded_spd(3, n, half)
    cov = dense_to_banded(a, half=half)
    weights = np.random.default_rng(0).normal(size=(*cov.data.shape[:1], n))

    def loss(diag_shift, off):
        data = cov.data.at[half].add(diag_shift).at[half + 1].multiply(1.0 + off)
        factor = banded_cholesky(BandedMatrix(cov.offsets, data))
        lower = banded_cholesky(BandedMatrix(cov.offsets, data)).data
        return jnp.sum(jnp.asarray(weights[: half + 1]) * lower) + jnp.sum(jnp.log(factor.data[0]))

    shift = jnp.full(n, 0.1)
    grad = jax.grad(loss, argnums=(0, 1))(shift, 0.0)
    step = 1e-6
    fd_shift = np.array(
        [
            (loss(shift.at[i].add(step), 0.0) - loss(shift.at[i].add(-step), 0.0)) / (2 * step)
            for i in range(n)
        ]
    )
    fd_off = (loss(shift, step) - loss(shift, -step)) / (2 * step)
    np.testing.assert_allclose(np.asarray(grad[0]), fd_shift, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(float(grad[1]), float(fd_off), rtol=1e-6)


def test_the_gram_of_a_factor_is_its_covariance_and_dense_conversion_round_trips():
    n, half = 60, 5
    a = _banded_spd(4, n, half)
    factor = banded_cholesky(dense_to_banded(a, half=half))
    gram = banded_gram(factor)
    np.testing.assert_allclose(np.asarray(banded_to_dense(gram)), np.tril(a), atol=1e-11)
    symmetric = dense_to_banded(a, half=half)
    np.testing.assert_allclose(np.asarray(banded_to_dense(symmetric)), a, atol=0.0)


def test_a_floor_on_a_banded_covariance_stays_banded_and_equals_the_dense_route():
    n, half = 80, 4
    a = _banded_spd(6, n, half)
    factor = banded_cholesky(dense_to_banded(a, half=half))
    model = 1.0 + 0.3 * np.sin(np.linspace(0, 6, n))
    f_cal = 0.05
    _, bands, offsets = floored_noise(
        jnp.asarray(model), None, factor.data, tuple(int(o) for o in factor.offsets), f_cal
    )
    assert bands.shape == factor.data.shape and offsets == tuple(-k for k in range(half + 1))
    reference = np.linalg.cholesky(a + np.diag((f_cal * model) ** 2))
    got = _lower_dense(BandedMatrix(np.asarray(offsets), bands), n)
    np.testing.assert_allclose(got, reference, atol=1e-12)


_COST_SNIPPET = textwrap.dedent(
    """
    import json, resource, sys, time
    import jax, jax.numpy as jnp, numpy as np
    from tengri.inference.likelihoods.terms import floored_noise
    route, n, half = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(0)
    bands = np.zeros((half + 1, n))
    bands[0] = 1.0 + 0.1 * rng.random(n)
    bands[1:] = 0.05 * rng.standard_normal((half, n))
    model = jnp.asarray(1.0 + 0.3 * np.sin(np.linspace(0, 6, n)))
    offsets = tuple(-k for k in range(half + 1))
    data = jnp.asarray(bands)
    if route == "banded":
        run = jax.jit(lambda m, d: floored_noise(m, None, d, offsets, 0.05)[1][0].sum())
    else:
        def dense(m, d):
            lower = jnp.zeros((n, n))
            for k in range(half + 1):
                lower = lower + jnp.diag(d[k, k:], -k)
            cov = lower @ lower.T + jnp.diag((0.05 * m) ** 2)
            return jnp.diagonal(jnp.linalg.cholesky(cov)).sum()
        run = jax.jit(dense)
    run(model, data).block_until_ready()
    start = time.perf_counter()
    for _ in range(3):
        run(model, data).block_until_ready()
    seconds = (time.perf_counter() - start) / 3
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    print(json.dumps({"seconds": seconds, "rss_mib": rss}))
    """
)


def _cost(route, n, half=10):
    out = subprocess.run(
        [sys.executable, "-c", _COST_SNIPPET, route, str(n), str(half)],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_banded_floor_uses_less_memory_than_the_dense_route_at_large_n():
    """Records time per jitted evaluation and peak RSS; asserts memory at n = 4000 only."""
    table = {}
    for n in (1000, 4000):
        for route in ("banded", "dense"):
            table[(route, n)] = _cost(route, n)
            print(
                f"COST route={route} n={n} {table[(route, n)]['seconds']:.4f} s/eval "
                f"{table[(route, n)]['rss_mib']:.0f} MiB"
            )
    assert table[("banded", 4000)]["rss_mib"] < table[("dense", 4000)]["rss_mib"]
