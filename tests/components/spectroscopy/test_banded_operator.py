# SPDX-License-Identifier: BSD-3-Clause
"""Banded-operator kernel for the spectroscopic forward model (#1163).

The banded matvec is the shared kernel behind the DESI/PFS resolution matrix
(Bolton & Schlegel 2010) and, in future, the SpectRes resample (Carnall 2017).
A dense matrix is the ground-truth reference for the storage convention.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401  (enables float64)
from tengri.observation.banded import (
    banded_matvec,
    gaussian_resolution_bands,
    resolution_bands_from_desi,
)
from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.contract


def _dense_from_bands(offsets, data):
    """Reference dense (n, n) matrix from tengri's banded convention.

    ``A[i, i + offsets[k]] = data[k, i]`` with out-of-range columns dropped.
    """
    offsets = np.asarray(offsets)
    data = np.asarray(data)
    n_diag, n = data.shape
    dense = np.zeros((n, n))
    for k in range(n_diag):
        o = int(offsets[k])
        for i in range(n):
            j = i + o
            if 0 <= j < n:
                dense[i, j] = data[k, i]
    return dense


def test_banded_matvec_matches_dense():
    rng = np.random.default_rng(0)
    n = 12
    offsets = np.array([-2, -1, 0, 1, 2])
    data = rng.standard_normal((offsets.shape[0], n))
    x = rng.standard_normal(n)
    y = banded_matvec(jnp.asarray(offsets), jnp.asarray(data), jnp.asarray(x))
    y_ref = _dense_from_bands(offsets, data) @ x
    np.testing.assert_allclose(np.asarray(y), y_ref, rtol=1e-12, atol=1e-12)


def test_desi_convention_roundtrip():
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(1)
    n = 15
    offsets = np.array([2, 1, 0, -1, -2])  # DESI orders diagonals high -> low
    diag_data = rng.standard_normal((offsets.shape[0], n))
    dense = sp.dia_matrix((diag_data, offsets), shape=(n, n)).toarray()
    x = rng.standard_normal(n)
    bm = resolution_bands_from_desi(jnp.asarray(diag_data), jnp.asarray(offsets))
    y = banded_matvec(bm.offsets, bm.data, jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(y), dense @ x, rtol=1e-12, atol=1e-12)


def test_banded_matvec_is_differentiable():
    rng = np.random.default_rng(2)
    n = 20
    offsets = jnp.asarray([-1, 0, 1])
    data = jnp.asarray(rng.standard_normal((3, n)))

    def loss(x):
        return jnp.sum(banded_matvec(offsets, data, x) ** 2)

    g = jax.grad(loss)(jnp.asarray(rng.standard_normal(n)))
    assert np.all(np.isfinite(np.asarray(g)))


def test_gaussian_bands_reproduce_apply_lsf():
    # Log-uniform observed grid (apply_lsf convolves in log-lambda).
    n = 400
    wave = np.geomspace(4000.0, 7000.0, n)
    resolution = 2000.0
    rng = np.random.default_rng(3)
    spec = 1.0 + 0.1 * rng.standard_normal(n)  # smooth-ish continuum
    bm = gaussian_resolution_bands(jnp.asarray(wave), resolution, n_diag=41)
    y_banded = banded_matvec(bm.offsets, bm.data, jnp.asarray(spec))
    y_fft = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), resolution, sigma_lib_kms=0.0)
    # Compare the interior (edges differ: FFT wraps, banded truncates).
    interior = slice(50, n - 50)
    np.testing.assert_allclose(
        np.asarray(y_banded)[interior],
        np.asarray(y_fft)[interior],
        rtol=0.0,
        atol=2e-2 * float(np.max(spec)),
    )
