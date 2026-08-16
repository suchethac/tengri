# SPDX-License-Identifier: BSD-3-Clause
"""Banded-operator kernel for the spectroscopic forward model (#1163).

The banded matvec is the shared kernel behind the DESI/PFS resolution matrix
(Bolton & Schlegel 2010) and, in future, the SpectRes resample (Carnall 2017).
A dense matrix is the ground-truth reference for the storage convention.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401  (enables float64)
from tengri.observation.banded import (
    BandedMatrix,
    banded_matvec,
    block_diagonal_bands,
    gaussian_resolution_bands,
    resolution_bands_from_desi,
)
from tengri.observation.spectrum import apply_lsf
from tests._grad_parity import assert_grad_matches_fd

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

    g = assert_grad_matches_fd(loss, jnp.asarray(rng.standard_normal(n)))
    assert np.all(np.isfinite(np.asarray(g)))


class TestBlockDiagonalBands:
    """Per-segment composition for multi-arm spectrographs (#1183).

    DESI delivers one operator per camera on its own grid; concatenating the
    camera grids in camera order makes the joint operator block diagonal.
    """

    @staticmethod
    def _blocks(rng, sizes, offsets=(-2, -1, 0, 1, 2)):
        offsets = np.asarray(offsets)
        return [
            BandedMatrix(
                offsets=jnp.asarray(offsets),
                data=jnp.asarray(rng.standard_normal((offsets.shape[0], n))),
            )
            for n in sizes
        ]

    def test_matches_a_dense_block_diagonal_reference(self):
        rng = np.random.default_rng(10)
        sizes = [7, 9, 5]
        blocks = self._blocks(rng, sizes)
        composed = block_diagonal_bands(blocks)

        n_total = sum(sizes)
        dense = np.zeros((n_total, n_total))
        start = 0
        for block, n in zip(blocks, sizes, strict=True):
            dense[start : start + n, start : start + n] = _dense_from_bands(
                block.offsets, block.data
            )
            start += n

        x = rng.standard_normal(n_total)
        y = banded_matvec(composed.offsets, composed.data, jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(y), dense @ x, rtol=1e-12, atol=1e-12)

    def test_no_cross_segment_leakage(self):
        """Zeroing one camera's input must not disturb any other camera."""
        rng = np.random.default_rng(11)
        sizes = [6, 8, 6]
        composed = block_diagonal_bands(self._blocks(rng, sizes))
        n_total = sum(sizes)
        x = rng.standard_normal(n_total)

        full = np.asarray(banded_matvec(composed.offsets, composed.data, jnp.asarray(x)))
        x_zeroed = x.copy()
        x_zeroed[: sizes[0]] = 0.0
        muted = np.asarray(banded_matvec(composed.offsets, composed.data, jnp.asarray(x_zeroed)))

        assert np.max(np.abs(muted[: sizes[0]])) == 0.0
        np.testing.assert_array_equal(muted[sizes[0] :], full[sizes[0] :])

    def test_gaussian_blocks_do_not_leak_across_the_seam(self):
        """``gaussian_resolution_bands`` does not zero its own edges.

        The composition must therefore zero the cross-boundary reach itself;
        inheriting it from the blocks would spill one camera's LSF into the
        next. This is the reason the zeroing lives in
        :func:`block_diagonal_bands`.
        """
        n = 20
        wave_a = np.geomspace(4000.0, 5000.0, n)
        wave_b = np.geomspace(4990.0, 6000.0, n)  # overlapping, as DESI arms are
        blocks = [
            gaussian_resolution_bands(jnp.asarray(w), 2000.0, n_diag=7) for w in (wave_a, wave_b)
        ]
        composed = block_diagonal_bands(blocks)
        dense = _dense_from_bands(composed.offsets, composed.data)

        assert np.max(np.abs(dense[:n, n:])) == 0.0
        assert np.max(np.abs(dense[n:, :n])) == 0.0

    def test_blocks_may_carry_different_offsets_and_widths(self):
        rng = np.random.default_rng(12)
        narrow = BandedMatrix(
            offsets=jnp.asarray([-1, 0, 1]),
            data=jnp.asarray(rng.standard_normal((3, 5))),
        )
        wide = BandedMatrix(
            offsets=jnp.asarray([-3, -1, 0, 2]),
            data=jnp.asarray(rng.standard_normal((4, 9))),
        )
        composed = block_diagonal_bands([narrow, wide])

        np.testing.assert_array_equal(np.asarray(composed.offsets), np.array([-3, -1, 0, 1, 2]))
        assert np.asarray(composed.data).shape == (5, 14)

        dense = np.zeros((14, 14))
        dense[:5, :5] = _dense_from_bands(narrow.offsets, narrow.data)
        dense[5:, 5:] = _dense_from_bands(wide.offsets, wide.data)
        x = rng.standard_normal(14)
        y = banded_matvec(composed.offsets, composed.data, jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(y), dense @ x, rtol=1e-12, atol=1e-12)

    def test_single_block_round_trips(self):
        rng = np.random.default_rng(13)
        (block,) = self._blocks(rng, [11])
        composed = block_diagonal_bands([block])
        x = rng.standard_normal(11)
        np.testing.assert_allclose(
            np.asarray(banded_matvec(composed.offsets, composed.data, jnp.asarray(x))),
            np.asarray(banded_matvec(block.offsets, block.data, jnp.asarray(x))),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_repeated_offsets_add_as_banded_matvec_would(self):
        """A block that repeats an offset means the diagonals add, not replace."""
        n = 6
        first = np.full(n, 0.25)
        second = np.full(n, 0.75)
        repeated = BandedMatrix(
            offsets=jnp.asarray([0, 0]),
            data=jnp.asarray(np.stack([first, second])),
        )
        composed = block_diagonal_bands([repeated])
        x = np.arange(1.0, n + 1.0)

        np.testing.assert_allclose(
            np.asarray(banded_matvec(composed.offsets, composed.data, jnp.asarray(x))),
            np.asarray(banded_matvec(repeated.offsets, repeated.data, jnp.asarray(x))),
            rtol=1e-12,
            atol=1e-12,
        )
        # Explicitly: 0.25 + 0.75 = 1.0, so the result is x itself. Assigning
        # instead of accumulating would have dropped one diagonal and given
        # 0.75 * x.
        np.testing.assert_allclose(
            np.asarray(banded_matvec(composed.offsets, composed.data, jnp.asarray(x))),
            x,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_empty_composition_raises(self):
        with pytest.raises(ValueError, match="at least one block"):
            block_diagonal_bands([])


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
