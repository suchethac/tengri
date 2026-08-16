# SPDX-License-Identifier: BSD-3-Clause
"""Banded linear operators for the spectroscopic forward model.

A banded operator ``A`` acts on a model vector as ``y = A @ x`` where ``A`` is
nonzero only on a handful of diagonals. Both the DESI/PFS instrument resolution
matrix (Bolton & Schlegel 2010 [1]_; Guy et al. 2023 [2]_) and, in future, the
SpectRes flux-conserving resample (Carnall 2017 [3]_) share this representation,
so a single ``O(n * K)`` matvec covers both.

The storage convention is diagonal-offsets:
``A[i, i + offsets[k]] = data[k, i]``, with entries whose column index falls
outside ``[0, n)`` treated as zero. This mirrors how DESI/desispec ships the
resolution data (a ``(n_diag, n_pix)`` array of diagonals).

References
----------
.. [1] Bolton, A. S. & Schlegel, D. J. 2010, "Spectro-Perfectionism: An
       Algorithmic Framework for Photon Noise-Limited Extraction of Optical
       Fiber Spectroscopy", PASP, 122, 248, arXiv:0911.2689,
       DOI 10.1086/651008.
.. [2] Guy, J. et al. 2023, "The Spectroscopic Data Processing Pipeline for the
       Dark Energy Spectroscopic Instrument", AJ, 165, 144, arXiv:2209.14482,
       DOI 10.3847/1538-3881/acb212.
.. [3] Carnall, A. C. 2017, "SpectRes: A Fast Spectral Resampling Tool in
       Python", arXiv:1705.05165.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Speed of light and Gaussian FWHM conversion — mirror observation/spectrum.py
# so the banded Gaussian LSF matches apply_lsf exactly.
_C_KM_S = 299792.458  # km/s
_FWHM_TO_SIGMA = 2.354820045030949  # 2 * sqrt(2 * ln(2))


class BandedMatrix(NamedTuple):
    """Diagonal-offsets banded matrix.

    Attributes
    ----------
    offsets : ndarray, shape (K,)
        Integer diagonal offsets; diagonal ``k`` holds ``A[i, i + offsets[k]]``.
    data : ndarray, shape (K, n)
        Diagonal values; ``data[k, i]`` is the weight applied to
        ``x[i + offsets[k]]`` when forming ``y[i]``.
    """

    offsets: jnp.ndarray
    data: jnp.ndarray


@jax.jit
def banded_matvec(offsets: jnp.ndarray, data: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    r"""Apply a banded operator to a vector: ``y = A @ x``.

    .. math::

        y_i = \sum_k \mathrm{data}[k, i] \; x_{\,i + \mathrm{offsets}[k]}

    with :math:`x_j = 0` for :math:`j \notin [0, n)`.

    Parameters
    ----------
    offsets : array_like, shape (K,)
        Integer diagonal offsets (static — baked into the trace).
    data : array_like, shape (K, n)
        Diagonal values.
    x : array_like, shape (n,)
        Input vector.

    Returns
    -------
    ndarray, shape (n,)
        ``A @ x``.

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes — linear in ``x`` and in ``data``.
    Cost is ``O(n * K)`` via a gather, not the dense ``O(n^2)`` product.
    """
    n = x.shape[0]
    cols = jnp.arange(n)[None, :] + offsets[:, None]  # (K, n): i + offsets[k]
    valid = (cols >= 0) & (cols < n)
    gathered = jnp.where(valid, x[jnp.clip(cols, 0, n - 1)], 0.0)  # (K, n)
    return jnp.sum(data * gathered, axis=0)


def resolution_bands_from_desi(diag_data: jnp.ndarray, offsets: jnp.ndarray) -> BandedMatrix:
    r"""Ingest a DESI/desispec resolution matrix into a :class:`BandedMatrix`.

    DESI extracted spectra ship the resolution operator as a ``(n_diag, n_pix)``
    array of diagonals with the scipy ``dia_matrix`` convention:
    ``A[i, j] = diag_data[k, j]`` where ``j - i = offsets[k]``. This re-indexes
    into tengri's convention ``data[k, i] = A[i, i + offsets[k]]``
    (:func:`banded_matvec`), which is a per-diagonal shift by ``offsets[k]``.

    Parameters
    ----------
    diag_data : array_like, shape (n_diag, n_pix)
        Resolution diagonals as stored by desispec.
    offsets : array_like, shape (n_diag,)
        Integer diagonal offsets (desispec uses descending order, e.g.
        ``[+5, +4, ..., -5]`` for ``n_diag = 11``).

    Returns
    -------
    BandedMatrix
        The same operator in tengri's banded convention.

    Notes
    -----
    Build-time helper (offsets are static). No mutation of inputs — a rolled
    copy is returned. See Bolton & Schlegel 2010 [1]_ for the resolution-matrix
    representation and Guy et al. 2023 [2]_ for the DESI per-camera (b/r/z)
    storage.

    References
    ----------
    .. [1] Bolton, A. S. & Schlegel, D. J. 2010, PASP, 122, 248,
           arXiv:0911.2689.
    .. [2] Guy, J. et al. 2023, AJ, 165, 144, arXiv:2209.14482.
    """
    diag_data = jnp.asarray(diag_data)
    offsets = jnp.asarray(offsets)
    n_diag, n = diag_data.shape
    cols = jnp.arange(n)[None, :] + offsets[:, None]  # (K, n): i + offsets[k]
    valid = (cols >= 0) & (cols < n)
    rows = jnp.arange(n_diag)[:, None]
    rolled = jnp.where(valid, diag_data[rows, jnp.clip(cols, 0, n - 1)], 0.0)
    return BandedMatrix(offsets=offsets, data=rolled)


def block_diagonal_bands(blocks: Sequence[BandedMatrix]) -> BandedMatrix:
    r"""Compose per-segment banded operators into one block-diagonal operator.

    Multi-arm spectrographs deliver one resolution operator per camera, each on
    its own pixel grid (DESI b/r/z; Guy et al. 2023 [2]_). Concatenating the
    camera grids in camera order gives a single pixel vector, and the resolution
    operator over that vector is block diagonal — camera :math:`m` occupying rows
    and columns :math:`[s_m, s_m + n_m)`:

    .. math::

        A = \mathrm{diag}(A_0, A_1, \ldots, A_{M-1}), \qquad
        s_m = \sum_{m' < m} n_{m'}

    No band is allowed to reach across a segment boundary: an entry of block
    :math:`m` whose column index leaves :math:`[0, n_m)` is set to zero, so
    camera :math:`m`'s LSF never mixes photons into camera :math:`m+1`.

    Parameters
    ----------
    blocks : sequence of BandedMatrix
        Per-segment operators, in the order their pixel grids are concatenated.
        Block ``m`` has ``data`` of shape ``(K_m, n_m)``; the ``K_m`` and the
        offsets may differ between blocks.

    Returns
    -------
    BandedMatrix
        Operator of width ``sum(n_m)`` whose ``offsets`` are the sorted union of
        the per-block offsets.

    Raises
    ------
    ValueError
        If ``blocks`` is empty.

    Notes
    -----
    Build-time helper (NumPy; offsets are static). JIT-compatible: the returned
    operator is consumed by :func:`banded_matvec`, which is. Gradient-safe: yes
    — the composition is a rearrangement of the block data.

    Zeroing the cross-boundary reach is done here rather than inherited from the
    blocks: :func:`resolution_bands_from_desi` already zeroes its own edges, but
    :func:`gaussian_resolution_bands` does not, so relying on the blocks would
    leak one camera's LSF into the next.

    References
    ----------
    .. [2] Guy, J. et al. 2023, AJ, 165, 144, arXiv:2209.14482.
    """
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("block_diagonal_bands requires at least one block")

    sizes = [int(np.asarray(b.data).shape[1]) for b in blocks]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    offsets = np.unique(np.concatenate([np.asarray(b.offsets).ravel() for b in blocks]))
    data = np.zeros((offsets.shape[0], int(sum(sizes))))

    for block, start, n in zip(blocks, starts, sizes, strict=True):
        b_off = np.asarray(block.offsets).ravel()
        b_data = np.asarray(block.data)
        local = np.arange(n)
        for k_local, offset in enumerate(b_off):
            k = int(np.searchsorted(offsets, offset))
            # Rows whose band would leave this segment contribute nothing.
            inside = (local + offset >= 0) & (local + offset < n)
            # Accumulate rather than assign: a block that repeats an offset means
            # the two diagonals add, which is what banded_matvec does when it
            # sums over k. Assigning would silently drop one of them. Column
            # slices are disjoint across blocks, so this is identical to
            # assignment in the ordinary distinct-offset case.
            data[k, start : start + n] += np.where(inside, b_data[k_local], 0.0)

    return BandedMatrix(offsets=jnp.asarray(offsets), data=jnp.asarray(data))


def gaussian_resolution_bands(wave_obs: jnp.ndarray, resolution, n_diag: int = 11) -> BandedMatrix:
    r"""Banded Gaussian LSF equivalent to :func:`~tengri.observation.spectrum.apply_lsf`.

    Builds a normalized Gaussian kernel in log-wavelength space at spectral
    resolution :math:`R = \lambda / \Delta\lambda`
    (:math:`\sigma_v = c / (\mathrm{FWHM} \cdot R)`,
    :math:`\sigma_{\mathrm{pix}} = (\sigma_v / c) / \Delta\ln\lambda`),
    truncated to ``n_diag`` diagonals. Provided so the banded operator can be
    validated against — and can subsume — the Gaussian ``apply_lsf`` path, and
    as an explicit-matrix fallback for instruments that publish only a
    scalar/array ``R``.

    Parameters
    ----------
    wave_obs : array_like, shape (n_pix,)
        Observed wavelength grid [Angstrom]. Any strictly increasing grid; the
        kernel width is set from the local ``d ln lambda`` at each pixel, so a
        linearly-spaced grid is handled exactly rather than approximately (#1791).
    resolution : float or array_like, shape (n_pix,)
        Spectral resolution ``R`` (scalar or per-pixel), dimensionless.
    n_diag : int, optional
        Number of diagonals (odd). Default 11.

    Returns
    -------
    BandedMatrix
        Row-normalized Gaussian LSF operator.

    Notes
    -----
    Build-time helper (NumPy). The Gaussian ``apply_lsf`` is only an
    approximation of the true instrument LSF; prefer
    :func:`resolution_bands_from_desi` when the survey ships a resolution
    matrix (Bolton & Schlegel 2010 [1]_).

    References
    ----------
    .. [1] Bolton, A. S. & Schlegel, D. J. 2010, PASP, 122, 248,
           arXiv:0911.2689.
    """
    wave = np.asarray(wave_obs, dtype=float)
    n = wave.shape[0]
    R = np.broadcast_to(np.asarray(resolution, dtype=float), (n,))
    # Local d(ln lambda) per pixel, not the blue-end value for the whole array.
    # An explicit banded operator can carry a position-dependent pixel scale, so
    # unlike the FFT path it is exact on a linearly-spaced grid without any
    # resampling; np.gradient reduces to the constant on a log-uniform one. A
    # single global scale under-broadened by wave[0]/lambda — the #1791 defect,
    # which reached here as well.
    dlnwave = np.gradient(np.log(wave))  # (n,)
    sigma_pix = (_C_KM_S / (_FWHM_TO_SIGMA * R)) / _C_KM_S / dlnwave  # (n,)
    half = n_diag // 2
    offsets = np.arange(-half, half + 1)
    data = np.zeros((offsets.shape[0], n))
    for k, o in enumerate(offsets):
        # Row i receives x[i + o]; weight is a Gaussian in pixel offset o.
        data[k, :] = np.exp(-0.5 * (o / sigma_pix) ** 2)
    data /= data.sum(axis=0, keepdims=True)  # normalize per output pixel
    return BandedMatrix(offsets=jnp.asarray(offsets), data=jnp.asarray(data))
