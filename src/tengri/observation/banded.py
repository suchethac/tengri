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
from scipy import linalg as scipy_linalg

# Speed of light and Gaussian FWHM conversion, mirror observation/spectrum.py
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
        Integer diagonal offsets (static, baked into the trace).
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
    JIT-compatible: yes. Gradient-safe: yes, linear in ``x`` and in ``data``.
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
    Build-time helper (offsets are static). No mutation of inputs, a rolled
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
    operator over that vector is block diagonal, camera :math:`m` occupying rows
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
    the composition is a rearrangement of the block data.

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
    validated against (and can subsume) the Gaussian ``apply_lsf`` path, and
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
    # single global scale under-broadened by wave[0]/lambda, the #1791 defect,
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


def row_sigma_kms(bm: BandedMatrix, wave_obs: jnp.ndarray) -> np.ndarray:
    r"""Gaussian-equivalent velocity width of each row of a banded LSF [km/s].

    Second central moment of row :math:`i` over its diagonal offsets,

    .. math::

        \sigma_{\mathrm{pix}, i}^2 = \sum_k w_{k,i} (o_k - \mu_i)^2, \qquad
        \mu_i = \sum_k w_{k,i} o_k, \qquad w_{k,i} = \frac{\mathrm{data}[k, i]}
        {\sum_{k'} \mathrm{data}[k', i]}

    with :math:`o_k` the diagonal offsets, converted to velocity with the
    local :math:`\mathrm{d}\ln\lambda` (exact on linear DESI grids, following
    the same per-pixel-scale convention as :func:`gaussian_resolution_bands`):
    :math:`\sigma_{v, i} = \sigma_{\mathrm{pix}, i} \, \mathrm{d}\ln\lambda_i
    \, c`.

    Parameters
    ----------
    bm : BandedMatrix
        Banded LSF operator, ``data`` shape ``(K, n_pix)``.
    wave_obs : array_like, shape (n_pix,)
        Observed wavelength grid [Angstrom].

    Returns
    -------
    ndarray, shape (n_pix,)
        Per-row Gaussian-equivalent velocity width [km/s].

    Notes
    -----
    Build-time NumPy helper; not traced. A row need not itself be Gaussian —
    this returns the second-moment-equivalent width regardless of shape, which
    is exact only when the row is Gaussian (used that way by
    :func:`deconvolve_library_lsf`). A column with zero row-sum (masked or
    edge pixels, common in real DESI resolution matrices) has no well-defined
    normalized weight; that column's width is returned as ``NaN``, computed
    via an explicit zero-sum mask rather than a bare division so it never
    raises ``RuntimeWarning: invalid value encountered``.
    """
    offs = np.asarray(bm.offsets, dtype=float).ravel()
    d = np.asarray(bm.data, dtype=float)
    col_sum = d.sum(axis=0)
    nonzero = col_sum != 0
    safe_sum = np.where(nonzero, col_sum, 1.0)  # placeholder divisor; result discarded below
    w = np.where(nonzero[None, :], d / safe_sum[None, :], np.nan)
    mu = (offs[:, None] * w).sum(0)
    var = ((offs[:, None] - mu) ** 2 * w).sum(0)
    dln = np.gradient(np.log(np.asarray(wave_obs, dtype=float)))
    return np.sqrt(var) * dln * _C_KM_S


def deconvolve_library_lsf(
    bm: BandedMatrix, wave_obs: jnp.ndarray, sigma_lib_kms, max_ratio: float = 0.6
) -> BandedMatrix:
    r"""Remove a Gaussian library LSF from a resolution matrix.

    :math:`R' \otimes G_{\sigma_{lib}} \approx R`.

    SSP spectra already carry the stellar library's resolution
    :math:`\sigma_{\mathrm{lib}}`; applying the instrument resolution matrix on
    top of a library-resolution model double-counts it. For each row this
    applies the second-order inverse of a Gaussian convolution along the
    diagonal index,

    .. math::

        R'[i, k] = R[i, k] - \frac{s_i^2}{2}\left(R[i, k+1] - 2 R[i, k]
        + R[i, k-1]\right), \qquad s_i = \frac{\sigma_{\mathrm{lib}, i}}
        {c \, \mathrm{d}\ln\lambda_i}

    with :math:`s_i` the library width in pixels, then renormalizes each row
    to its original sum. This keeps the row's non-Gaussian shape, centroid,
    and normalization; it is the standard finite-difference approximation of
    deconvolution by a narrow Gaussian kernel (second-order Taylor expansion
    of the convolution operator in the kernel width). Being a truncated
    Taylor expansion rather than an exact inverse, it can push a small amount
    of weight negative in a row's tails (measured on a Gaussian row: about
    -5e-6 of the row sum at ``sigma_lib/sigma_row = 0.2``, growing to about
    -3e-3 near ``max_ratio = 0.6``); this is expected and not corrected, since
    clipping would break the exact renormalization this function guarantees.

    Parameters
    ----------
    bm : BandedMatrix
        Banded resolution matrix ``R`` at library-and-instrument combined
        resolution, ``data`` shape ``(K, n_pix)``, with contiguous integer
        diagonal offsets (e.g. ``-5, -4, ..., 5``).
    wave_obs : array_like, shape (n_pix,)
        Observed wavelength grid [Angstrom].
    sigma_lib_kms : float or array_like, shape (n_pix,)
        Stellar library's Gaussian LSF width to remove [km/s]. ``0`` (scalar
        or every element) returns ``bm`` unchanged.
    max_ratio : float, optional
        Refuse rows where :math:`\sigma_{\mathrm{lib}} / \sigma_{\mathrm{row}}`
        exceeds this. Default 0.6.

    Returns
    -------
    BandedMatrix
        ``R'``, the resolution matrix with the library LSF removed, same
        offsets and shape as ``bm``. A column whose input row-sum is exactly
        zero (masked or edge pixels, common in real DESI resolution
        matrices) has no LSF to deconvolve and is returned exactly zero, not
        ``NaN``; such columns are also excluded from the ``max_ratio`` check
        below since :func:`row_sigma_kms` cannot define a width for them.

    Raises
    ------
    ValueError
        If ``bm.offsets`` are not contiguous integers; if any row with a
        nonzero sum has ``sigma_lib / sigma_row > max_ratio`` (no correct
        model can be narrower than its library, e.g. MILES at ~70 km/s
        against DESI's 25-40 km/s r/z arms, and the second-order expansion
        this function relies on is only valid for a library width small
        against the row it is being removed from); or if the deconvolution
        produces a non-finite value in any nonzero row (a defect in the
        inputs, not an expected outcome of this expansion).

    Notes
    -----
    Build-time NumPy helper; not traced. Usage contract: the model evaluated
    pre-``R`` is at library resolution, so pass
    ``resolution_matrix=deconvolve_library_lsf(R, wave, sigma_lib)`` together
    with ``sigma_lib_kms=0.0`` to :func:`~tengri.observation.spectrum.project_spectrum`
    (which applies velocity broadening before ``R @ model``), rather than
    letting ``R`` apply the instrument LSF on top of the library's.

    Same curve source as the default Gaussian ``apply_lsf`` path (#2518):
    pass ``sigma_lib=SSPData.ssp_resolution_kms`` (rest-frame array,
    interpolated onto ``wave`` in the observed frame at the model's
    redshift -- see :func:`~tengri.observation.spectrum.resolve_sigma_lib_kms`)
    when the loaded SSP grid documents one, rather than a flat per-library
    scalar, so both paths subtract the same per-wavelength library
    resolution.
    """
    offs = np.asarray(bm.offsets).ravel()
    order = np.argsort(offs)
    if not np.array_equal(offs[order], np.arange(offs.min(), offs.max() + 1)):
        raise ValueError("deconvolve_library_lsf needs contiguous diagonal offsets")
    d = np.asarray(bm.data, dtype=float)[order]
    wave = np.asarray(wave_obs, dtype=float)
    lib = np.broadcast_to(np.asarray(sigma_lib_kms, dtype=float), wave.shape)
    if np.all(lib == 0):
        return bm

    col_sum = d.sum(axis=0)
    nonzero = col_sum != 0
    if not np.any(nonzero):
        # Every row is masked/zero (e.g. a fully-flagged chunk); nothing to
        # deconvolve and row_sigma_kms is undefined everywhere.
        return bm

    sigma_row = row_sigma_kms(bm, wave)
    max_r = np.max(lib[nonzero] / sigma_row[nonzero])
    if max_r > max_ratio:
        raise ValueError(
            f"library LSF too broad for this resolution matrix: max sigma_lib/sigma_row = "
            f"{max_r:.2f} > {max_ratio}; use a higher-resolution library "
            f"(e.g. C3K R10K)"
        )

    s_pix = lib / _C_KM_S / np.gradient(np.log(wave))
    pad = np.pad(d, ((1, 1), (0, 0)))
    second = pad[2:] - 2.0 * pad[1:-1] + pad[:-2]
    out = d - 0.5 * s_pix[None, :] ** 2 * second
    out_sum = out.sum(axis=0)
    scale = np.ones_like(col_sum)
    scale[nonzero] = col_sum[nonzero] / out_sum[nonzero]
    out = out * scale[None, :]
    out[:, ~nonzero] = 0.0  # exact zero, never a 0/0 NaN from the renormalization above

    if not np.all(np.isfinite(out[:, nonzero])):
        raise ValueError(
            "deconvolve_library_lsf produced a non-finite value in a row with nonzero sum"
        )

    inv = np.argsort(order)
    return BandedMatrix(offsets=bm.offsets, data=jnp.asarray(out[inv]))


def _lower_storage(bm: BandedMatrix) -> np.ndarray:
    """Lower-band storage ``ab[k, j] = A[j + k, j]`` of a symmetric banded matrix."""
    offs = np.asarray(bm.offsets).ravel()
    data = np.asarray(bm.data, dtype=float)
    n = data.shape[1]
    half = int(max(-offs.min(), offs.max()))
    ab = np.zeros((half + 1, n))
    for k in range(half + 1):
        if -k in offs:
            ab[k, : n - k] = data[int(np.where(offs == -k)[0][0]), k:]
        elif k in offs:
            ab[k, : n - k] = data[int(np.where(offs == k)[0][0]), : n - k]
    return ab


def banded_cholesky(
    cov: BandedMatrix | np.ndarray, pix_mask: np.ndarray | None = None
) -> BandedMatrix:
    r"""Lower Cholesky factor of a banded covariance, with masked pixels set to identity.

    Factors :math:`C' = M C M + (I - M)` once on the host, where :math:`M` is the
    diagonal pixel mask. A masked row of the factor is the identity row, so a masked
    pixel carries no information and no normalization.

    Parameters
    ----------
    cov : BandedMatrix or array_like, shape (n, n)
        Symmetric positive-definite covariance [data units squared]. A dense array is
        reduced to the banded storage of its nonzero bandwidth.
    pix_mask : array_like, shape (n,), optional
        Pixel mask :math:`w \in \{0, 1\}^n`. Default: all ones.

    Returns
    -------
    BandedMatrix
        Lower factor :math:`L` with ``offsets = (0, -1, ..., -p)`` and
        ``data[k, i] = L[i, i - k]`` [data units], where :math:`p` is the half-bandwidth.

    Raises
    ------
    ValueError
        If the masked covariance is not positive definite.

    Notes
    -----
    **JIT-compatible**: no. Build-time NumPy and SciPy routine; call it outside traced
    code, once per data vector.

    The factor satisfies :math:`C' = L L^\mathsf{T}` with

    .. math::

        L_{ii} = \sqrt{C'_{ii} - \sum_{k=1}^{p} L_{i,i-k}^2}, \qquad
        L_{ij} = \frac{1}{L_{jj}}\Bigl(C'_{ij} - \sum_{k} L_{i,k} L_{j,k}\Bigr),

    computed by LAPACK band Cholesky (:func:`scipy.linalg.cholesky_banded`). The
    cost is :math:`O(n p^2)`.

    References
    ----------
    .. [1] Golub, G. H. & Van Loan, C. F. 2013, "Matrix Computations", 4th ed.,
       Johns Hopkins University Press, Section 4.3.6 (band Cholesky).
    """
    if isinstance(cov, BandedMatrix):
        bm = cov
    else:
        dense = np.asarray(cov, dtype=float)
        n_dense = dense.shape[0]
        i_idx, j_idx = np.nonzero(dense)
        half = int(np.max(np.abs(i_idx - j_idx))) if i_idx.size else 0
        offs = np.arange(-half, half + 1)
        data = np.zeros((offs.size, n_dense))
        for k, off in enumerate(offs):
            rows = np.arange(max(0, -off), min(n_dense, n_dense - off))
            data[k, rows] = dense[rows, rows + off]
        bm = BandedMatrix(offsets=jnp.asarray(offs), data=jnp.asarray(data))
    ab = _lower_storage(bm)
    half, n = ab.shape[0] - 1, ab.shape[1]
    if pix_mask is None:
        mask = np.ones(n)
    else:
        mask = np.asarray(pix_mask, dtype=float)
    for k in range(half + 1):
        ab[k, : n - k] = ab[k, : n - k] * mask[: n - k] * mask[k:]
    ab[0] = mask * ab[0] + (1.0 - mask)
    try:
        ab_l = scipy_linalg.cholesky_banded(ab, lower=True)
    except scipy_linalg.LinAlgError as exc:
        raise ValueError(f"covariance is not positive definite: {exc}") from exc
    data_l = np.zeros((half + 1, n))
    for lag in range(half + 1):
        data_l[lag, lag:] = ab_l[lag, : n - lag]
    return BandedMatrix(offsets=jnp.asarray(-np.arange(half + 1)), data=jnp.asarray(data_l))


def banded_forward_substitution(chol: BandedMatrix, r: jnp.ndarray) -> jnp.ndarray:
    r"""Solve :math:`L y = r` for a banded lower-triangular factor :math:`L`.

    Parameters
    ----------
    chol : BandedMatrix
        Lower factor from :func:`banded_cholesky`, ``offsets = (0, -1, ..., -p)``,
        ``data`` shape ``(p + 1, n)``.
    r : array_like, shape (n,)
        Right-hand side [data units].

    Returns
    -------
    ndarray, shape (n,)
        :math:`y = L^{-1} r` [data units of ``r`` divided by the factor units].

    Raises
    ------
    ValueError
        If ``chol.offsets`` are not the lower-factor layout ``(0, -1, ..., -p)``.

    Notes
    -----
    **JIT-compatible**: yes; the scan length is ``n`` and the band width comes from the
    static ``offsets``. **Gradient-safe**: yes, linear in ``r``. **vmap**: yes; ``vmap``
    over the columns of a matrix gives :math:`L^{-1} X`.

    The recurrence is

    .. math::

        y_i = \frac{1}{L_{ii}}\Bigl(r_i - \sum_{k=1}^{p} L_{i,i-k}\, y_{i-k}\Bigr),

    evaluated with :func:`jax.lax.scan` over pixels with a carry of the last :math:`p`
    values, costing :math:`O(n p)`. Rows of a masked pixel are identity rows, so
    :math:`y_i = r_i = 0` there.

    References
    ----------
    .. [1] Golub, G. H. & Van Loan, C. F. 2013, "Matrix Computations", 4th ed.,
       Johns Hopkins University Press, Section 3.1 (triangular systems).
    """
    offs = np.asarray(chol.offsets).ravel()
    half = -int(offs.min())
    if not np.array_equal(offs, -np.arange(half + 1)):
        raise ValueError("banded_forward_substitution needs offsets (0, -1, ..., -p)")
    diag = chol.data[0]
    coef = chol.data[1:].T
    if half == 0:
        return r / diag

    def step(prev, xs):
        r_i, c_i, d_i = xs
        y_i = (r_i - jnp.dot(c_i, prev)) / d_i
        return jnp.concatenate([y_i[None], prev[:-1]]), y_i

    init = jnp.zeros(half, dtype=jnp.result_type(r, diag))
    _, y = jax.lax.scan(step, init, (r, coef, diag))
    return y


def dense_to_banded(dense: np.ndarray, rtol: float = 1e-12) -> BandedMatrix | None:
    r"""Reduce a dense symmetric matrix to banded storage when it is banded.

    Parameters
    ----------
    dense : array_like, shape (n, n)
        Symmetric matrix [data units squared].
    rtol : float, optional
        Entries with :math:`|A_{ij}| \le` ``rtol`` :math:`\max_k |A_{kk}|` count as zero.
        Default: 1e-12.

    Returns
    -------
    BandedMatrix or None
        Symmetric banded storage with ``offsets = (-p, ..., p)``, or ``None`` when the
        half-bandwidth :math:`p` reaches a quarter of the matrix size, where banded
        storage saves nothing.

    Notes
    -----
    **JIT-compatible**: no. Build-time NumPy routine.

    Entries beyond the half-bandwidth are dropped, so the stored matrix differs from
    ``dense`` by at most ``rtol`` times the largest diagonal entry per element.
    """
    a = np.asarray(dense, dtype=float)
    n = a.shape[0]
    cutoff = rtol * np.max(np.abs(np.diag(a)))
    i_idx, j_idx = np.nonzero(np.abs(a) > cutoff)
    half = int(np.max(np.abs(i_idx - j_idx))) if i_idx.size else 0
    if 4 * half >= n:
        return None
    offs = np.arange(-half, half + 1)
    data = np.zeros((offs.size, n))
    for k, off in enumerate(offs):
        rows = np.arange(max(0, -off), min(n, n - off))
        data[k, rows] = a[rows, rows + off]
    return BandedMatrix(offsets=offs, data=jnp.asarray(data))


def banded_to_dense(bm: BandedMatrix) -> np.ndarray:
    """Dense ``(n, n)`` matrix of a banded matrix, with ``A[i, i + offsets[k]] = data[k, i]``."""
    offs = np.asarray(bm.offsets).ravel()
    data = np.asarray(bm.data, dtype=float)
    n = data.shape[1]
    out = np.zeros((n, n))
    for k, off in enumerate(offs):
        rows = np.arange(max(0, -off), min(n, n - off))
        out[rows, rows + off] = data[k, rows]
    return out
