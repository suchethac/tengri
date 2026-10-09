# SPDX-License-Identifier: BSD-3-Clause
"""Batched per-galaxy spectroscopic observations for catalog fits.

A catalog fit evaluates many galaxies in one compiled function. Each galaxy has
its own pixel grid, pixel count, resolution operator, redshift and photometry,
so the observation has to enter the function as an argument rather than as a
closed-over constant. This module holds that per-galaxy data as one pytree,
:class:`SpectroBatch`, with a leading batch axis, and builds it from a list of
:class:`GalaxySpectrum` records.

Static and traced parts
-----------------------
The data split into two halves:

* **Traced** (:class:`SpectroBatch`): pixel wavelengths, masks, fluxes, errors,
  the resolution rows, redshifts, calibration ranges and photometry. Each
  leaf has a leading batch axis ``B``, so the batch can be mapped with
  :func:`jax.vmap` or passed as a regular argument to a jitted function.
* **Static** (:class:`SpectroBatchSpec`): the bucketed pixel count ``n_max``,
  the banded diagonal offsets, the grid kind, the spectral-LSF binning and the
  photometry width. These fix array shapes and control flow, so they are part
  of the compile signature. The builder groups galaxies by spec so that each
  bucket compiles once.

Why the grid kind is static
---------------------------
The grid kind decides which resolution path the forward model takes: a
log-uniform grid admits a single-FFT constant-R convolution, a nonuniform one
does not, and a banded operator is applied directly. The kind has to be
decided from concrete NumPy arrays here, at batch construction. The existing
:func:`tengri.observation.spectrum._is_log_uniform` returns ``True`` for any
tracer because it has no values to inspect, so a grid passed through
``jit`` as an argument would silently take the log-uniform path even when it
is linear in wavelength.

JIT compatibility
-----------------
The builders and padding helpers in this module are NumPy and run once, outside
any trace. :class:`SpectroBatch` is a registered pytree and is fully
JIT-, ``vmap``- and ``grad``-compatible. Its leaves are the only traced
quantities; :class:`SpectroBatchSpec` is hashable and must be closed over or
passed as a static argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.observation.banded import BandedMatrix

_GRID_KINDS: frozenset[str] = frozenset({"banded", "log_uniform", "nonuniform"})

# Relative tolerance on d(ln lambda) for a grid to count as log-uniform.
_LOG_UNIFORM_RTOL = 1e-6

_BATCH_FIELDS: tuple[str, ...] = (
    "wave",
    "pix_mask",
    "flux",
    "sigma",
    "r_data",
    "resolution_pp",
    "z",
    "cal_lo",
    "cal_hi",
    "phot_flux",
    "phot_err",
    "phot_presence",
)


class GalaxySpectrum(NamedTuple):
    """One galaxy's observation, before batching.

    Attributes
    ----------
    wave : array_like, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom], strictly increasing.
    flux : array_like, shape (n_pix,)
        Observed flux [erg/s/cm^2/Angstrom].
    ivar : array_like, shape (n_pix,)
        Inverse variance of ``flux`` [(erg/s/cm^2/Angstrom)^-2]. Pixels with
        ``ivar <= 0`` are excluded.
    resolution : BandedMatrix, float or array_like, shape (n_pix,)
        Resolution operator: a banded matrix over the pixel grid, or a
        spectral resolution ``R = lambda / delta lambda`` (scalar or per pixel).
    z : float
        Redshift of the galaxy.
    mask : array_like of bool, shape (n_pix,), optional
        Pixel mask, ``True`` for good pixels. ``None`` means all pixels are good.
    phot_flux : array_like, shape (n_filt,), optional
        Broadband fluxes [erg/s/cm^2/Angstrom], one per filter.
    phot_err : array_like, shape (n_filt,), optional
        1-sigma errors on ``phot_flux``, same units. Required with ``phot_flux``.
    """

    wave: Any
    flux: Any
    ivar: Any
    resolution: Any
    z: float
    mask: Any = None
    phot_flux: Any = None
    phot_err: Any = None


@dataclass(frozen=True)
class SpectroBatchSpec:
    """Static bucket key shared by every galaxy in a :class:`SpectroBatch`.

    Attributes
    ----------
    n_max : int
        Padded pixel count of every galaxy in the bucket.
    offsets : tuple of int or None
        Diagonal offsets of the banded resolution operator. ``None`` for a
        Gaussian resolution ``R``.
    grid_kind : str
        One of ``"banded"`` (banded resolution), ``"log_uniform"`` or
        ``"nonuniform"`` (Gaussian resolution on that grid).
    lsf_n_bins : int
        Number of bins used by the spectral LSF approximation. Stored in the
        key so that buckets built with different binning never share a compile.
    has_phot : bool
        Whether the bucket carries photometry.
    n_filt : int
        Number of photometric filters; ``0`` when ``has_phot`` is ``False``.

    Raises
    ------
    ValueError
        If a field is out of range, or if the fields disagree: ``grid_kind``
        must be ``"banded"`` exactly when ``offsets`` is set, and ``n_filt``
        must be positive exactly when ``has_phot`` is ``True``.

    Notes
    -----
    Frozen and hashable, so it is safe as a static argument to ``jit`` and as a
    dictionary key. It is not a pytree and carries no arrays.
    """

    n_max: int
    offsets: tuple[int, ...] | None
    grid_kind: str
    lsf_n_bins: int = 16
    has_phot: bool = False
    n_filt: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.n_max, int) or self.n_max < 1:
            raise ValueError(f"n_max must be a positive integer, got {self.n_max!r}")
        if self.grid_kind not in _GRID_KINDS:
            raise ValueError(
                f"grid_kind must be one of {sorted(_GRID_KINDS)}, got {self.grid_kind!r}"
            )
        if self.offsets is not None and not (
            isinstance(self.offsets, tuple)
            and len(self.offsets) > 0
            and all(isinstance(o, int) for o in self.offsets)
        ):
            raise ValueError("offsets must be None or a non-empty tuple of ints")
        if (self.grid_kind == "banded") != (self.offsets is not None):
            raise ValueError("grid_kind is 'banded' exactly when offsets are given")
        if not isinstance(self.lsf_n_bins, int) or self.lsf_n_bins < 1:
            raise ValueError(f"lsf_n_bins must be a positive integer, got {self.lsf_n_bins!r}")
        if self.has_phot and self.n_filt < 1:
            raise ValueError("has_phot=True requires n_filt >= 1")
        if not self.has_phot and self.n_filt != 0:
            raise ValueError("n_filt must be 0 when has_phot is False")


@dataclass(frozen=True)
class SpectroBatch:
    """Batch of per-galaxy spectroscopic observations, with a leading batch axis.

    Attributes
    ----------
    wave : ndarray, shape (B, n_max)
        Padded observed wavelength grid [Angstrom].
    pix_mask : ndarray, shape (B, n_max)
        1.0 for a good real pixel, 0.0 for padding, masked or ``ivar <= 0``.
    flux : ndarray, shape (B, n_max)
        Observed flux [erg/s/cm^2/Angstrom]; 0.0 on excluded pixels.
    sigma : ndarray, shape (B, n_max)
        1-sigma flux error; 1.0 on excluded pixels.
    r_data : ndarray, shape (B, K, n_max)
        Banded resolution rows, ``K`` being the number of offsets. ``K = 0``
        (a zero-size placeholder) when the resolution is Gaussian.
    resolution_pp : ndarray, shape (B, n_max)
        Per-pixel spectral resolution ``R``. A zero-size placeholder of shape
        ``(B, 0)`` when the resolution is banded.
    z : ndarray, shape (B,)
        Redshift.
    cal_lo, cal_hi : ndarray, shape (B,)
        Smallest and largest real wavelength [Angstrom].
    phot_flux, phot_err : ndarray, shape (B, n_filt)
        Photometry; zero-width when the bucket has no photometry.
    phot_presence : ndarray, shape (B, n_filt)
        1.0 where a filter has a finite flux and a positive error, else 0.0.

    Notes
    -----
    Registered as a pytree with every field a data leaf, so
    :func:`jax.tree_util.tree_leaves` returns all arrays and :func:`jax.vmap`
    maps over the batch axis. The pixel count and the offsets are not stored
    here; read them from the :class:`SpectroBatchSpec` returned by the builder.
    JIT-, vmap- and grad-compatible.
    """

    wave: jax.Array
    pix_mask: jax.Array
    flux: jax.Array
    sigma: jax.Array
    r_data: jax.Array
    resolution_pp: jax.Array
    z: jax.Array
    cal_lo: jax.Array
    cal_hi: jax.Array
    phot_flux: jax.Array
    phot_err: jax.Array
    phot_presence: jax.Array

    def galaxy(self, i: int) -> SpectroBatch:
        """Return galaxy ``i`` as an unbatched :class:`SpectroBatch`.

        Parameters
        ----------
        i : int
            Index along the batch axis.

        Returns
        -------
        SpectroBatch
            Same fields with the leading batch axis removed.

        Notes
        -----
        Indexing is a plain gather, so it is JIT-compatible when ``i`` is traced.
        """
        return jax.tree_util.tree_map(lambda a: a[i], self)


jax.tree_util.register_dataclass(
    SpectroBatch,
    data_fields=list(_BATCH_FIELDS),
    meta_fields=[],
)


def classify_grid(wave) -> str:
    """Classify a concrete wavelength grid as log-uniform or nonuniform.

    Parameters
    ----------
    wave : array_like, shape (n_pix,)
        Wavelength grid [Angstrom]. Must be concrete (not traced).

    Returns
    -------
    str
        ``"log_uniform"`` if ``diff(log wave)`` is constant to relative
        tolerance ``1e-6``, otherwise ``"nonuniform"``.

    Raises
    ------
    ValueError
        If the grid has fewer than 3 pixels, contains non-finite values, or is
        not strictly increasing. Concatenated DESI camera grids that overlap at
        their seams are not strictly increasing; merge or sort the cameras first.

    Notes
    -----
    Build-time NumPy helper, not JIT-compatible. It must run on concrete values:
    under a trace it would have nothing to inspect (see the module docstring).
    """
    w = np.asarray(wave, dtype=np.float64)
    if w.ndim != 1 or w.size < 3:
        raise ValueError("classify_grid needs a 1-D grid of at least 3 pixels")
    if not np.all(np.isfinite(w)):
        raise ValueError("wavelength grid contains non-finite values")
    if not np.all(np.diff(w) > 0.0):
        raise ValueError(
            "wavelength grid is not strictly increasing; concatenated DESI camera "
            "grids must be merged or sorted by wavelength before batching"
        )
    dln = np.diff(np.log(w))
    spread = np.max(np.abs(dln - dln[0]))
    return "log_uniform" if spread <= _LOG_UNIFORM_RTOL * abs(dln[0]) else "nonuniform"


def pad_banded(bm: BandedMatrix, n_real: int, n_max: int) -> np.ndarray:
    """Pad a banded resolution matrix to ``n_max`` columns, zeroing out-of-range bands.

    Parameters
    ----------
    bm : BandedMatrix
        Operator with ``data`` of shape ``(K, n_real)``.
    n_real : int
        Number of real pixels.
    n_max : int
        Padded pixel count, ``n_max >= n_real``.

    Returns
    -------
    ndarray, shape (K, n_max)
        Copy of ``bm.data`` on the real columns, with every entry
        ``data[k, i]`` whose column ``i + offsets[k]`` falls outside
        ``[0, n_real)`` set to zero. Entries in real rows that would read a
        padded pixel are zeroed as well, and all padded rows are zero.

    Raises
    ------
    ValueError
        If ``n_max < n_real`` or ``data`` does not have ``n_real`` columns.

    Notes
    -----
    Build-time NumPy helper. With this zeroing, a banded matvec on the padded
    vector gives the same real rows as the unpadded operator, whatever the
    padded entries hold. Gradient-safe with respect to the real data.
    """
    offsets = np.asarray(bm.offsets).ravel()
    data = np.asarray(bm.data, dtype=np.float64)
    if n_max < n_real:
        raise ValueError(f"n_max ({n_max}) must be at least n_real ({n_real})")
    if data.shape != (offsets.shape[0], n_real):
        raise ValueError(
            f"banded data has shape {data.shape}, expected {(offsets.shape[0], n_real)}"
        )
    cols = np.arange(n_real)[None, :] + offsets[:, None]
    in_range = (cols >= 0) & (cols < n_real)
    out = np.zeros((offsets.shape[0], n_max))
    out[:, :n_real] = np.where(in_range, data, 0.0)
    return out


def pad_wave(wave, n_max: int) -> np.ndarray:
    """Extend a wavelength grid to ``n_max`` pixels with its final spacing.

    Parameters
    ----------
    wave : array_like, shape (n_pix,)
        Strictly increasing wavelength grid [Angstrom].
    n_max : int
        Target length, ``n_max >= n_pix``.

    Returns
    -------
    ndarray, shape (n_max,)
        ``wave`` followed by ``n_max - n_pix`` points spaced by the last step
        ``wave[-1] - wave[-2]``.

    Raises
    ------
    ValueError
        If the grid has fewer than 2 pixels, is not finite and strictly
        increasing, or ``n_max < n_pix``.

    Notes
    -----
    Build-time NumPy helper. Padded pixels carry a zero mask, so the extension
    only needs to keep the grid valid; it does not affect the likelihood.
    """
    w = np.asarray(wave, dtype=np.float64)
    n = w.shape[0] if w.ndim == 1 else -1
    if n < 2:
        raise ValueError("pad_wave needs a 1-D grid of at least 2 pixels")
    if not (np.all(np.isfinite(w)) and np.all(np.diff(w) > 0.0)):
        raise ValueError("pad_wave needs a finite, strictly increasing grid")
    if n_max < n:
        raise ValueError(f"n_max ({n_max}) must be at least the grid length ({n})")
    step = w[-1] - w[-2]
    tail = w[-1] + step * np.arange(1, n_max - n + 1)
    return np.concatenate([w, tail])


def _prepare_resolution(g: GalaxySpectrum, n: int, n_max: int):
    """Return ``(offsets, r_data, resolution_pp)`` for one galaxy.

    Exactly one of ``r_data`` and ``resolution_pp`` is a zero-size placeholder.
    """
    if isinstance(g.resolution, BandedMatrix):
        offsets = tuple(int(o) for o in np.asarray(g.resolution.offsets).ravel())
        return offsets, pad_banded(g.resolution, n, n_max), np.zeros((0,))
    r = np.broadcast_to(np.asarray(g.resolution, dtype=np.float64), (n,))
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("spectral resolution R must be finite and positive")
    r_pp = np.concatenate([r, np.full(n_max - n, r[-1])])
    return None, np.zeros((0, n_max)), r_pp


def _prepare_photometry(g: GalaxySpectrum):
    """Return ``(has_phot, flux, err, presence)`` with flux and error zero-filled."""
    if g.phot_flux is None:
        if g.phot_err is not None:
            raise ValueError("phot_err was given without phot_flux")
        empty = np.zeros((0,))
        return False, empty, empty, empty
    pf = np.asarray(g.phot_flux, dtype=np.float64)
    if g.phot_err is None:
        raise ValueError("phot_flux was given without phot_err")
    pe = np.asarray(g.phot_err, dtype=np.float64)
    if pf.ndim != 1 or pf.size == 0 or pf.shape != pe.shape:
        raise ValueError("phot_flux and phot_err must be matching non-empty 1-D arrays")
    present = np.isfinite(pf) & np.isfinite(pe) & (pe > 0.0)
    return (
        True,
        np.where(present, pf, 0.0),
        np.where(present, pe, 1.0),
        present.astype(np.float64),
    )


def _prepare_galaxy(index: int, g: GalaxySpectrum, quantum, lsf_n_bins, rest_wave_range):
    """Validate one galaxy and return its bucket spec and unbatched fields."""
    try:
        wave = np.asarray(g.wave, dtype=np.float64)
        if wave.ndim != 1 or wave.size < 3:
            raise ValueError("wave must be 1-D with at least 3 pixels")
        n = wave.size
        flux = np.asarray(g.flux, dtype=np.float64)
        ivar = np.asarray(g.ivar, dtype=np.float64)
        if flux.shape != (n,) or ivar.shape != (n,):
            raise ValueError("flux and ivar must match the wave shape")
        good = np.ones(n, dtype=bool) if g.mask is None else np.asarray(g.mask, dtype=bool)
        if good.shape != (n,):
            raise ValueError("mask must match the wave shape")
        good = good & (ivar > 0.0)

        z = float(g.z)
        if not np.isfinite(z):
            raise ValueError("redshift must be finite")

        n_max = n if quantum is None else -(-n // quantum) * quantum
        offsets, r_data, r_pp = _prepare_resolution(g, n, n_max)
        grid_kind = "banded" if offsets is not None else classify_grid(wave)

        wave_pad = pad_wave(wave, n_max)
        if rest_wave_range is not None:
            lo, hi = rest_wave_range
            rest = wave_pad / (1.0 + z)
            if rest.min() < lo or rest.max() > hi:
                raise ValueError(
                    f"rest-frame grid [{rest.min():.1f}, {rest.max():.1f}] Angstrom "
                    f"leaves rest_wave_range [{lo}, {hi}]"
                )

        has_phot, p_flux, p_err, p_pres = _prepare_photometry(g)
        spec = SpectroBatchSpec(
            n_max=n_max,
            offsets=offsets,
            grid_kind=grid_kind,
            lsf_n_bins=lsf_n_bins,
            has_phot=has_phot,
            n_filt=int(p_flux.shape[0]),
        )
    except ValueError as exc:
        raise ValueError(f"galaxy {index}: {exc}") from exc

    pix_mask = np.zeros(n_max)
    pix_mask[:n] = good
    flux_out = np.zeros(n_max)
    flux_out[:n] = np.where(good, flux, 0.0)
    sigma = np.ones(n_max)
    sigma[:n] = np.where(good, 1.0 / np.sqrt(np.where(good, ivar, 1.0)), 1.0)
    row = {
        "wave": wave_pad,
        "pix_mask": pix_mask,
        "flux": flux_out,
        "sigma": sigma,
        "r_data": r_data,
        "resolution_pp": r_pp,
        "z": np.asarray(z),
        "cal_lo": np.asarray(wave.min()),
        "cal_hi": np.asarray(wave.max()),
        "phot_flux": p_flux,
        "phot_err": p_err,
        "phot_presence": p_pres,
    }
    return spec, row


def _stack_bucket(rows: list[dict]) -> SpectroBatch:
    """Stack unbatched rows along a new leading axis into a :class:`SpectroBatch`."""
    return SpectroBatch(**{k: jnp.asarray(np.stack([r[k] for r in rows])) for k in _BATCH_FIELDS})


def build_spectro_batches(
    galaxies: Sequence[GalaxySpectrum],
    *,
    quantum: int | None = None,
    lsf_n_bins: int = 16,
    rest_wave_range: tuple[float, float] | None = None,
) -> list[tuple[SpectroBatchSpec, SpectroBatch, np.ndarray]]:
    """Group galaxies into static buckets and stack each bucket into a batch.

    Parameters
    ----------
    galaxies : sequence of GalaxySpectrum
        Per-galaxy observations, in catalog order.
    quantum : int, optional
        Round each galaxy's pixel count up to a multiple of ``quantum``, which
        bounds the number of distinct ``n_max`` values and so the number of
        compiles. ``None`` uses each galaxy's exact pixel count.
    lsf_n_bins : int, optional
        Stored in every bucket spec. Default 16.
    rest_wave_range : tuple of float, optional
        ``(lo, hi)`` [Angstrom]. When given, every galaxy's padded grid divided
        by ``1 + z`` must lie inside it, or a ``ValueError`` is raised.

    Returns
    -------
    list of (SpectroBatchSpec, SpectroBatch, ndarray)
        One entry per bucket, in order of first appearance. The ndarray holds
        the catalog indices of that bucket's galaxies, in ascending order.

    Raises
    ------
    ValueError
        If a galaxy is malformed (the message names its index), if its padded
        rest-frame grid leaves ``rest_wave_range``, or if ``quantum`` is not a
        positive integer.

    Notes
    -----
    Build-time NumPy helper, not JIT-compatible; call it once per catalog.
    Galaxies are grouped by :class:`SpectroBatchSpec`, so galaxies with
    different offsets, grid kinds or photometry widths never share a bucket,
    even when their pixel counts match.

    Padded pixels get ``flux = 0``, ``sigma = 1`` and ``pix_mask = 0``. Real
    pixels with ``ivar <= 0`` or a false mask get ``pix_mask = 0``,
    ``sigma = 1`` and ``flux = 0``; the other pixels get
    ``sigma = 1 / sqrt(ivar)``.
    """
    if quantum is not None and (not isinstance(quantum, int) or quantum < 1):
        raise ValueError(f"quantum must be a positive integer or None, got {quantum!r}")

    buckets: dict[SpectroBatchSpec, list[tuple[int, dict]]] = {}
    for i, g in enumerate(galaxies):
        spec, row = _prepare_galaxy(i, g, quantum, lsf_n_bins, rest_wave_range)
        buckets.setdefault(spec, []).append((i, row))

    return [
        (
            spec,
            _stack_bucket([row for _, row in members]),
            np.asarray([i for i, _ in members], dtype=int),
        )
        for spec, members in buckets.items()
    ]


def require_spec(spec) -> SpectroBatchSpec:
    """Return ``spec`` if it is a :class:`SpectroBatchSpec`, else raise.

    Parameters
    ----------
    spec : object
        Candidate bucket spec.

    Returns
    -------
    SpectroBatchSpec
        The same object.

    Raises
    ------
    ValueError
        If ``spec`` is ``None`` or not a :class:`SpectroBatchSpec`.

    Notes
    -----
    Build-time check, intended for the batched forward pass to reject a missing
    static spec before tracing. Not JIT-compatible; it inspects Python types.
    """
    if not isinstance(spec, SpectroBatchSpec):
        raise ValueError(
            f"a SpectroBatchSpec is required (from build_spectro_batches), got {spec!r}"
        )
    return spec
