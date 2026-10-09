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
from tengri.observation.spectroscopy import resolve_resample_mode
from tengri.utils.conversions import flambda_to_fnu

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

    The flux is spectral flux density per frequency, the basis of the model
    prediction (:meth:`~tengri.observation.observation.Observation.predict`).
    Use :meth:`from_flambda` for data published per wavelength.

    Attributes
    ----------
    wave : array_like, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom], strictly increasing.
    flux : array_like, shape (n_pix,)
        Observed spectral flux density F_nu [erg/s/cm^2/Hz].
    ivar : array_like, shape (n_pix,)
        Inverse variance of ``flux`` [(erg/s/cm^2/Hz)^-2]. Pixels whose ``ivar``
        is not finite or not positive, or whose ``flux`` is not finite, are
        excluded.
    resolution : BandedMatrix, float or array_like, shape (n_pix,)
        Resolution operator: a banded matrix over the pixel grid, or a
        spectral resolution ``R = lambda / delta lambda`` (scalar or per pixel).
    z : float
        Redshift of the galaxy.
    mask : array_like of bool, shape (n_pix,), optional
        Pixel mask, ``True`` for good pixels. ``None`` means all pixels are good.
    phot_flux : array_like, shape (n_filt,), optional
        Broadband flux densities F_nu [erg/s/cm^2/Hz], one per filter. A filter
        with a non-finite flux or error is excluded.
    phot_err : array_like, shape (n_filt,), optional
        1-sigma errors on ``phot_flux`` [erg/s/cm^2/Hz]. Required with ``phot_flux``.
    """

    wave: Any
    flux: Any
    ivar: Any
    resolution: Any
    z: float
    mask: Any = None
    phot_flux: Any = None
    phot_err: Any = None

    @classmethod
    def from_flambda(
        cls,
        wave,
        flux_lambda,
        ivar_lambda,
        *,
        resolution,
        z: float,
        mask=None,
        phot_flux_lambda=None,
        phot_err_lambda=None,
        phot_wave_eff=None,
    ) -> GalaxySpectrum:
        """Build a record from data per wavelength, converting to F_nu.

        Parameters
        ----------
        wave : array_like, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        flux_lambda : array_like, shape (n_pix,)
            Observed flux F_lambda [erg/s/cm^2/Angstrom].
        ivar_lambda : array_like, shape (n_pix,)
            Inverse variance of ``flux_lambda`` [(erg/s/cm^2/Angstrom)^-2].
        resolution : BandedMatrix, float or array_like
            Resolution operator, as in :class:`GalaxySpectrum`.
        z : float
            Redshift of the galaxy.
        mask : array_like of bool, optional
            Pixel mask, as in :class:`GalaxySpectrum`.
        phot_flux_lambda, phot_err_lambda : array_like, shape (n_filt,), optional
            Broadband fluxes and errors per wavelength [erg/s/cm^2/Angstrom].
            Requires ``phot_wave_eff``.
        phot_wave_eff : array_like, shape (n_filt,), optional
            Effective wavelength of each filter [Angstrom], the wavelength at
            which the per-wavelength photometry is converted.

        Returns
        -------
        GalaxySpectrum
            Record with ``flux`` and ``ivar`` in F_nu units.

        Raises
        ------
        ValueError
            If the photometry is given without ``phot_wave_eff``, or only partly.

        Notes
        -----
        Uses :func:`tengri.utils.conversions.flambda_to_fnu`, so F_nu = F_lambda
        lambda^2 / c with c from :mod:`tengri.utils.physics_constants`. Errors
        scale the same way, and an inverse variance divides by the square of
        that factor: ivar_nu = ivar_lambda / (lambda^2 / c)^2.
        """
        w = np.asarray(wave, dtype=np.float64)
        scale = np.asarray(flambda_to_fnu(np.ones_like(w), w), dtype=np.float64)
        flux_nu = np.asarray(flambda_to_fnu(np.asarray(flux_lambda, dtype=np.float64), w))
        ivar_nu = np.asarray(ivar_lambda, dtype=np.float64) / scale**2
        phot_flux = phot_err = None
        if phot_flux_lambda is not None or phot_err_lambda is not None:
            if phot_flux_lambda is None or phot_err_lambda is None or phot_wave_eff is None:
                raise ValueError(
                    "photometry per wavelength needs phot_flux_lambda, phot_err_lambda "
                    "and phot_wave_eff together"
                )
            pw = np.asarray(phot_wave_eff, dtype=np.float64)
            phot_flux = np.asarray(
                flambda_to_fnu(np.asarray(phot_flux_lambda, dtype=np.float64), pw)
            )
            phot_err = np.asarray(
                flambda_to_fnu(np.asarray(phot_err_lambda, dtype=np.float64), pw)
            )
        return cls(
            wave=w,
            flux=flux_nu,
            ivar=ivar_nu,
            resolution=resolution,
            z=z,
            mask=mask,
            phot_flux=phot_flux,
            phot_err=phot_err,
        )


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
    conserving : bool or None
        Flux-conserving pixel integral, resolved from the template's ``resample``
        mode for this bucket's grid and redshift. ``None`` when the bucket was
        built without a template model (see :func:`build_spectro_batches`); such
        a bucket cannot be evaluated.

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
    conserving: bool | None = None

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
        if self.conserving is not None and not isinstance(self.conserving, bool):
            raise ValueError(f"conserving must be a bool or None, got {self.conserving!r}")


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
        Observed spectral flux density F_nu [erg/s/cm^2/Hz]; 0.0 on excluded pixels.
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


def pad_wave(wave, n_max: int, grid_kind: str = "nonuniform") -> np.ndarray:
    """Extend a wavelength grid to ``n_max`` pixels, keeping its spacing law.

    Parameters
    ----------
    wave : array_like, shape (n_pix,)
        Strictly increasing wavelength grid [Angstrom].
    n_max : int
        Target length, ``n_max >= n_pix``.
    grid_kind : {"nonuniform", "log_uniform"}, optional
        Spacing law to continue. ``"log_uniform"`` extends geometrically, with the
        last constant step ``d ln(lambda)``, so the padded grid stays log-uniform
        and the single-FFT path stays valid. Any other value extends linearly
        with the last step ``lambda[-1] - lambda[-2]``. Default ``"nonuniform"``.

    Returns
    -------
    ndarray, shape (n_max,)
        ``wave`` followed by ``n_max - n_pix`` extension points.

    Raises
    ------
    ValueError
        If the grid has fewer than 2 pixels, is not finite and strictly
        increasing, ``n_max < n_pix``, or ``grid_kind`` is unknown.

    Notes
    -----
    Build-time NumPy helper. Padded pixels carry a zero mask. Whether the
    extension changes the likelihood is decided by :func:`padding_exact`, not
    here: a flux-conserving pixel integral takes the upper edge of the last real
    pixel from its padded neighbour, which only a linear extension leaves
    unchanged.
    """
    if grid_kind not in ("nonuniform", "log_uniform"):
        raise ValueError(f"grid_kind must be 'nonuniform' or 'log_uniform', got {grid_kind!r}")
    w = np.asarray(wave, dtype=np.float64)
    n = w.shape[0] if w.ndim == 1 else -1
    if n < 2:
        raise ValueError("pad_wave needs a 1-D grid of at least 2 pixels")
    if not (np.all(np.isfinite(w)) and np.all(np.diff(w) > 0.0)):
        raise ValueError("pad_wave needs a finite, strictly increasing grid")
    if n_max < n:
        raise ValueError(f"n_max ({n_max}) must be at least the grid length ({n})")
    k = np.arange(1, n_max - n + 1)
    if grid_kind == "log_uniform":
        dln = np.log(w[-1]) - np.log(w[-2])
        tail = w[-1] * np.exp(dln * k)
    else:
        tail = w[-1] + (w[-1] - w[-2]) * k
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


@dataclass(frozen=True)
class TemplatePolicy:
    """What a bucket builder needs to know about the template model.

    Attributes
    ----------
    lsf_n_bins : int
        The template's spectral LSF bin count.
    resample : str
        The template's ``resample`` mode, ``"point"``, ``"conserving"`` or ``"auto"``.
    rest_wave : ndarray, shape (n_wave,)
        The template's rest-frame model grid [Angstrom], used by ``"auto"``.
    sigma_v_zero : bool
        True when ``sigma_v_kms`` is absent or fixed at exactly 0, so the galaxy
        velocity broadening is the identity.
    igm : bool
        Whether the template applies an IGM transmission.
    n_filters : int or None
        Photometric filter count of the template, ``None`` without photometry.
    """

    lsf_n_bins: int
    resample: str
    rest_wave: np.ndarray
    sigma_v_zero: bool
    igm: bool
    n_filters: int | None


def template_policy(model) -> TemplatePolicy:
    """Read a :class:`TemplatePolicy` from a template model.

    Parameters
    ----------
    model : SEDModel
        Template with spectroscopy configured.

    Returns
    -------
    TemplatePolicy

    Raises
    ------
    ValueError
        If the model has no spectroscopy.

    Notes
    -----
    Build-time Python helper. Reads model attributes by name, so the observation
    layer does not import the forward model.
    """
    observation = getattr(model, "observation", None)
    spectroscopy = getattr(observation, "spectroscopy", None) if observation else None
    if spectroscopy is None:
        raise ValueError("template model has no spectroscopy configured")
    free = set(model.spec.free_params)
    sigma_v_zero = "sigma_v_kms" not in free and float(model._get_sigma_v_kms({})) == 0.0
    photometry = getattr(observation, "photometry", None)
    return TemplatePolicy(
        lsf_n_bins=int(model._lsf_n_bins),
        resample=str(spectroscopy.resample),
        rest_wave=np.asarray(model.wavelengths, dtype=np.float64),
        sigma_v_zero=bool(sigma_v_zero),
        igm=bool(model._uses_igm),
        n_filters=None if photometry is None else int(photometry.n_filters),
    )


def padding_exact(policy: TemplatePolicy, spec: SpectroBatchSpec, log_grid: bool):
    """Decide whether zero-masked padding leaves the real pixels unchanged.

    Parameters
    ----------
    policy : TemplatePolicy
        Template policy the bucket was built against.
    spec : SpectroBatchSpec
        Bucket key, with ``offsets`` and ``conserving`` resolved.
    log_grid : bool
        Whether this galaxy's own wavelength grid is log-uniform (its padding
        is then geometric).

    Returns
    -------
    exact : bool
        True when every real pixel's value is the same padded or unpadded.
    reason : str
        Empty when ``exact``; otherwise why padding changes the real pixels.

    Notes
    -----
    The rule follows the spectral operator in
    :func:`~tengri.observation.observation.project_spectrum_kernel_split`:

    * Banded resolution, point sampling: the matrix zeroes every band that
      reaches a padded pixel, so padding is exact when the galaxy broadening
      does not run on the observed grid: ``sigma_v`` fixed at 0, or IGM on (then
      ``sigma_v`` acts on the rest grid before resampling).
    * Gaussian resolution, flux-conserving: the LSF acts on the rest grid
      (:func:`~tengri.observation.spectrum.compute_spectrum_conserving_lsf`), so
      ``sigma_v`` is never on the observed grid. Point sampling applies the
      Gaussian LSF to the observed grid, so it is not exact.
    * Flux-conserving sampling takes the upper edge of the last real pixel from
      its padded neighbour. A linear extension reproduces the edge exactly; a
      geometric one does not, so a log-uniform grid is refused under it.
    """
    if spec.offsets is not None:
        if not (policy.sigma_v_zero or policy.igm):
            return False, (
                "sigma_v is free or fixed nonzero and the template has no IGM, so the "
                "galaxy broadening runs on the padded observed grid"
            )
        if spec.conserving and log_grid:
            return False, (
                "the flux-conserving pixel edge of the last real pixel moves when a "
                "log-uniform grid is padded geometrically"
            )
        return True, ""
    if not spec.conserving:
        return False, (
            "a Gaussian LSF on the observed grid with point sampling reads the padded pixels"
        )
    if log_grid:
        return False, (
            "the flux-conserving pixel edge of the last real pixel moves when a "
            "log-uniform grid is padded geometrically"
        )
    return True, ""


def _prepare_galaxy(index: int, g: GalaxySpectrum, quantum, lsf_n_bins, rest_wave_range, policy):
    """Validate one galaxy and return ``(spec, row, n_real, log_grid)``."""
    try:
        wave = np.asarray(g.wave, dtype=np.float64)
        if wave.ndim != 1 or wave.size < 3:
            raise ValueError("wave must be 1-D with at least 3 pixels")
        if not np.all(np.isfinite(wave)):
            raise ValueError("wave contains non-finite values")
        n = wave.size
        flux = np.asarray(g.flux, dtype=np.float64)
        ivar = np.asarray(g.ivar, dtype=np.float64)
        if flux.shape != (n,) or ivar.shape != (n,):
            raise ValueError("flux and ivar must match the wave shape")
        good = np.ones(n, dtype=bool) if g.mask is None else np.asarray(g.mask, dtype=bool)
        if good.shape != (n,):
            raise ValueError("mask must match the wave shape")
        good = good & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0.0)

        z = float(g.z)
        if not np.isfinite(z):
            raise ValueError("redshift must be finite")

        n_max = n if quantum is None else -(-n // quantum) * quantum
        offsets, r_data, r_pp = _prepare_resolution(g, n, n_max)
        kind = classify_grid(wave)
        log_grid = kind == "log_uniform"
        grid_kind = "banded" if offsets is not None else kind
        conserving = None
        if policy is not None:
            conserving = resolve_resample_mode(policy.resample, wave, policy.rest_wave, z)

        wave_pad = pad_wave(wave, n_max, "log_uniform" if log_grid else "nonuniform")
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
            conserving=conserving,
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
    return spec, row, n, log_grid


def _stack_bucket(rows: list[dict]) -> SpectroBatch:
    """Stack unbatched rows along a new leading axis into a :class:`SpectroBatch`."""
    return SpectroBatch(**{k: jnp.asarray(np.stack([r[k] for r in rows])) for k in _BATCH_FIELDS})


def _padding_error(bucket_spec, offenders) -> ValueError:
    """Build the refusal for padded galaxies whose padding would change real pixels."""
    listed = "; ".join(
        f"galaxy {i} (n_real={n}, n_max={bucket_spec.n_max})" for i, n, _ in offenders
    )
    reason = offenders[0][2]
    return ValueError(
        f"padding would change real pixels for {listed}: {reason}. Use quantum=None "
        "(buckets by exact n_pix, zero padding), or fix sigma_v at 0 or give the "
        "template an IGM."
    )


def build_spectro_batches(
    galaxies: Sequence[GalaxySpectrum],
    *,
    quantum: int | None = None,
    lsf_n_bins: int | None = None,
    rest_wave_range: tuple[float, float] | None = None,
    model=None,
) -> list[tuple[SpectroBatchSpec, SpectroBatch, np.ndarray]]:
    """Group galaxies into static buckets and stack each bucket into a batch.

    Parameters
    ----------
    galaxies : sequence of GalaxySpectrum
        Per-galaxy observations, in catalog order.
    quantum : int, optional
        Round each galaxy's pixel count up to a multiple of ``quantum``, which
        bounds the number of distinct ``n_max`` values and so the number of
        compiles. ``None`` uses each galaxy's exact pixel count (no padding).
    lsf_n_bins : int, optional
        LSF bin count stored in every bucket spec. Default: the template's
        ``lsf_n_bins`` when ``model`` is given, else 16. An explicit value that
        disagrees with the template raises.
    rest_wave_range : tuple of float, optional
        ``(lo, hi)`` [Angstrom]. When given, every galaxy's padded grid divided
        by ``1 + z`` must lie inside it, or a ``ValueError`` is raised.
    model : SEDModel, optional
        Template model. When given, each galaxy's flux-conserving flag is resolved
        from the template's ``resample`` mode with the same rule as
        :meth:`~tengri.observation.observation.Observation.predict`, and every
        padded bucket is checked with :func:`padding_exact`. Without it the
        buckets carry ``conserving=None`` and cannot be evaluated.

    Returns
    -------
    list of (SpectroBatchSpec, SpectroBatch, ndarray)
        One entry per bucket, in order of first appearance. The ndarray holds
        the catalog indices of that bucket's galaxies, in ascending order.

    Raises
    ------
    ValueError
        If a galaxy is malformed (the message names its index), if its padded
        rest-frame grid leaves ``rest_wave_range``, if ``quantum`` is not a
        positive integer, if ``lsf_n_bins`` disagrees with the template, or if
        padding would change the real pixels of a galaxy (the message names the
        galaxies, their n_real and n_max, the reason, and the remedy).

    Notes
    -----
    Build-time NumPy helper, not JIT-compatible; call it once per catalog.
    Galaxies are grouped by :class:`SpectroBatchSpec`, so galaxies with
    different offsets, grid kinds, photometry widths or resample modes never share
    a bucket, even when their pixel counts match.

    Padded pixels get ``flux = 0``, ``sigma = 1`` and ``pix_mask = 0``. Real
    pixels with a non-finite flux or ivar, ``ivar <= 0`` or a false mask get
    ``pix_mask = 0``, ``sigma = 1`` and ``flux = 0``; the other pixels get
    ``sigma = 1 / sqrt(ivar)``.
    """
    if quantum is not None and (not isinstance(quantum, int) or quantum < 1):
        raise ValueError(f"quantum must be a positive integer or None, got {quantum!r}")
    policy = None
    if model is not None:
        policy = template_policy(model)
        if lsf_n_bins is not None and lsf_n_bins != policy.lsf_n_bins:
            raise ValueError(
                f"lsf_n_bins={lsf_n_bins} disagrees with the template model "
                f"({policy.lsf_n_bins}); omit it to take the template's value"
            )
        lsf_n_bins = policy.lsf_n_bins
    elif lsf_n_bins is None:
        lsf_n_bins = 16

    buckets: dict[SpectroBatchSpec, list[tuple[int, dict, int, bool]]] = {}
    for i, g in enumerate(galaxies):
        spec, row, n_real, log_grid = _prepare_galaxy(
            i, g, quantum, lsf_n_bins, rest_wave_range, policy
        )
        buckets.setdefault(spec, []).append((i, row, n_real, log_grid))

    if policy is not None:
        for spec, members in buckets.items():
            offenders = []
            for i, _, n_real, log_grid in members:
                if n_real < spec.n_max:
                    exact, reason = padding_exact(policy, spec, log_grid)
                    if not exact:
                        offenders.append((i, n_real, reason))
            if offenders:
                raise _padding_error(spec, offenders)

    return [
        (
            spec,
            _stack_bucket([row for _, row, _, _ in members]),
            np.asarray([i for i, _, _, _ in members], dtype=int),
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
