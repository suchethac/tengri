# SPDX-License-Identifier: BSD-3-Clause
"""DESI spectrum reader: per-camera grids, fluxes, and resolution matrices.

DESI delivers each target as three independently-extracted camera spectra
(b/r/z), each on its own wavelength grid and each with its own resolution
operator stored as a ``(n_diag, n_pix)`` band array (Bolton & Schlegel 2010
[1]_; Guy et al. 2023 [2]_). The cameras **overlap** in wavelength, so merging
them into one sorted grid interleaves pixels whose line-spread functions differ
and destroys the per-pixel correspondence the resolution operator requires.
This module therefore keeps the cameras separate
(:func:`read_desi_cameras`) and concatenates them in *camera order*; never
sorted: so a block-diagonal operator
(:func:`~tengri.observation.banded.block_diagonal_bands`) describes the result
exactly.

References
----------
.. [1] Bolton, A. S. & Schlegel, D. J. 2010, "Spectro-Perfectionism: An
       Algorithmic Framework for Photon Noise-Limited Extraction of Optical
       Fiber Spectroscopy", PASP, 122, 248, arXiv:0911.2689,
       DOI 10.1086/651008.
.. [2] Guy, J. et al. 2023, "The Spectroscopic Data Processing Pipeline for the
       Dark Energy Spectroscopic Instrument", AJ, 165, 144, arXiv:2209.14482,
       DOI 10.3847/1538-3881/acb212.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import numpy as np

from tengri.io.arrays import SpectrumTuple

#: DESI camera names, in the order their grids are concatenated.
DESI_CAMERAS: tuple[str, ...] = ("B", "R", "Z")

# ``10**-17 erg/(s cm2 Angstrom)`` / ``1e-17 ...`` -> the decimal exponent.
_BUNIT_SCALE_RE = re.compile(r"^\s*(?:10\*\*|1e|10\^)\s*([+-]?\d+)", re.IGNORECASE)


class DesiCamera(NamedTuple):
    """One DESI camera's extracted spectrum for a single target.

    Attributes
    ----------
    name : str
        Camera name, one of ``"B"``, ``"R"``, ``"Z"``.
    wave : ndarray, shape (n_pix,)
        Observed-frame vacuum wavelength [Angstrom].
    flux : ndarray, shape (n_pix,)
        Observed flux [erg/s/cm^2/A], already scaled by the ``BUNIT`` the file
        declares.
    flux_err : ndarray, shape (n_pix,)
        1-sigma flux error [erg/s/cm^2/A]; ``NaN`` where ``ivar <= 0``.
    ivar : ndarray, shape (n_pix,)
        Inverse variance in the file's native units.
    mask : ndarray, shape (n_pix,) or None
        Pixel mask, or ``None`` when the file ships no ``_MASK`` HDU.
    resolution : ndarray, shape (n_diag, n_pix) or None
        Resolution band array in desispec ``dia_matrix`` storage, or ``None``
        when the file ships no ``_RESOLUTION`` HDU.
    """

    name: str
    wave: Any
    flux: Any
    flux_err: Any
    ivar: Any
    mask: Any
    resolution: Any


def _import_fits():
    """Import ``astropy.io.fits`` with install guidance on failure."""
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("read_desi requires astropy: pip install astropy") from None
    return fits


def bunit_scale(bunit: str | None) -> float:
    """Decimal scale factor a ``BUNIT`` string declares.

    DESI flux HDUs carry ``BUNIT = '10**-17 erg/(s cm2 Angstrom)'``; the stored
    values must be multiplied by ``1e-17`` to reach the erg/s/cm^2/A that
    :class:`~tengri.io.arrays.SpectrumTuple` documents.

    Parameters
    ----------
    bunit : str or None
        The ``BUNIT`` header value, or ``None`` when absent.

    Returns
    -------
    float
        The multiplicative scale, or ``1.0`` when no leading power of ten is
        declared (an unlabelled file is passed through unchanged rather than
        guessed at).
    """
    if not bunit:
        return 1.0
    match = _BUNIT_SCALE_RE.match(str(bunit))
    return 10.0 ** int(match.group(1)) if match else 1.0


def desi_resolution_offsets(n_diag: int) -> np.ndarray:
    """Diagonal offsets desispec uses for an ``n_diag``-band resolution matrix.

    ``desispec.resolution.Resolution`` defaults to descending offsets
    ``[+n_diag//2, ..., -n_diag//2]`` (Guy et al. 2023 [2]_).

    Parameters
    ----------
    n_diag : int
        Number of stored diagonals (odd).

    Returns
    -------
    ndarray, shape (n_diag,)
        Integer offsets in desispec order.

    References
    ----------
    .. [2] Guy, J. et al. 2023, AJ, 165, 144, arXiv:2209.14482.
    """
    half = int(n_diag) // 2
    return np.arange(half, -half - 1, -1)


def _native(array: np.ndarray) -> np.ndarray:
    """Return ``array`` in native byte order.

    FITS stores big-endian by specification, so astropy hands back ``>f8``.
    JAX refuses any non-native dtype, so every array that can reach a
    :class:`~tengri.observation.banded.BandedMatrix` must be byte-swapped here.
    """
    array = np.asarray(array)
    if array.dtype.byteorder not in ("=", "|"):
        return array.astype(array.dtype.newbyteorder("="))
    return array


def _row_of(array: np.ndarray, row: int, n_pix: int, *, what: str = "spectrum") -> np.ndarray:
    """Select one target's row, tolerating the 1-D single-spectrum layout.

    Raises rather than silently returning the wrong target: a row index a file
    cannot satisfy is a caller error, and returning row 0 for ``row=2`` would
    look like a successful read of the wrong object.
    """
    array = np.asarray(array)
    if array.ndim == 1:
        if row != 0:
            raise ValueError(
                f"row={row} requested but this file holds a single {what} "
                "(the array has no target axis)"
            )
        return array
    if array.shape[-1] != n_pix and array.shape[0] == n_pix:
        # (n_pix, ...) rather than (n_spec, n_pix): no target axis to select.
        return array
    if not 0 <= row < array.shape[0]:
        raise ValueError(f"row={row} is out of range: this file holds {array.shape[0]} spectra")
    return array[row]


def _resolve_row(hdul, targetid: int | None, row: int) -> int:
    """Map a TARGETID onto its FIBERMAP row index."""
    if targetid is None:
        return int(row)
    names = [hdu.name.upper() if hdu.name else "" for hdu in hdul]
    if "FIBERMAP" not in names:
        raise ValueError(
            f"targetid={targetid} requested but the file has no FIBERMAP HDU. "
            f"Available HDUs: {names}. Pass row=<int> to select by index instead."
        )
    fibermap = hdul[names.index("FIBERMAP")].data
    if fibermap is None or "TARGETID" not in (fibermap.dtype.names or ()):
        raise ValueError("FIBERMAP HDU carries no TARGETID column")
    matches = np.flatnonzero(np.asarray(fibermap["TARGETID"]) == targetid)
    if matches.size == 0:
        raise ValueError(f"targetid={targetid} not present in FIBERMAP")
    return int(matches[0])


def _hdu_array(hdul, names: list[str], key: str) -> np.ndarray | None:
    """Fetch an HDU's data by name, or ``None`` when absent."""
    if key not in names:
        return None
    data = hdul[names.index(key)].data
    return None if data is None else np.asarray(data)


def _read_camera(hdul, names: list[str], cam: str, row: int) -> DesiCamera | None:
    """Read one camera's wavelength/flux/ivar/mask/resolution for one target."""
    fits = _import_fits()
    wave_key = f"{cam}_WAVELENGTH"
    if wave_key not in names:
        return None
    wave_hdu = hdul[names.index(wave_key)]

    # Column-table layout (``WAVELENGTH``/``FLUX``/``IVAR`` inside one BinTable).
    if isinstance(wave_hdu, fits.BinTableHDU) and wave_hdu.data.dtype.names:
        cols = wave_hdu.data.dtype.names
        wname = "WAVELENGTH" if "WAVELENGTH" in cols else wave_key
        wave = np.asarray(wave_hdu.data[wname], dtype=np.float64)
        n_pix = wave.shape[0]
        fname = "FLUX" if "FLUX" in cols else f"{cam}_FLUX"
        flux = _row_of(wave_hdu.data[fname], row, n_pix).astype(np.float64)
        iname = "IVAR" if "IVAR" in cols else f"{cam}_IVAR"
        ivar = (
            _row_of(wave_hdu.data[iname], row, n_pix).astype(np.float64) if iname in cols else None
        )
        scale = 1.0
        mask = resolution = None
    else:
        wave = np.asarray(wave_hdu.data, dtype=np.float64).ravel()
        n_pix = wave.shape[0]
        flux_hdu = hdul[names.index(f"{cam}_FLUX")]
        flux = _row_of(flux_hdu.data, row, n_pix).astype(np.float64)
        scale = bunit_scale(flux_hdu.header.get("BUNIT"))
        raw_ivar = _hdu_array(hdul, names, f"{cam}_IVAR")
        ivar = None if raw_ivar is None else _row_of(raw_ivar, row, n_pix).astype(np.float64)
        raw_mask = _hdu_array(hdul, names, f"{cam}_MASK")
        mask = None if raw_mask is None else _native(_row_of(raw_mask, row, n_pix))
        raw_res = _hdu_array(hdul, names, f"{cam}_RESOLUTION")
        # (n_spec, n_diag, n_pix) for a coadd; (n_diag, n_pix) for one spectrum.
        resolution = (
            None
            if raw_res is None
            else (raw_res[row] if raw_res.ndim == 3 else raw_res).astype(np.float64)
        )

    flux = flux * scale
    if ivar is None:
        flux_err = np.full_like(flux, np.nan)
        ivar = np.zeros_like(flux)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_err = np.where(ivar > 0, scale / np.sqrt(ivar), np.nan)
    return DesiCamera(cam, wave, flux, flux_err, ivar, mask, resolution)


def read_desi_cameras(
    path: str,
    *,
    targetid: int | None = None,
    row: int = 0,
    cameras: tuple[str, ...] = DESI_CAMERAS,
) -> tuple[DesiCamera, ...]:
    """Read a DESI coadd's per-camera spectra for one target.

    Each camera is returned on its own wavelength grid with its own resolution
    band array, because the b/r/z grids overlap and a merged grid admits no
    single per-pixel resolution operator (Guy et al. 2023 [2]_).

    Parameters
    ----------
    path : str
        Path to a DESI FITS file (``coadd-*.fits``, ``spectra-*.fits``).
    targetid : int, optional
        Select the target by ``FIBERMAP`` TARGETID. Mutually exclusive with
        ``row`` in intent; when given, ``row`` is ignored.
    row : int, optional
        Zero-based spectrum index when ``targetid`` is not given. Default 0.
    cameras : tuple of str, optional
        Camera names to read, in concatenation order. Default ``("B", "R", "Z")``.

    Returns
    -------
    tuple of DesiCamera
        One entry per camera actually present in the file, in the requested
        order.

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If no requested camera is present, or ``targetid`` cannot be resolved.

    Notes
    -----
    **JIT-compatible**: no, file I/O.

    Fluxes are scaled by the ``BUNIT`` the FLUX HDU declares (DESI ships
    ``10**-17 erg/(s cm2 Angstrom)``), so the returned arrays are in
    erg/s/cm^2/A. A file that declares no ``BUNIT`` is passed through unscaled.

    References
    ----------
    .. [2] Guy, J. et al. 2023, AJ, 165, 144, arXiv:2209.14482.

    Examples
    --------
    >>> cams = read_desi_cameras("coadd-sv1-bright-1234.fits", targetid=39627)
    >>> [(c.name, c.wave.shape, c.resolution.shape) for c in cams]
    """
    fits = _import_fits()
    with fits.open(path) as hdul:
        names = [hdu.name.upper() if hdu.name else f"HDU{i}" for i, hdu in enumerate(hdul)]
        index = _resolve_row(hdul, targetid, row)
        out = [_read_camera(hdul, names, cam.upper(), index) for cam in cameras]
    found = tuple(cam for cam in out if cam is not None)
    if not found:
        raise ValueError(
            f"No per-camera wavelength HDUs among {[f'{c}_WAVELENGTH' for c in cameras]}. "
            f"Available HDUs: {names}"
        )
    return found


def desi_resolution_matrix(cameras: tuple[DesiCamera, ...]):
    """Block-diagonal resolution operator over the concatenated camera grids.

    Parameters
    ----------
    cameras : tuple of DesiCamera
        Cameras in the same order their grids are concatenated.

    Returns
    -------
    BandedMatrix
        Operator of width ``sum(n_pix)``, block diagonal by camera.

    Raises
    ------
    ValueError
        If any camera carries no resolution data; a partial operator would
        silently apply no LSF to the cameras that lack one.

    Notes
    -----
    Build-time helper. See
    :func:`~tengri.observation.banded.block_diagonal_bands`.
    """
    from tengri.observation.banded import block_diagonal_bands, resolution_bands_from_desi

    missing = [cam.name for cam in cameras if cam.resolution is None]
    if missing:
        raise ValueError(
            f"cameras {missing} carry no _RESOLUTION HDU, so a block-diagonal operator "
            "would leave them unconvolved. Read a file with resolution data, or build a "
            "Gaussian operator per camera with "
            "tengri.observation.banded.gaussian_resolution_bands."
        )
    blocks = [
        resolution_bands_from_desi(
            np.asarray(cam.resolution),
            desi_resolution_offsets(np.asarray(cam.resolution).shape[0]),
        )
        for cam in cameras
    ]
    return block_diagonal_bands(blocks)


def desi_spectroscopy(cameras: tuple[DesiCamera, ...], **kwargs):
    """Build a :class:`~tengri.observation.spectroscopy.Spectroscopy` from cameras.

    The wavelength grid is the camera grids concatenated **in camera order**
    (never sorted), and the instrument response is the block-diagonal resolution
    operator over that grid; which replaces the Gaussian ``apply_lsf`` in
    projection (#1163).

    Parameters
    ----------
    cameras : tuple of DesiCamera
        As returned by :func:`read_desi_cameras`.
    **kwargs
        Forwarded to :class:`~tengri.observation.spectroscopy.Spectroscopy`.

    Returns
    -------
    Spectroscopy
        Instrument model on the concatenated grid.

    Raises
    ------
    ValueError
        If a flux-conserving ``resample`` mode is requested on a grid whose
        cameras overlap; the bin-integral resampler needs a strictly
        increasing grid, and the overlap makes it decrease at the seams.

    Notes
    -----
    Build-time helper. The matching flux/error vectors are the same
    concatenation, which :func:`read_desi` returns.
    """
    from tengri.observation.spectroscopy import Spectroscopy

    wave = np.concatenate([np.asarray(cam.wave) for cam in cameras])
    resample = kwargs.get("resample", "point")
    if resample != "point" and np.any(np.diff(wave) <= 0.0):
        seams = [cam.name for cam in cameras]
        raise ValueError(
            f"resample={resample!r} needs a strictly increasing grid, but the "
            f"concatenated {'/'.join(seams)} grid decreases where the cameras overlap. "
            "Use resample='point' (the DESI resolution matrix already encodes the LSF "
            "at pixel resolution), or read a single camera with "
            "read_desi_cameras(..., cameras=('B',))."
        )
    return Spectroscopy(
        wave_obs=wave,
        resolution_matrix=desi_resolution_matrix(cameras),
        **kwargs,
    )


def read_desi(
    path: str,
    *,
    targetid: int | None = None,
    row: int = 0,
) -> SpectrumTuple:
    """Read a DESI coadd or healpix spectrum for one target.

    Reads the BRZ-combined HDU when present, otherwise the per-camera (b/r/z)
    HDUs concatenated **in camera order**. The grid is not re-sorted: DESI's
    cameras overlap, and sorting interleaves pixels from cameras with different
    line-spread functions, leaving a grid no resolution operator describes.

    Parameters
    ----------
    path : str
        Path to a DESI FITS file (e.g. ``coadd-*.fits``).
    targetid : int, optional
        Select the target by ``FIBERMAP`` TARGETID.
    row : int, optional
        Zero-based spectrum index when ``targetid`` is not given. Default 0.

    Returns
    -------
    SpectrumTuple
        ``(wave, flux, flux_err, meta)`` in erg/s/cm^2/A on the camera-order
        grid. ``meta`` carries ``instrument`` (``"DESI"``), ``redshift`` (if the
        primary header has ``Z``), ``header_keys``, ``cameras`` (the camera
        names read), ``n_pix_per_camera``, and ``resolution`` (per-camera
        ``(n_diag, n_pix)`` band arrays, or ``None`` when the file ships none).

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If neither combined nor per-camera HDUs are found.

    Notes
    -----
    **JIT-compatible**: no; file I/O and astropy required.

    Fluxes are scaled by the declared ``BUNIT`` (DESI: ``10**-17 erg/(s cm2
    Angstrom)``). Files that declare no ``BUNIT`` pass through unscaled.

    Pair with :func:`desi_spectroscopy` to obtain the matching instrument model
    with the block-diagonal resolution operator on this same grid.

    Examples
    --------
    >>> wave, flux, err, meta = read_desi("coadd-123-45.fits", targetid=39627)
    >>> meta["cameras"]
    ('B', 'R', 'Z')
    """
    fits = _import_fits()

    with fits.open(path) as hdul:
        names = [hdu.name.upper() if hdu.name else f"HDU{i}" for i, hdu in enumerate(hdul)]
        primary_header = hdul[0].header
        index = _resolve_row(hdul, targetid, row)

        if "BRZ_WAVELENGTH" in names:
            cameras = (_read_camera(hdul, names, "BRZ", index),)
            if cameras[0] is None:  # pragma: no cover - guarded by the name check
                raise ValueError(f"BRZ_WAVELENGTH present but unreadable. HDUs: {names}")
        else:
            cameras = tuple(
                cam
                for cam in (_read_camera(hdul, names, c, index) for c in DESI_CAMERAS)
                if cam is not None
            )

    if not cameras:
        raise ValueError(f"Could not find BRZ or per-camera wavelength HDUs. Available: {names}")

    wave = np.concatenate([cam.wave for cam in cameras])
    flux = np.concatenate([cam.flux for cam in cameras])
    flux_err = np.concatenate([cam.flux_err for cam in cameras])

    meta = {
        "instrument": "DESI",
        "redshift": primary_header.get("Z", None),
        "header_keys": dict(primary_header),
        "cameras": tuple(cam.name for cam in cameras),
        "n_pix_per_camera": tuple(int(cam.wave.shape[0]) for cam in cameras),
        "resolution": tuple(cam.resolution for cam in cameras),
    }
    return SpectrumTuple(wave, flux, flux_err, meta)
