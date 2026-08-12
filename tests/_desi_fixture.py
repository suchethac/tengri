# SPDX-License-Identifier: BSD-3-Clause
"""Write a DESI-format coadd FITS for tests (#1183).

The fixture reproduces the parts of the real ``coadd-*.fits`` layout that the
loader has to get right, and that a hand-rolled NumPy fixture cannot exercise:

* **per-camera HDUs** ``{B,R,Z}_{WAVELENGTH,FLUX,IVAR,MASK,RESOLUTION}``;
* **overlapping cameras** — the real b/r and r/z coverage overlaps, which is
  what makes "concatenate then sort" observably different from "concatenate in
  camera order";
* ``FLUX`` shaped ``(n_spec, n_pix)`` and ``RESOLUTION`` shaped
  ``(n_spec, n_diag, n_pix)``, so target selection is exercised;
* a declared ``BUNIT`` of ``10**-17 erg/(s cm2 Angstrom)``;
* **big-endian storage**, which FITS mandates — the byte order JAX rejects.

No committed data: every byte is generated at test time.

References
----------
.. [1] Guy, J. et al. 2023, "The Spectroscopic Data Processing Pipeline for the
       Dark Energy Spectroscopic Instrument", AJ, 165, 144, arXiv:2209.14482.
"""

from __future__ import annotations

import numpy as np

#: Real DESI camera coverage [Angstrom] (Guy et al. 2023, Table 1). b/r and r/z
#: overlap — that overlap is the point of the fixture.
DESI_CAMERA_RANGES: dict[str, tuple[float, float]] = {
    "B": (3600.0, 5800.0),
    "R": (5760.0, 7620.0),
    "Z": (7520.0, 9824.0),
}

#: The ``BUNIT`` DESI flux HDUs declare.
DESI_FLUX_BUNIT = "10**-17 erg/(s cm2 Angstrom)"


def gaussian_dia_resolution(n_pix: int, n_diag: int, sigma_pix: float) -> np.ndarray:
    """Row-normalized Gaussian LSF in desispec ``dia_matrix`` storage.

    Builds the dense operator ``A[i, j]`` first — a Gaussian in ``j - i``,
    normalized so every row sums to one — then converts to the storage desispec
    ships, ``data[k, j] = A[j - offsets[k], j]``.

    Parameters
    ----------
    n_pix : int
        Number of pixels.
    n_diag : int
        Number of stored diagonals (odd).
    sigma_pix : float
        Gaussian sigma [pixels].

    Returns
    -------
    ndarray, shape (n_diag, n_pix)
        Band array in desispec storage.

    Notes
    -----
    Rows within ``n_diag // 2`` of either edge are truncated by the band limit,
    so they sum to less than one — exactly as a delivered DESI matrix does.
    """
    half = n_diag // 2
    offsets = np.arange(half, -half - 1, -1)
    i = np.arange(n_pix)[:, None]
    j = np.arange(n_pix)[None, :]
    dense = np.exp(-0.5 * ((j - i) / sigma_pix) ** 2)
    dense[np.abs(j - i) > half] = 0.0
    dense /= dense.sum(axis=1, keepdims=True)

    data = np.zeros((offsets.shape[0], n_pix))
    for k, offset in enumerate(offsets):
        rows = np.arange(n_pix) - offset  # A[j - offset, j]
        valid = (rows >= 0) & (rows < n_pix)
        data[k, valid] = dense[rows[valid], np.arange(n_pix)[valid]]
    return data


def write_desi_coadd(
    path,
    *,
    n_pix: int = 48,
    n_diag: int = 5,
    targetids: tuple[int, ...] = (1001, 1002, 1003),
    sigma_pix: float = 1.1,
    with_resolution: bool = True,
    with_bunit: bool = True,
    with_fibermap: bool = True,
    cameras: tuple[str, ...] = ("B", "R", "Z"),
) -> dict:
    """Write a DESI-format coadd and return the values it was built from.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination FITS path.
    n_pix : int, optional
        Pixels per camera. Default 48.
    n_diag : int, optional
        Resolution diagonals (odd). Default 5.
    targetids : tuple of int, optional
        TARGETIDs written to FIBERMAP; target ``k`` gets a constant raw flux of
        ``k + 1``, so a row can be identified from its value alone.
    sigma_pix : float, optional
        LSF sigma [pixels] for the generated resolution matrices. Default 1.1.
    with_resolution, with_bunit, with_fibermap : bool, optional
        Toggle each optional part of the layout, for tests that need it absent.
    cameras : tuple of str, optional
        Cameras to write. Default ``("B", "R", "Z")``.

    Returns
    -------
    dict
        ``wave`` (per camera), ``resolution`` (per camera, dia storage),
        ``targetids``, ``flux_scale`` (the BUNIT factor, ``1.0`` when absent),
        and ``raw_flux`` (per target).
    """
    from astropy.io import fits

    hdus = [fits.PrimaryHDU()]
    if with_fibermap:
        hdus.append(
            fits.BinTableHDU.from_columns(
                [fits.Column(name="TARGETID", format="K", array=np.asarray(targetids))],
                name="FIBERMAP",
            )
        )

    n_spec = len(targetids)
    waves: dict[str, np.ndarray] = {}
    resolutions: dict[str, np.ndarray] = {}

    for cam in cameras:
        lo, hi = DESI_CAMERA_RANGES[cam]
        wave = np.linspace(lo, hi, n_pix)
        flux = np.stack([np.full(n_pix, float(k + 1)) for k in range(n_spec)])
        ivar = np.full((n_spec, n_pix), 4.0)
        mask = np.zeros((n_spec, n_pix), dtype=np.int32)

        waves[cam] = wave
        wave_hdu = fits.ImageHDU(wave, name=f"{cam}_WAVELENGTH")
        wave_hdu.header["BUNIT"] = "Angstrom"
        flux_hdu = fits.ImageHDU(flux, name=f"{cam}_FLUX")
        if with_bunit:
            flux_hdu.header["BUNIT"] = DESI_FLUX_BUNIT
        hdus += [
            wave_hdu,
            flux_hdu,
            fits.ImageHDU(ivar, name=f"{cam}_IVAR"),
            fits.ImageHDU(mask, name=f"{cam}_MASK"),
        ]
        if with_resolution:
            band = gaussian_dia_resolution(n_pix, n_diag, sigma_pix)
            resolutions[cam] = band
            hdus.append(
                fits.ImageHDU(
                    np.broadcast_to(band, (n_spec, *band.shape)).copy(), name=f"{cam}_RESOLUTION"
                )
            )

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return {
        "wave": waves,
        "resolution": resolutions,
        "targetids": tuple(targetids),
        "flux_scale": 1e-17 if with_bunit else 1.0,
        "raw_flux": tuple(float(k + 1) for k in range(n_spec)),
        "n_pix": n_pix,
        "n_diag": n_diag,
        "cameras": tuple(cameras),
    }
