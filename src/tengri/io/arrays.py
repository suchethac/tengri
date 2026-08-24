# SPDX-License-Identifier: BSD-3-Clause
"""Core data structure for spectroscopic I/O."""

from __future__ import annotations

from typing import Any, NamedTuple


class SpectrumTuple(NamedTuple):
    """Immutable tuple holding a loaded spectrum with metadata.

    All flux values are in erg/s/cm^2/A. Wavelength is in Angstrom.

    Attributes
    ----------
    wave: array_like, shape (n_wave,)
        Vacuum wavelength [Angstrom].
    flux: array_like, shape (n_wave,)
        Observed-frame flux [erg/s/cm^2/A].
    flux_err: array_like, shape (n_wave,)
        1-sigma error on flux [erg/s/cm^2/A]. May contain NaN for
        flagged/invalid pixels.
    meta: dict
        Metadata dictionary. Common keys: ``redshift`` (float or None),
        ``instrument`` (str), ``header_keys`` (dict of original FITS
        keywords). Application-specific loaders may add other keys.

    Notes
    -----
    SpectrumTuple is a NamedTuple, so it is fully tuple-compatible:
    you can unpack it as ``(wave, flux, err, meta) = spectrum``,
    iterate, slice, or use positional indexing. It is also immutable.

    Examples
    --------
    >>> spectrum = SpectrumTuple(wave, flux, flux_err, {"redshift": 0.5})
    >>> wave, flux, err, meta = spectrum  # unpack
    >>> z = spectrum.meta["redshift"]  # named access
    """

    wave: Any
    flux: Any
    flux_err: Any
    meta: dict
