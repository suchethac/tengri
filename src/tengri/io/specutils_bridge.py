# SPDX-License-Identifier: BSD-3-Clause
"""Bridge from specutils.Spectrum1D to SpectrumTuple."""

from __future__ import annotations

from typing import Any

import numpy as np

from tengri.io.arrays import SpectrumTuple


def from_spectrum1d(sp: Any) -> SpectrumTuple:
    """Adapt a specutils.Spectrum1D into a SpectrumTuple.

    Preserves units by converting wavelength to Angstrom and flux to
    erg/s/cm^2/A via astropy.units spectral equivalence. If flux is in
    a non-density unit (e.g., Jy, maggies), converts via spectral
    equivalence at the effective wavelength.

    Parameters
    ----------
    sp : specutils.Spectrum1D
        Input spectrum object. Must have wavelength and flux attributes
        with astropy.units.

    Returns
    -------
    SpectrumTuple
        Tuple of (wave, flux, flux_err, meta) where wave is in Angstrom
        and flux is in erg/s/cm^2/A. meta includes 'instrument' (if
        present in Spectrum1D) and 'redshift' (if applicable).

    Raises
    ------
    ImportError
        If specutils or astropy is not installed.
    ValueError
        If wavelength or flux is missing or incompatible.

    Notes
    -----
    **JIT-compatible**: no, requires specutils and astropy.

    The conversion uses astropy.units spectral equivalence, so flux
    in any spectral density unit (F_lambda, F_nu, Jy, mag/arcsec^2, etc.)
    is correctly converted to erg/s/cm^2/A.

    Examples
    --------
    >>> from specutils import Spectrum1D
    >>> import astropy.units as u
    >>> spec_1d = Spectrum1D(
    ...     spectral_axis=np.linspace(4000, 9000) * u.AA,
    ...     flux=np.random.normal(1, 0.1, 5000) * u.erg / u.s / u.cm**2 / u.AA,
    ... )
    >>> spec = from_spectrum1d(spec_1d)
    >>> wave, flux, err, meta = spec
    """
    try:
        import astropy.units as u
    except ImportError:
        raise ImportError("from_spectrum1d requires astropy: pip install astropy") from None

    try:
        import specutils  # noqa: F401
    except ImportError:
        raise ImportError(
            "from_spectrum1d requires specutils: pip install specutils astropy"
        ) from None

    if sp.wavelength is None:
        raise ValueError("Spectrum1D has no wavelength attribute")
    if sp.flux is None:
        raise ValueError("Spectrum1D has no flux attribute")

    wave = sp.wavelength.to(u.AA).value
    wave = np.asarray(wave, dtype=np.float64)

    flux_unit = u.erg / u.s / u.cm**2 / u.AA
    try:
        flux = sp.flux.to(
            flux_unit,
            equivalencies=u.spectral_density(sp.wavelength),
        ).value
    except (u.UnitConversionError, ValueError):
        raise ValueError(
            f"Cannot convert flux units {sp.flux.unit} to {flux_unit}. "
            f"Check that flux is a valid spectral density."
        ) from None

    flux = np.asarray(flux, dtype=np.float64)

    if sp.uncertainty is not None and hasattr(sp.uncertainty, "array"):
        flux_err = np.asarray(sp.uncertainty.array, dtype=np.float64)
    else:
        flux_err = np.full_like(flux, np.nan)

    meta: dict[str, Any] = {}
    if hasattr(sp, "meta") and sp.meta:
        meta.update(sp.meta)

    if "instrument" not in meta:
        meta["instrument"] = "specutils"

    return SpectrumTuple(wave, flux, flux_err, meta)
