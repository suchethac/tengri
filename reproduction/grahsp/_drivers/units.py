"""Unit adapters: AGNFITTER-RX (log10 nu, F_nu) <-> tengri (Angstrom, erg/s/Hz).

AGNFITTER-RX stores every model template as a pair ``(log10(nu / Hz), F_nu)``
with the frequency axis in *descending* wavelength order. tengri works in
ascending wavelength [Angstrom] with the luminosity density as L_nu
[erg/s/Hz]. F_nu and L_nu are the same physical quantity (a per-frequency
density), so the conversion is a reorder plus the nu <-> lambda axis map -- no
Jacobian factor enters the density itself. The bolometric round-trip below
checks that the nu <-> lambda mapping we plot with preserves the integral.

The ``regrid``, ``panel``, and ``two_panel_fig`` helpers are kept identical to
the CIGALE/Prospector drivers so the three notebooks read as one series; only
the luminosity converter and ``L_SUN`` differ between codes.

References
----------
.. [1] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Speed of light in the two unit systems used below.
C_ANGSTROM_PER_S: float = 2.998e18
# Solar luminosity (AGNFITTER-RX normalizes templates per L_sun; see
# MODEL_AGNfitter.renorm_template). IAU 2015 nominal value.
L_SUN_ERG_PER_S: float = 3.828e33


def lognu_fnu_to_lnu(log_nu_hz, F_nu):
    """Convert an AGNFITTER-RX template to tengri's plotting convention.

    AGNFITTER-RX templates carry the frequency axis as ``log10(nu / Hz)`` in
    descending-wavelength order and the SED as ``F_nu``. This returns the same
    SED on an ascending wavelength grid [Angstrom] with L_nu [erg/s/Hz]
    (numerically identical values, reordered).

    Parameters
    ----------
    log_nu_hz : array_like, shape (n,)
        Base-10 logarithm of frequency in Hz.
    F_nu : array_like, shape (n,)
        Per-frequency luminosity density (arbitrary AGNFITTER-RX
        normalization). [erg/s/Hz up to a constant]

    Returns
    -------
    wave_aa : ndarray, shape (n,)
        Wavelength in Angstrom, ascending.
    L_nu : ndarray, shape (n,)
        Luminosity density on ``wave_aa``. [erg/s/Hz]
    """
    log_nu_hz = np.asarray(log_nu_hz, dtype=np.float64).ravel()
    F_nu = np.asarray(F_nu, dtype=np.float64).ravel()
    nu_hz = 10.0**log_nu_hz
    wave_aa = C_ANGSTROM_PER_S / nu_hz
    order = np.argsort(wave_aa)
    return wave_aa[order], F_nu[order]


def verify_unit_conversion(rtol: float = 1e-3) -> dict:
    """Assert the nu <-> lambda axis map preserves bolometric luminosity.

    Every "agrees to a fraction of a percent" claim downstream rests on the
    frequency/wavelength bookkeeping in :func:`lognu_fnu_to_lnu`. This builds a
    Gaussian L_nu of known bolometric luminosity on a log-frequency grid, maps
    it through the converter, and independently integrates the bolometric in
    both ``int L_nu dnu`` and ``int L_lambda dlambda`` forms. They must agree
    within ``rtol`` (default 1e-3) or a units bug has crept in.

    Parameters
    ----------
    rtol : float
        Relative tolerance on the bolometric round-trip.

    Returns
    -------
    dict
        ``{"L_bol_nu": ..., "L_bol_lambda": ..., "rel_err": ...}``.

    Raises
    ------
    AssertionError
        If the round-trip exceeds ``rtol``.
    """
    # Gaussian in log-nu, peak near the optical (log nu ~ 14.9 -> ~5000 A).
    log_nu = np.linspace(12.0, 16.0, 5000)  # FIR to FUV
    F_nu = np.exp(-((log_nu - 14.9) ** 2) / (2 * 0.4**2))
    nu = 10.0**log_nu
    # int L_nu dnu on the (ascending-nu) grid.
    L_bol_nu = float(np.trapezoid(F_nu, nu))
    # Map to wavelength and integrate int L_lambda dlambda, L_lambda = L_nu c / lambda^2.
    wave_aa, L_nu = lognu_fnu_to_lnu(log_nu, F_nu)
    L_lambda = L_nu * C_ANGSTROM_PER_S / wave_aa**2
    L_bol_lambda = float(np.trapezoid(L_lambda, wave_aa))
    rel_err = abs(L_bol_lambda - L_bol_nu) / L_bol_nu
    if rel_err > rtol:
        raise AssertionError(
            f"lognu_fnu_to_lnu fails the bolometric round-trip: "
            f"int L_nu dnu = {L_bol_nu:.6e}, int L_lambda dlambda = {L_bol_lambda:.6e}, "
            f"rel_err = {rel_err:.3e} > rtol = {rtol:.3e}. A frequency/wavelength "
            f"bookkeeping bug would silently misshape every AGNFITTER-vs-tengri panel."
        )
    return {"L_bol_nu": L_bol_nu, "L_bol_lambda": L_bol_lambda, "rel_err": rel_err}


def regrid(wave_src: np.ndarray, y_src: np.ndarray, wave_dst: np.ndarray) -> np.ndarray:
    """Log-log interpolate ``y_src(wave_src)`` onto ``wave_dst``; zero outside.

    Parameters
    ----------
    wave_src : array_like, shape (n_src,)
        Source wavelength grid in Angstroms (must be positive).
    y_src : array_like, shape (n_src,)
        Source values (positive entries only contribute; non-positive masked).
    wave_dst : array_like, shape (n_dst,)
        Destination wavelength grid.

    Returns
    -------
    y_dst : ndarray, shape (n_dst,)
        Interpolated values; zero outside the source wavelength range.
    """
    wave_src = np.asarray(wave_src)
    y_src = np.asarray(y_src)
    wave_dst = np.asarray(wave_dst)

    mask = (wave_src > 0) & (y_src > 0)
    if not np.any(mask):
        return np.zeros_like(wave_dst)

    log_y = np.interp(
        np.log10(wave_dst),
        np.log10(wave_src[mask]),
        np.log10(y_src[mask]),
        left=np.nan,
        right=np.nan,
    )
    y_dst = 10.0**log_y
    in_range = (wave_dst >= wave_src[mask].min()) & (wave_dst <= wave_src[mask].max())
    y_dst[~in_range] = 0.0
    return y_dst


def panel(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    *,
    label_l: str = "AGNFITTER-RX",
    label_r: str = "tengri",
) -> tuple[plt.Axes, plt.Axes]:
    """Configure two SED comparison panels in matched log-log axes."""
    for ax in (ax_left, ax_right):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$ [Å]")
        ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
    ax_left.set_title(label_l)
    ax_right.set_title(label_r)
    return ax_left, ax_right


def two_panel_fig(figsize: tuple[float, float] = (12, 4.5)):
    """Create a standard 2-panel figure for SED comparisons."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, figsize=figsize)
    return fig, ax_left, ax_right
