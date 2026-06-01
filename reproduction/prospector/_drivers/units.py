"""Unit conversion adapters: FSPS L⊙/Hz ↔ tengri erg/s/Hz.

Prospector's forward engine is FSPS (via ``python-fsps``). FSPS'
``StellarPopulation.get_spectrum(peraa=False)`` returns the spectral
luminosity :math:`L_\\nu` in :math:`L_\\odot/\\mathrm{Hz}` per unit
stellar mass formed, on a rest-frame Angstrom grid. Tengri's convention
is :math:`L_\\nu` in erg/s/Hz. The bridge is a single multiplicative
constant — FSPS' own solar luminosity, :math:`L_\\odot = 3.839\\times
10^{33}` erg/s (``sps_vars.f90``). For the rare path where FSPS hands
back :math:`L_\\lambda` (``peraa=True``) the :math:`\\lambda^2/c`
Jacobian is also provided, and the bolometric round-trip self-test
exercises it.

References
----------
.. [1] Conroy, C., Gunn, J.E., White, M. (2009). The Propagation of
       Uncertainties in Stellar Population Synthesis Modeling. I.
       ApJ, 699, 486. arXiv:0809.4261.
.. [2] Johnson, B.D., Leja, J., Conroy, C., Speagle, J.S. (2021).
       Stellar Population Inference with Prospector. ApJS, 254, 22.
       arXiv:2012.01426.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Physical constants. L_SUN matches FSPS' internal value (sps_vars.f90,
# ``lsun = 3.839d33``) rather than IAU 2015 (3.828e33); using FSPS' own
# constant keeps absolute-flux panels (§3, §7) free of a 0.3 % offset
# that would otherwise masquerade as a physics disagreement.
C_ANGSTROM_PER_S: float = 2.998e18
L_SUN_ERG_PER_S: float = 3.839e33


def lnu_lsun_to_erg(L_nu_lsun: np.ndarray) -> np.ndarray:
    """Convert :math:`L_\\nu` from :math:`L_\\odot/\\mathrm{Hz}` to erg/s/Hz.

    Parameters
    ----------
    L_nu_lsun : array_like, shape (..., n_wave)
        Spectral luminosity in :math:`L_\\odot/\\mathrm{Hz}` as returned
        by ``fsps.StellarPopulation.get_spectrum(peraa=False)``.

    Returns
    -------
    L_nu_erg : ndarray, same shape
        Spectral luminosity in erg/s/Hz.
    """
    return np.asarray(L_nu_lsun, dtype=np.float64) * L_SUN_ERG_PER_S


def ergs_per_aa_to_erg_per_hz(
    wave_aa: np.ndarray, L_lambda_erg_s_aa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert :math:`L_\\lambda` [erg/s/Å] to :math:`L_\\nu` [erg/s/Hz].

    The Jacobian is :math:`L_\\nu = L_\\lambda\\,\\lambda^2/c`. With
    :math:`\\lambda` in Å and :math:`c` in Å/s the result is in erg/s/Hz
    directly.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Wavelength grid in Angstroms (positive, finite).
    L_lambda_erg_s_aa : array_like, shape (..., n_wave)
        Spectral luminosity in erg/s/Å.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Identity — returned for symmetry with the other drivers.
    L_nu_erg_per_hz : ndarray, same shape as input
        Spectral luminosity in erg/s/Hz.
    """
    wave_aa_arr = np.asarray(wave_aa)
    L_nu = np.asarray(L_lambda_erg_s_aa) * (wave_aa_arr**2) / C_ANGSTROM_PER_S
    return wave_aa_arr, L_nu


def verify_unit_conversion(rtol: float = 1e-3) -> dict[str, float]:
    """Assert the L_λ ↔ L_ν converter preserves bolometric luminosity.

    Every cross-code panel rests on the converters here returning the
    same integrated luminosity they were handed. A factor-of-c or
    factor-of-:math:`L_\\odot` bug would silently misshape every figure.
    This builds an SED of known bolometric :math:`L_\\lambda`, runs it
    through :func:`ergs_per_aa_to_erg_per_hz`, integrates
    :math:`\\int L_\\nu\\,d\\nu` independently, and demands the two match.

    Parameters
    ----------
    rtol : float, optional
        Relative tolerance on the round-trip. Default 1e-3; in practice
        we see ~1e-6.

    Returns
    -------
    dict
        ``{"L_bol_in_erg_s": …, "L_bol_out_erg_s": …, "rel_err": …}``.

    Raises
    ------
    AssertionError
        If the round-trip exceeds ``rtol``.
    """
    wave_aa = np.linspace(91.0, 24000.0, 5000)
    L_lambda = np.exp(-((wave_aa - 5000.0) ** 2) / (2 * 2000.0**2))
    L_bol_in = float(np.trapezoid(L_lambda, wave_aa))

    _, L_nu = ergs_per_aa_to_erg_per_hz(wave_aa, L_lambda)
    nu = C_ANGSTROM_PER_S / wave_aa[::-1]
    L_bol_out = float(np.trapezoid(L_nu[::-1], nu))

    rel_err = abs(L_bol_out - L_bol_in) / L_bol_in
    if rel_err > rtol:
        raise AssertionError(
            f"ergs_per_aa_to_erg_per_hz fails bolometric round-trip: "
            f"L_bol_in={L_bol_in:.6e}, L_bol_out={L_bol_out:.6e}, "
            f"rel_err={rel_err:.3e} > rtol={rtol:.3e}. "
            f"A factor-of-c bug in the converter would silently misshape "
            f"every Prospector-vs-tengri panel."
        )
    return {
        "L_bol_in_erg_s": L_bol_in,
        "L_bol_out_erg_s": L_bol_out,
        "rel_err": rel_err,
    }


def regrid(wave_src: np.ndarray, y_src: np.ndarray, wave_dst: np.ndarray) -> np.ndarray:
    """Log-log interpolate ``y_src(wave_src)`` onto ``wave_dst``; zero outside.

    Parameters
    ----------
    wave_src : array_like, shape (n_src,)
        Source wavelength grid in Angstroms (must be positive).
    y_src : array_like, shape (n_src,)
        Source values (must be positive for the log-log interpolation to
        be meaningful — non-positive entries are masked).
    wave_dst : array_like, shape (n_dst,)
        Destination wavelength grid.

    Returns
    -------
    y_dst : ndarray, shape (n_dst,)
        Interpolated values on the destination grid; zero outside the
        source wavelength range.
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
    label_l: str = "Prospector (FSPS)",
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
