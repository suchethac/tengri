"""Unit conversion adapters: bagpipes erg/s/Å ↔ tengri erg/s/Hz.

Bagpipes' ``model_galaxy.spectrum_full`` at ``redshift=0`` is in erg/s/Å
(rest-frame :math:`L_\\lambda` per unit total formed mass, multiplied by
:math:`10^{\\mathrm{massformed}}\\,M_\\odot`). Tengri's convention is
:math:`L_\\nu` in erg/s/Hz on an Angstrom grid. This module bridges
the two and ships a bolometric round-trip self-test.

References
----------
.. [1] Carnall, A.C., et al. (2018). Inferring the star formation
       histories of massive quiescent galaxies with BAGPIPES.
       MNRAS, 480, 4379.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Physical constants — must match what bagpipes itself uses internally.
# Bagpipes hard-codes 3.826e33 (line 492 of model_galaxy.py); we use the
# same value to make the round-trip exact at machine precision.
C_ANGSTROM_PER_S: float = 2.998e18
L_SUN_ERG_PER_S: float = 3.826e33


def ergs_per_aa_to_erg_per_hz(
    wave_aa: np.ndarray, L_lambda_erg_s_aa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert :math:`L_\\lambda` [erg/s/Å] to :math:`L_\\nu` [erg/s/Hz].

    The Jacobian is :math:`L_\\nu = L_\\lambda\\,\\lambda^2/c`. With
    :math:`\\lambda` in Å and :math:`c` in Å/s the result is in erg/s/Hz
    directly — no further conversion needed.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Wavelength grid in Angstroms (positive, finite).
    L_lambda_erg_s_aa : array_like, shape (..., n_wave)
        Spectral luminosity in erg/s/Å. May be broadcast against
        ``wave_aa`` along the trailing axis.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Identity — returned for symmetry with the CIGALE driver.
    L_nu_erg_per_hz : ndarray, same shape as input
        Spectral luminosity in erg/s/Hz.

    Notes
    -----
    Bagpipes' ``spectrum_full`` at ``redshift=0`` carries an implicit
    factor of :math:`10^{\\mathrm{massformed}}\\,M_\\odot`; callers
    that want a per-Msun quantity must divide by
    ``10**model_components['massformed']`` themselves.
    """
    wave_aa_arr = np.asarray(wave_aa)
    L_nu = np.asarray(L_lambda_erg_s_aa) * (wave_aa_arr**2) / C_ANGSTROM_PER_S
    return wave_aa_arr, L_nu


def verify_unit_conversion(rtol: float = 1e-3) -> dict[str, float]:
    """Assert ``ergs_per_aa_to_erg_per_hz`` preserves bolometric luminosity.

    Every cross-code panel rests on this converter giving back the same
    integrated luminosity it was handed. A factor-of-10 or factor-of-c
    bug here would silently misshape every figure. This function builds
    an SED of known bolometric :math:`L_\\lambda`, runs it through the
    converter, integrates :math:`\\int L_\\nu\\,d\\nu` independently,
    and demands the two match within ``rtol``.

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
            f"every bagpipes-vs-tengri panel."
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
        Source values (must be positive for the log-log interpolation
        to be meaningful — non-positive entries are masked).
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
    label_l: str = "bagpipes.model_galaxy",
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


def line_lum(wave, L_nu, center, half=12.0):
    """Integrated emission-line luminosity [erg/s] within +/- ``half`` Å of ``center``.

    Converts ``L_nu`` [erg/s/Hz] to ``L_lambda`` [erg/s/Å], subtracts the
    in-window floor as a flat local continuum, and integrates the excess.

    This is **width- and grid-independent** — the fair way to compare an
    emission line between codes that represent it with different intrinsic
    broadening or wavelength sampling. A single-bin peak ``L_nu`` ratio
    measures line *width* (and grid resolution), not luminosity: tengri's Cue
    applies an intrinsic velocity broadening while several reference codes
    place lines at the grid resolution, so peak ratios are meaningless.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    L_nu : array_like, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz] on ``wave``.
    center : float
        Line-center wavelength [Å].
    half : float, optional
        Half-width of the integration window [Å]. Default 12.

    Returns
    -------
    float
        Continuum-subtracted integrated line luminosity [erg/s]; 0.0 if the
        window holds fewer than two grid points.
    """
    wave = np.asarray(wave)
    L_nu = np.asarray(L_nu)
    m = (wave >= center - half) & (wave <= center + half)
    if int(m.sum()) < 2:
        return 0.0
    order = np.argsort(wave[m])
    lam = wave[m][order]
    l_lambda = L_nu[m][order] * C_ANGSTROM_PER_S / lam**2  # erg/s/Å
    return float(np.trapezoid(np.clip(l_lambda - l_lambda.min(), 0.0, None), lam))
