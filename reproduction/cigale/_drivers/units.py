"""Unit conversion adapters: CIGALE W/nm ↔ tengri erg/s/Hz.

Functions to convert between CIGALE's wavelength and luminosity conventions
and those used in tengri and standard astrophysical codes.

References
----------
.. [1] Boquien, M., et al. (2019). CIGALE: Code Investigating GALaxy
       Emission. Astronomy & Astrophysics, 622, A103.
"""

import numpy as np
import matplotlib.pyplot as plt


def wnm_to_erg_per_hz_per_aa(wave_nm, L_wnm):
    """Convert CIGALE SED to tengri units.

    CIGALE returns wavelength in nanometers and luminosity density as
    L_λ in watts per nanometer. This function converts to the tengri
    convention: wavelength in Angstroms and luminosity density as L_ν
    in erg/s/Hz.

    The conversion uses the Jacobian: L_ν = L_λ · λ² / c, with careful
    unit handling. 1 W = 1e7 erg/s; 1 nm = 10 Å; c = 2.998e18 Å/s.

    Parameters
    ----------
    wave_nm : array_like, shape (n_wave,)
        Wavelength in nanometers.
    L_wnm : array_like, shape (n_wave,)
        Luminosity density in W/nm.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength in Angstroms.
    L_nu_erg_per_hz : ndarray, shape (n_wave,)
        Luminosity density in erg/s/Hz.
    """
    wave_aa = wave_nm * 10.0
    c_aa_per_s = 2.998e18  # speed of light in Angstrom/s

    # L_ν [erg/s/Hz] = L_λ [W/nm] * (1e7 erg/s / W) * (1 nm / 10 Å)
    #                  * λ² [Å²] / c [Å/s]
    #                = L_λ [W/nm] * 1e6 * λ² [Å²] / c [Å/s]
    L_nu_erg_per_hz = L_wnm * 1e6 * (wave_aa ** 2) / c_aa_per_s

    return wave_aa, L_nu_erg_per_hz


def regrid(wave_src, y_src, wave_dst):
    """Interpolate in log-log space; zero outside source range.

    Parameters
    ----------
    wave_src : array_like, shape (n_src,)
        Source wavelength grid (must be positive).
    y_src : array_like, shape (n_src,)
        Source flux or SED values (must be positive).
    wave_dst : array_like, shape (n_dst,)
        Destination wavelength grid.

    Returns
    -------
    y_dst : ndarray, shape (n_dst,)
        Interpolated values on destination grid. Zero outside source
        wavelength range.
    """
    wave_src = np.asarray(wave_src)
    y_src = np.asarray(y_src)
    wave_dst = np.asarray(wave_dst)

    # Clamp log-space interpolation to avoid nans/infs from negative values
    mask_src = (wave_src > 0) & (y_src > 0)
    if not np.any(mask_src):
        return np.zeros_like(wave_dst)

    log_wave_src = np.log10(wave_src[mask_src])
    log_y_src = np.log10(y_src[mask_src])

    # Interpolate in log-log space
    log_y_dst = np.interp(
        np.log10(wave_dst),
        log_wave_src,
        log_y_src,
        left=np.nan,
        right=np.nan,
    )
    y_dst = 10.0 ** log_y_dst

    # Set to zero outside source wavelength range
    in_range = (wave_dst >= wave_src[mask_src].min()) & (
        wave_dst <= wave_src[mask_src].max()
    )
    y_dst[~in_range] = 0.0

    return y_dst


def panel(ax_left, ax_right, *, label_l="pcigale.sed_modules", label_r="tengri"):
    """Configure two SED comparison panels.

    Sets both axes to log-log scale, adds shared x-label (wavelength
    in Angstroms), y-label (νL_ν or L_ν in erg/s/Hz), and titles.

    Parameters
    ----------
    ax_left : matplotlib.axes.Axes
        Left subplot axes.
    ax_right : matplotlib.axes.Axes
        Right subplot axes.
    label_l : str, optional
        Title for left panel. Default: "pcigale.sed_modules".
    label_r : str, optional
        Title for right panel. Default: "tengri".

    Returns
    -------
    ax_left, ax_right : tuple of matplotlib.axes.Axes
        The configured axes for further use.
    """
    for ax in (ax_left, ax_right):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$ [Å]")
        ax.set_ylabel(r"$\nu L_\nu$ or $L_\nu$ [erg/s/Hz]")

    ax_left.set_title(label_l)
    ax_right.set_title(label_r)

    return ax_left, ax_right


def two_panel_fig(figsize=(12, 4.5)):
    """Create a standard 2-panel figure for SED comparisons.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (12, 4.5).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax_left, ax_right : tuple of matplotlib.axes.Axes
        Left and right subplot axes, with shared y-axis.
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, sharey=True, figsize=figsize
    )
    return fig, ax_left, ax_right
