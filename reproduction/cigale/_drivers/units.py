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


def verify_unit_conversion(rtol: float = 1e-3) -> dict:
    """Assert ``wnm_to_erg_per_hz_per_aa`` preserves bolometric luminosity.

    Every reproduction-notebook claim of "agrees to a fraction of a percent"
    rests on the W/nm → erg/s/Hz conversion in
    :func:`wnm_to_erg_per_hz_per_aa`. A 10× or 1e7× factor-of-units bug
    there would silently misshape every panel. This function constructs
    an SED of known bolometric L_λ, runs it through the converter, and
    independently computes the bolometric integral in the converted
    units. The two must agree within ``rtol`` (default 1e-3) or this
    raises.

    Parameters
    ----------
    rtol : float
        Relative tolerance on the bolometric round-trip.

    Returns
    -------
    dict
        ``{"L_bol_in_erg_s": …, "L_bol_out_erg_s": …, "rel_err": …}``.

    Raises
    ------
    AssertionError
        If the round-trip exceeds ``rtol``. Suggests a units bug in
        :func:`wnm_to_erg_per_hz_per_aa`.
    """
    # Build a Gaussian L_λ profile (W/nm) on a UV–NIR grid (91–24000 Å).
    # Magnitude doesn't matter — the test is a ratio.
    wave_nm = np.linspace(9.1, 2400.0, 5000)  # 91 Å – 24000 Å
    L_wnm = np.exp(-((wave_nm - 500.0) ** 2) / (2 * 200.0**2))  # peak at 5000 Å
    # CIGALE-side bolometric in erg/s: ∫ L_λ dλ, with W → erg/s (×1e7) and
    # nm units carried through dλ (so the result has W (=erg/s) units after ×1e7).
    L_bol_in_erg_s = float(np.trapezoid(L_wnm, wave_nm)) * 1e7
    # Convert and integrate ∫ L_ν dν on the increasing-ν grid.
    wave_aa, L_nu = wnm_to_erg_per_hz_per_aa(wave_nm, L_wnm)
    c_aa_per_s = 2.998e18
    nu = c_aa_per_s / wave_aa[::-1]  # increasing
    L_nu_rev = L_nu[::-1]
    L_bol_out_erg_s = float(np.trapezoid(L_nu_rev, nu))
    rel_err = abs(L_bol_out_erg_s - L_bol_in_erg_s) / L_bol_in_erg_s
    if rel_err > rtol:
        raise AssertionError(
            f"wnm_to_erg_per_hz_per_aa fails bolometric round-trip: "
            f"L_bol_in={L_bol_in_erg_s:.6e} erg/s, "
            f"L_bol_out={L_bol_out_erg_s:.6e} erg/s, "
            f"rel_err={rel_err:.3e} > rtol={rtol:.3e}. "
            f"A factor-of-10 or 1e7 bug in the converter would silently "
            f"misshape every CIGALE-vs-tengri panel."
        )
    return {
        "L_bol_in_erg_s": L_bol_in_erg_s,
        "L_bol_out_erg_s": L_bol_out_erg_s,
        "rel_err": rel_err,
    }


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
