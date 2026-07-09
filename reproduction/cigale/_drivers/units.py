"""Unit conversion adapters: CIGALE W/nm ↔ tengri erg/s/Hz.

Functions to convert between CIGALE's wavelength and luminosity conventions
and those used in tengri and standard astrophysical codes.

References
----------
.. [1] Boquien, M., et al. (2019). CIGALE: Code Investigating GALaxy
       Emission. Astronomy & Astrophysics, 622, A103.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Physical constants. CIGALE works in W/nm; the converter below carries the
# full unit chain explicitly, so only the speed of light is needed here.
C_ANGSTROM_PER_S: float = 2.998e18


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

    # L_ν [erg/s/Hz] = L_λ [W/nm] * (1e7 erg/s / W) * (1 nm / 10 Å)
    #                  * λ² [Å²] / c [Å/s]
    #                = L_λ [W/nm] * 1e6 * λ² [Å²] / c [Å/s]
    L_nu_erg_per_hz = L_wnm * 1e6 * (wave_aa**2) / C_ANGSTROM_PER_S

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
    nu = C_ANGSTROM_PER_S / wave_aa[::-1]  # increasing
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


def regrid(wave_src: np.ndarray, y_src: np.ndarray, wave_dst: np.ndarray) -> np.ndarray:
    """Log-log PCHIP-interpolate ``y_src(wave_src)`` onto ``wave_dst``; zero outside.

    Uses a shape-preserving monotone cubic (PCHIP) interpolant in
    ``log10(wave)`` vs ``log10(y)`` space. PCHIP is preferred over linear
    ``np.interp`` for the SED comparison because linear interpolation chords
    *under* convex features (the torus IR bump, the stellar Wien tail),
    biasing the regridded reference below the true curve in a
    wavelength-dependent way that mimics a physics residual. PCHIP tracks
    curvature without the overshoot/ringing of a natural cubic spline, so it
    neither under-shoots peaks nor invents oscillations.

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

    Notes
    -----
    PCHIP requires a strictly increasing abscissa, so the positive-value mask
    is followed by a sort + duplicate-``wave`` collapse before fitting.
    """
    from scipy.interpolate import PchipInterpolator

    wave_src = np.asarray(wave_src)
    y_src = np.asarray(y_src)
    wave_dst = np.asarray(wave_dst)

    mask = (wave_src > 0) & (y_src > 0)
    if not np.any(mask):
        return np.zeros_like(wave_dst)

    log_x = np.log10(wave_src[mask])
    log_y = np.log10(y_src[mask])
    # PCHIP needs strictly increasing, unique abscissa.
    order = np.argsort(log_x)
    log_x, log_y = log_x[order], log_y[order]
    uniq = np.concatenate(([True], np.diff(log_x) > 0))
    log_x, log_y = log_x[uniq], log_y[uniq]
    if log_x.size < 2:
        return np.zeros_like(wave_dst)

    interp = PchipInterpolator(log_x, log_y, extrapolate=False)
    y_dst = 10.0 ** interp(np.log10(wave_dst))
    # ``extrapolate=False`` yields NaN outside the source span → zero them.
    y_dst = np.where(np.isfinite(y_dst), y_dst, 0.0)
    return y_dst


def panel(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    *,
    label_l: str = "pcigale.sed_modules",
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


def overlay_ratio_fig(
    wave_c: np.ndarray,
    L_c: np.ndarray,
    wave_t: np.ndarray,
    L_t: np.ndarray,
    *,
    x_of_wave=None,
    xlabel: str = r"$\lambda$ [Å]",
    title: str = "",
    label_c: str = "CIGALE",
    label_t: str = "tengri (regridded)",
    xlim: tuple[float, float] | None = None,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    band: tuple[float, float] = (0.9, 1.1),
    dyn_range: float = 1e-5,
    figsize: tuple[float, float] = (10, 7),
):
    """Overlay CIGALE + tengri on ONE shared axis with a tengri/CIGALE ratio panel.

    Replaces the side-by-side independent-y-axis layout (which hides
    normalization discrepancies, #864) with the §9c format: both codes on a
    single log-log axis, tengri regridded onto CIGALE's grid via :func:`regrid`,
    plus a lower ratio panel with a tolerance band. The x-axis can be
    transformed away from wavelength (e.g. photon energy for X-ray, frequency
    for radio) via ``x_of_wave``.

    Parameters
    ----------
    wave_c, L_c : ndarray
        CIGALE wavelength grid [Å] and L_nu [erg/s/Hz].
    wave_t, L_t : ndarray
        tengri wavelength grid [Å] and L_nu [erg/s/Hz]; regridded onto ``wave_c``.
    x_of_wave : callable, optional
        Maps the Å grid to the plotted abscissa (e.g. ``lambda w: 12.398/w`` for
        keV). Identity (wavelength) when ``None``.
    xlabel, title, label_c, label_t : str
        Axis label, panel title, and legend labels.
    xlim : tuple, optional
        x-limits in the plotted abscissa units.
    ratio_ylim, band : tuple
        Ratio-panel y-limits and shaded tolerance band (default ±10%).
    dyn_range : float
        Top-panel y floor as a fraction of the peak.

    Returns
    -------
    fig, ax, ax_ratio, ratio : the figure, main axis, ratio axis, and the
        tengri/CIGALE ratio array on ``wave_c``.
    """
    L_t_on_c = regrid(wave_t, L_t, wave_c)
    ratio = L_t_on_c / np.maximum(L_c, 1e-50)
    x = wave_c if x_of_wave is None else x_of_wave(wave_c)

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    pos = L_c > 0
    ax.plot(x[pos], L_c[pos], "C0-", linewidth=1.6, label=label_c)
    post = L_t_on_c > 0
    ax.plot(x[post], L_t_on_c[post], "C1--", linewidth=1.4, label=label_t)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    _ymx_c = float(np.nanmax(L_c[pos])) if pos.any() else 1.0
    _ymx_t = float(np.nanmax(L_t_on_c[post])) if post.any() else 1.0
    ymx = max(_ymx_c, _ymx_t)
    ax.set_ylim(ymx * dyn_range, ymx * 2.0)
    ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax_r.axhspan(*band, color="0.85", zorder=0)
    ax_r.axhline(1.0, color="0.5", linewidth=0.8)
    ax_r.plot(x[pos], ratio[pos], "C1-", linewidth=1.0)
    ax_r.set_xscale("log")
    ax_r.set_ylim(*ratio_ylim)
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel("tengri / CIGALE", fontsize=9)
    ax_r.grid(True, alpha=0.3)
    return fig, ax, ax_r, ratio


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
