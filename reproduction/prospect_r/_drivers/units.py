"""Unit conversion adapters: ProSpect L⊙/Å ↔ tengri erg/s/Hz.

ProSpect (Robotham et al. 2020) returns rest-frame spectra as a spectral
luminosity density :math:`L_\\lambda` in :math:`L_\\odot/\\mathrm{\\AA}` on a
rest-frame Angstrom grid (the ``$lum`` column of its ``speclib`` slices and of
``ProSpectSED``'s ``FinalLum``). Tengri's convention is :math:`L_\\nu` in
erg/s/Hz. The bridge carries two factors: ProSpect's own solar luminosity
:math:`L_\\odot = 3.828\\times10^{33}` erg/s (``.lsol_to_erg`` in ProSpect's
constants), and the :math:`\\lambda^2/c` Jacobian that turns a per-wavelength
density into a per-frequency one.

Using ProSpect's *own* :math:`L_\\odot` rather than the IAU 2015 value keeps
absolute-flux panels free of a ~0.1 % offset that would otherwise read as a
physics disagreement.

References
----------
.. [1] Robotham, A.S.G., Bellstedt, S., Lagos, C.d.P., et al. (2020).
       ProSpect: generating rapid spectral energy distributions with
       complex star formation and metallicity histories. MNRAS, 495, 905.
       arXiv:2002.06980.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Speed of light in Angstrom/s, for the L_lambda -> L_nu Jacobian.
C_ANGSTROM_PER_S: float = 2.998e18
# ProSpect's internal solar luminosity (constants.R, ``.lsol_to_erg``),
# matched here so the absolute normalization agrees with the R side.
L_SUN_ERG_PER_S: float = 3.828e33


def lsun_per_aa_to_erg_per_hz(
    wave_aa: np.ndarray, L_lsun_per_aa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert ProSpect :math:`L_\lambda` [L\ :sub:`\odot`/\ \AA] to :math:`L_\nu` [erg/s/Hz].

    ProSpect spectra carry wavelength in Angstroms and luminosity density as
    :math:`L_\lambda` in :math:`L_\odot/\mathrm{\AA}` (rest frame). The
    conversion is :math:`L_\nu = L_\lambda\,L_\odot\,\lambda^2/c`, with
    :math:`L_\odot` in erg/s, :math:`\lambda` in \AA, and :math:`c` in \AA/s.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Rest-frame wavelength in Angstroms.
    L_lsun_per_aa : array_like, shape (..., n_wave)
        Spectral luminosity density in :math:`L_\odot/\mathrm{\AA}`.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength in Angstroms (returned for symmetry with the other drivers).
    L_nu_erg_per_hz : ndarray, same shape as ``L_lsun_per_aa``
        Spectral luminosity density in erg/s/Hz.
    """
    wave_aa_arr = np.asarray(wave_aa, dtype=np.float64)
    L_lambda_erg = np.asarray(L_lsun_per_aa, dtype=np.float64) * L_SUN_ERG_PER_S
    L_nu = L_lambda_erg * (wave_aa_arr**2) / C_ANGSTROM_PER_S
    return wave_aa_arr, L_nu


def verify_unit_conversion(rtol: float = 1e-3) -> dict[str, float]:
    r"""Assert :func:`lsun_per_aa_to_erg_per_hz` preserves bolometric luminosity.

    Every cross-code panel rests on the converter returning the same
    integrated luminosity it was handed. A factor-of-:math:`c` or
    factor-of-:math:`L_\odot` bug would silently misshape every figure. This
    builds an SED of known bolometric :math:`L_\lambda` in
    :math:`L_\odot/\mathrm{\AA}`, runs it through the converter, integrates
    :math:`\int L_\nu\,d\nu` independently, and demands the two agree.

    Parameters
    ----------
    rtol : float, optional
        Relative tolerance on the bolometric round-trip. Default 1e-3; in
        practice the round-trip lands near 1e-6.

    Returns
    -------
    dict
        ``{"L_bol_in_erg_s": …, "L_bol_out_erg_s": …, "rel_err": …}``.

    Raises
    ------
    AssertionError
        If the round-trip exceeds ``rtol``.
    """
    # Gaussian L_lambda profile in Lsun/Aa on a UV-NIR grid; magnitude is
    # arbitrary because the test is a ratio.
    wave_aa = np.linspace(91.0, 24000.0, 5000)
    L_lambda_lsun = np.exp(-((wave_aa - 5000.0) ** 2) / (2 * 2000.0**2))
    # Bolometric in erg/s on the ProSpect side: integral of L_lambda dlambda,
    # with Lsun -> erg/s applied.
    L_bol_in = float(np.trapezoid(L_lambda_lsun, wave_aa)) * L_SUN_ERG_PER_S

    _, L_nu = lsun_per_aa_to_erg_per_hz(wave_aa, L_lambda_lsun)
    nu = C_ANGSTROM_PER_S / wave_aa[::-1]  # increasing frequency
    L_bol_out = float(np.trapezoid(L_nu[::-1], nu))

    rel_err = abs(L_bol_out - L_bol_in) / L_bol_in
    if rel_err > rtol:
        raise AssertionError(
            f"lsun_per_aa_to_erg_per_hz fails bolometric round-trip: "
            f"L_bol_in={L_bol_in:.6e} erg/s, L_bol_out={L_bol_out:.6e} erg/s, "
            f"rel_err={rel_err:.3e} > rtol={rtol:.3e}. "
            f"A factor-of-c or factor-of-Lsun bug in the converter would "
            f"silently misshape every ProSpect-vs-tengri panel."
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
        Source values (must be positive for the log-log interpolation to be
        meaningful — non-positive entries are masked).
    wave_dst : array_like, shape (n_dst,)
        Destination wavelength grid.

    Returns
    -------
    y_dst : ndarray, shape (n_dst,)
        Interpolated values on the destination grid; zero outside the source
        wavelength range.
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
    label_l: str = "ProSpect",
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
