# SPDX-License-Identifier: BSD-3-Clause
r"""Unit adapters: Synthesizer ``unyt`` spectra ↔ tengri erg/s/Hz.

Synthesizer (Lovell et al. 2025; Roper et al. 2026 — cite both) attaches physical units to
every array through ``unyt``. Its :class:`synthesizer.emissions.Sed` already
reports the spectral luminosity :math:`L_\nu` in ``erg/(Hz*s)`` on a
rest-frame Angstrom grid — the same convention tengri uses internally — so the
bridge is mostly stripping the ``unyt`` tag, not a numerical conversion. The
one genuine conversion is the :math:`L_\lambda \leftrightarrow L_\nu` Jacobian
(:math:`L_\nu = L_\lambda\,\lambda^2/c`), exercised by the bolometric
round-trip self-test so a factor-of-:math:`c` slip in any downstream panel
trips here at import time rather than silently misshaping a figure.

References
----------
.. [1] Synthesizer (cite BOTH papers):
       Lovell, C.C., et al. (2025), Open J. Astrophys. 8, doi:10.33232/001c.145766;
       Roper, W.J., et al. (2026), JOSS 11, 9436, doi:10.21105/joss.09436.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Speed of light in Å/s for the L_λ ↔ L_ν Jacobian.
C_ANGSTROM_PER_S: float = 2.998e18
# IAU 2015 nominal solar luminosity [erg/s]. Synthesizer's stellar grids are
# already absolute (erg/s/Hz per M⊙ formed), so this is only used when the SSP
# grid is repackaged to express ``ssp_flux`` in L⊙/Hz/M⊙ — and it must match the
# constant multiplied back on load, so §1 stays a numerical statement.
L_SUN_ERG_PER_S: float = 3.828e33


def sed_to_lnu(sed) -> tuple[np.ndarray, np.ndarray]:
    r"""Strip a Synthesizer :class:`Sed` to plain ``(wave_aa, L_nu)`` arrays.

    Parameters
    ----------
    sed : synthesizer.emissions.Sed
        A Synthesizer SED with ``lam`` [Å] and ``lnu`` [erg/s/Hz] (``unyt``).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity [erg/s/Hz].
    """
    wave_aa = np.asarray(sed.lam.to("angstrom").value, dtype=np.float64)
    L_nu = np.asarray(sed.lnu.to("erg/s/Hz").value, dtype=np.float64)
    return wave_aa, L_nu


def ergs_per_aa_to_erg_per_hz(
    wave_aa: np.ndarray, L_lambda_erg_s_aa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert :math:`L_\lambda` [erg/s/Å] to :math:`L_\nu` [erg/s/Hz].

    The Jacobian is :math:`L_\nu = L_\lambda\,\lambda^2/c`; with :math:`\lambda`
    in Å and :math:`c` in Å/s the result is erg/s/Hz directly.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Wavelength grid [Å] (positive, finite).
    L_lambda_erg_s_aa : array_like, shape (..., n_wave)
        Spectral luminosity [erg/s/Å].

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Identity — returned for symmetry with the other drivers.
    L_nu_erg_per_hz : ndarray, same shape as input
        Spectral luminosity [erg/s/Hz].
    """
    wave_aa_arr = np.asarray(wave_aa)
    L_nu = np.asarray(L_lambda_erg_s_aa) * (wave_aa_arr**2) / C_ANGSTROM_PER_S
    return wave_aa_arr, L_nu


def verify_unit_conversion(rtol: float = 1e-3) -> dict[str, float]:
    r"""Assert the :math:`L_\lambda \leftrightarrow L_\nu` converter is bolometric-safe.

    Builds an SED of known bolometric :math:`L_\lambda`, runs it through
    :func:`ergs_per_aa_to_erg_per_hz`, integrates :math:`\int L_\nu\,d\nu`
    independently, and demands the two match. Called from the notebook Setup
    cell so the whole notebook trips here if the converter ever drifts.

    Parameters
    ----------
    rtol : float, optional
        Relative tolerance on the round-trip. Default 1e-3; in practice ~1e-6.

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
            f"rel_err={rel_err:.3e} > rtol={rtol:.3e}. A factor-of-c bug in "
            f"the converter would silently misshape every Synthesizer-vs-tengri panel."
        )
    return {"L_bol_in_erg_s": L_bol_in, "L_bol_out_erg_s": L_bol_out, "rel_err": rel_err}


def regrid(wave_src: np.ndarray, y_src: np.ndarray, wave_dst: np.ndarray) -> np.ndarray:
    """Log-log interpolate ``y_src(wave_src)`` onto ``wave_dst``; zero outside.

    Parameters
    ----------
    wave_src : array_like, shape (n_src,)
        Source wavelength grid [Å] (positive).
    y_src : array_like, shape (n_src,)
        Source values; non-positive entries are masked.
    wave_dst : array_like, shape (n_dst,)
        Destination wavelength grid [Å].

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
    label_l: str = "Synthesizer",
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
