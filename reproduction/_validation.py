"""Shared helpers for the matched-input validators: broadband photometry,
emission lines, radio, and X-ray.

The per-comparison ``validate_matched_physics.py`` scripts started with six
hand-written wavelength *windows* (``FUV 1216-1900 A`` and friends) topping out
at 300 um. Two things were missing:

* **Reach.** A window table cannot say anything about the sub-mm, where the
  cold-dust mass lives and where the two codes' dust-emission templates are
  least constrained by the optical fit.
* **Realism.** A flat window is not what an SED fitter predicts. Surveys measure
  a *bandpass* average, and a band's answer can differ from its window's when
  the SED has structure inside the band -- which is exactly the case across the
  Balmer/4000 A break, the PAH complex, and any strong line.

This module supplies both, and asks the same question of the three channels a
window table never reached at all: emission lines, radio, and X-ray. Those
three are compared at discrete lines/frequencies/energies rather than through
bandpasses -- no cm-wave or X-ray transmission curves ship in ``data/filters``,
and it is how each of those measurements is actually quoted.

Why here and not in ``_drivers/units.py``
-----------------------------------------
CONTRACT section 3 says shared helpers live in ``_drivers/units.py``,
byte-identical across comparisons. That rule is about helpers the *validators*
use. Validator-only: putting filter I/O and a 23-entry bandpass ladder
into six copies of a module every notebook imports would cost the notebook path
and buy it nothing. The sweep helpers ``sweep_fig``, ``window_rows``, and
``print_window_table`` are imported by the reproduction notebooks; they live
here to keep the validators and the sweep plotting synchronized. CONTRACT
section 3 records both carry-outs.

The bandpass convention, and why it is not a trap here
------------------------------------------------------
The band-averaged luminosity density is

.. math::

    \\langle L_\\nu\\rangle = \\frac{\\int L_\\nu(\\lambda) T(\\lambda) w(\\lambda)
    \\, d\\lambda}{\\int T(\\lambda) w(\\lambda)\\, d\\lambda},

with :math:`w = 1/\\lambda` for photon-counting detectors (``"photon"``, what
tengri, DSPS, FSPS, sedpy and prospector default to) and :math:`w = 1/\\lambda^2`
for energy-counting (``"energy"``, CIGALE's and bagpipes' convention). See
:class:`tengri.utils.filter_convention.FilterConvention`, which carries the
references.

Choosing the wrong one matters when you compare against a *published magnitude*
-- it is a 5-40 mmag, band- and slope-dependent offset. It very largely does
**not** matter here, because both spectra go through the identical average and
the convention cancels from the ratio to first order; what survives is second
order in the tengri-minus-reference difference across one band.
:func:`convention_sensitivity` measures that residual rather than asserting it,
so the claim is checkable in every run.

Units throughout: wavelengths [Angstrom], :math:`L_\\nu` [erg/s/Hz], integrated
line luminosities [erg/s].
"""

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

__all__ = [
    "BROAD_FILTERS",
    "IR_BANDS",
    "KEY_LINES",
    "RADIO_FREQS",
    "UV_TO_NIR",
    "XRAY_ENERGIES",
    "band_average",
    "convention_sensitivity",
    "filter_rows",
    "filter_rows_native",
    "line_rows",
    "load_filter",
    "overlay_ratio_fig",
    "pivot_wavelength",
    "print_filter_table",
    "print_line_table",
    "print_radio_table",
    "print_window_table",
    "print_xray_table",
    "radio_rows",
    "sweep_fig",
    "window_rows",
    "xray_rows",
]

_ROOT = Path(__file__).resolve().parents[1]
_FILTER_DIR = _ROOT / "data" / "filters"

C_AA = 2.99792458e18  # speed of light [Angstrom/s]


# ---------------------------------------------------------------------------
# The bandpass ladder
# ---------------------------------------------------------------------------

BROAD_FILTERS: tuple[tuple[str, str], ...] = (
    ("GALEX_GALEX_FUV", "GALEX FUV"),
    ("GALEX_GALEX_NUV", "GALEX NUV"),
    ("SLOAN_SDSS_u", "SDSS u"),
    ("SLOAN_SDSS_g", "SDSS g"),
    ("SLOAN_SDSS_r", "SDSS r"),
    ("SLOAN_SDSS_i", "SDSS i"),
    ("SLOAN_SDSS_z", "SDSS z"),
    ("2MASS_2MASS_J", "2MASS J"),
    ("2MASS_2MASS_H", "2MASS H"),
    ("2MASS_2MASS_Ks", "2MASS Ks"),
    ("WISE_WISE_W1", "WISE W1"),
    ("WISE_WISE_W2", "WISE W2"),
    ("Spitzer_IRAC_I4", "IRAC 8um"),
    ("WISE_WISE_W3", "WISE W3"),
    ("WISE_WISE_W4", "WISE W4"),
    ("Spitzer_MIPS_24mu", "MIPS 24"),
    ("Spitzer_MIPS_70mu", "MIPS 70"),
    ("Herschel_Pacs_green", "PACS 100"),
    ("Herschel_Pacs_red", "PACS 160"),
    ("Herschel_SPIRE_PSW", "SPIRE 250"),
    ("Herschel_SPIRE_PMW", "SPIRE 350"),
    ("Herschel_SPIRE_PLW", "SPIRE 500"),
    ("JCMT_SCUBA2_850GHz", "SCUBA2 850"),
)
"""Twenty-three bandpasses from 0.15 to 863 um -- 3.75 decades.

Ordered blue to red by pivot wavelength. Chosen to be the filters an
observer would actually assemble for a panchromatic fit (GALEX + SDSS +
2MASS + WISE/Spitzer + Herschel + SCUBA-2) rather than to tile the axis
evenly, so the table reads like a real SED.
"""


UV_TO_NIR: tuple[tuple[str, str], ...] = BROAD_FILTERS[:10]
"""The GALEX-through-2MASS subset: 0.15 to 2.2 um, ten bands.

For comparisons that are **stellar + attenuation only**. Several reference
codes apply energy balance unconditionally and re-emit the absorbed luminosity
in the IR, while the matched tengri build carries no ``dust.emission`` block,
so past a few um the two are modeling different things and every red band
reads as a large disagreement that is really a scope mismatch.

Handing those comparisons this subset keeps the table honest. Extending one to
the full ladder is a real change -- add a matched dust-emission block to both
sides -- not a wider slice. ``prospector`` does carry Draine & Li 2007 on both
sides and uses the full :data:`BROAD_FILTERS`, which is why its ladder reaches
SCUBA-2.
"""


IR_BANDS: tuple[tuple[str, str], ...] = BROAD_FILTERS[10:]
"""The WISE-through-SCUBA-2 subset: 3.4 to 863 um, thirteen bands.

For comparisons whose subject is the **dust IR alone** -- agnfitter's
peak-normalized cold-dust templates, where there is no stellar continuum to
compare and the UV/optical half of the ladder would be empty.
"""


def _key_lines() -> dict[str, tuple[float, ...]]:
    """The canonical vacuum line table, read from the package.

    Sourced from :data:`tengri.utils.sed_quantities.KEY_LINES` rather than
    restated here: CLAUDE.md forbids local copies of physical constants, and
    the notebooks' own line lists drifted to air wavelengths (4861, 5007,
    6563) while the contract mandates vacuum throughout.
    """
    from tengri.utils.sed_quantities import KEY_LINES as _KL

    return dict(_KL)


KEY_LINES = _key_lines()

_LINE_LABELS = {
    "lya": "Lya 1216",
    "civ_1549": "C IV 1549",
    "oii": "[O II] 3728",
    "hbeta": "Hbeta 4863",
    "oiii_4959": "[O III] 4960",
    "oiii_5007": "[O III] 5008",
    "nii_6548": "[N II] 6550",
    "halpha": "Halpha 6565",
    "nii_6584": "[N II] 6585",
    "sii_6717": "[S II] 6718",
    "sii_6731": "[S II] 6733",
}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def load_filter(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a shipped transmission curve.

    Parameters
    ----------
    name : str
        File stem under ``data/filters`` (e.g. ``"SLOAN_SDSS_g"``).

    Returns
    -------
    wave : ndarray, shape (n_filt,)
        Wavelength [Angstrom], ascending.
    trans : ndarray, shape (n_filt,)
        Transmission (dimensionless), normalized to peak 1.

    Raises
    ------
    FileNotFoundError
        If the curve is not shipped, naming the directory searched -- these
        scripts must never reach outside the repository for data.
    """
    path = _FILTER_DIR / f"{name}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"no transmission curve {name!r} in {_FILTER_DIR}")
    d = np.loadtxt(path)
    w, t = np.asarray(d[:, 0], float), np.asarray(d[:, 1], float)
    order = np.argsort(w)
    w, t = w[order], t[order]
    peak = t.max()
    return w, (t / peak if peak > 0 else t)


def pivot_wavelength(fw: np.ndarray, ft: np.ndarray) -> float:
    """Pivot wavelength [Angstrom] of a transmission curve.

    Parameters
    ----------
    fw, ft : array_like, shape (n_filt,)
        Filter wavelength [Angstrom] and transmission.

    Returns
    -------
    float
        :math:`\\lambda_p = \\sqrt{\\int T\\lambda\\,d\\lambda / \\int (T/\\lambda)\\,d\\lambda}`,
        the wavelength at which :math:`\\langle F_\\nu\\rangle` and
        :math:`\\langle F_\\lambda\\rangle` are exactly interconvertible.
    """
    return float(np.sqrt(np.trapezoid(ft * fw, fw) / np.trapezoid(ft / fw, fw)))


def band_average(
    wave: np.ndarray,
    L_nu: np.ndarray,
    fw: np.ndarray,
    ft: np.ndarray,
    *,
    weight: str = "photon",
) -> float:
    """Bandpass-averaged :math:`L_\\nu` [erg/s/Hz], or NaN if uncovered.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    L_nu : array_like, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz] on ``wave``.
    fw, ft : array_like, shape (n_filt,)
        Filter wavelength [Angstrom] and transmission.
    weight : {"photon", "energy"}, optional
        Bandpass weight :math:`w(\\lambda)`: ``"photon"`` uses
        :math:`1/\\lambda` (default; tengri, DSPS, FSPS, prospector),
        ``"energy"`` uses :math:`1/\\lambda^2` (CIGALE, bagpipes).

    Returns
    -------
    float
        The band average, or ``nan`` when the SED does not span the filter.

    Notes
    -----
    Returns NaN rather than extrapolating. A validator that silently reports a
    number for a band its SED never covered is worse than one that reports a
    gap: the caller prints ``--`` and the reader sees the limit of the
    comparison instead of a fabricated agreement.

    Two ways a band can be uncovered, and both must be caught. The obvious one
    is a grid that stops short. The other is a **zero-filled** SED: the
    comparisons' ``units.regrid`` sets out-of-range samples to 0.0, so a
    spectrum regridded onto a wider reference grid is silently zero past its
    own edge. Reading that as flux gave BAGPIPES ten far-IR bands at
    ``0.000x <-- check`` -- ten false alarms whose real content was "the
    stellar-only comparison does not reach here". Requiring positive flux
    everywhere the transmission is significant catches both, including the
    partial case of a band straddling the edge.
    """
    if weight == "photon":
        w_exp = -1.0
    elif weight == "energy":
        w_exp = -2.0
    else:
        raise ValueError(f"weight must be 'photon' or 'energy', got {weight!r}")

    wave = np.asarray(wave, float)
    L_nu = np.asarray(L_nu, float)
    live = ft > 1e-3
    lo, hi = fw[live].min(), fw[live].max()
    if wave.min() > lo or wave.max() < hi:
        return float("nan")

    L_on_f = np.interp(fw, wave, L_nu)
    if not np.all(L_on_f[live] > 0):
        return float("nan")
    wgt = ft * fw**w_exp
    denom = np.trapezoid(wgt, fw)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.trapezoid(L_on_f * wgt, fw) / denom)


def _filter_rows_inner(
    w_t: np.ndarray,
    L_t: np.ndarray,
    w_ref: np.ndarray,
    L_ref: np.ndarray,
    *,
    filters: tuple[tuple[str, str], ...],
    weight: str,
) -> list[tuple[str, float, float, float, float]]:
    """Shared logic for filter_rows and filter_rows_native.

    Parameters
    ----------
    w_t : array_like, shape (n_wave_t,)
        Rest-frame wavelength grid for tengri [Angstrom].
    L_t : array_like, shape (n_wave_t,)
        tengri :math:`L_\\nu` [erg/s/Hz] on ``w_t``.
    w_ref : array_like, shape (n_wave_ref,)
        Rest-frame wavelength grid for reference [Angstrom].
    L_ref : array_like, shape (n_wave_ref,)
        Reference-code :math:`L_\\nu` [erg/s/Hz] on ``w_ref``.
    filters : tuple of (str, str)
        ``(file stem, label)`` pairs.
    weight : {"photon", "energy"}
        Bandpass weight to pass to :func:`band_average`.

    Returns
    -------
    list of tuple
        ``(label, pivot_um, L_t_band, L_ref_band, ratio)`` per filter.
    """
    rows = []
    for stem, label in filters:
        fw, ft = load_filter(stem)
        a = band_average(w_t, L_t, fw, ft, weight=weight)
        b = band_average(w_ref, L_ref, fw, ft, weight=weight)
        ratio = a / b if (np.isfinite(a) and np.isfinite(b) and b > 0) else float("nan")
        rows.append((label, pivot_wavelength(fw, ft) / 1e4, a, b, ratio))
    return rows


def filter_rows(
    w_ref: np.ndarray,
    L_t: np.ndarray,
    L_ref: np.ndarray,
    *,
    filters: tuple[tuple[str, str], ...] = BROAD_FILTERS,
    weight: str = "photon",
) -> list[tuple[str, float, float, float, float]]:
    """Band-average both SEDs through each filter and form the ratio.

    Both spectra must already share ``w_ref`` -- regrid before calling, so the
    identical operation reaches both sides.

    For line-rich spectra on fine grids compared against coarse reference grids,
    see :func:`filter_rows_native`, which band-averages each side on its own grid
    and avoids aliasing from interpolation.

    Parameters
    ----------
    w_ref : array_like, shape (n_wave,)
        Shared rest-frame wavelength grid [Angstrom].
    L_t, L_ref : array_like, shape (n_wave,)
        tengri and reference-code :math:`L_\\nu` [erg/s/Hz] on ``w_ref``.
    filters : tuple of (str, str), optional
        ``(file stem, label)`` pairs. Defaults to :data:`BROAD_FILTERS`.
    weight : {"photon", "energy"}, optional
        Passed to :func:`band_average`.

    Returns
    -------
    list of tuple
        ``(label, pivot_um, L_t_band, L_ref_band, ratio)`` per filter, in the
        order given. Uncovered bands carry ``nan`` and are kept in the list so
        the caller can show the gap.
    """
    return _filter_rows_inner(w_ref, L_t, w_ref, L_ref, filters=filters, weight=weight)


def filter_rows_native(
    w_t: np.ndarray,
    L_t: np.ndarray,
    w_ref: np.ndarray,
    L_ref: np.ndarray,
    *,
    filters: tuple[tuple[str, str], ...] = BROAD_FILTERS,
    weight: str = "photon",
) -> list[tuple[str, float, float, float, float]]:
    """Band-average each SED on its own grid and form the ratio.

    Each spectrum is band-averaged independently on its native wavelength grid,
    avoiding interpolation artifacts that arise when a fine-gridded spectrum
    (especially one with narrow emission lines) is resampled onto a coarse grid
    before band-averaging.

    Parameters
    ----------
    w_t : array_like, shape (n_wave_t,)
        Rest-frame wavelength grid for tengri [Angstrom].
    L_t : array_like, shape (n_wave_t,)
        tengri :math:`L_\\nu` [erg/s/Hz] on ``w_t``.
    w_ref : array_like, shape (n_wave_ref,)
        Rest-frame wavelength grid for reference [Angstrom].
    L_ref : array_like, shape (n_wave_ref,)
        Reference-code :math:`L_\\nu` [erg/s/Hz] on ``w_ref``.
    filters : tuple of (str, str), optional
        ``(file stem, label)`` pairs. Defaults to :data:`BROAD_FILTERS`.
    weight : {"photon", "energy"}, optional
        Bandpass weight; defaults to ``"photon"`` (tengri, DSPS, FSPS,
        prospector). Both sides are integrated with the same weight and filter
        curves.

    Returns
    -------
    list of tuple
        ``(label, pivot_um, L_t_band, L_ref_band, ratio)`` per filter, in the
        order given. Uncovered bands carry ``nan`` and are kept in the list so
        the caller can show the gap.

    Notes
    -----
    Use this instead of :func:`filter_rows` when comparing a line-rich spectrum
    on a fine grid (e.g., tengri with nebular emission) against a coarse
    reference grid. Interpolating the fine spectrum onto the coarse grid before
    band-averaging aliases narrow lines and distorts the band values; this
    function avoids that by computing band averages on each side's native grid.
    """
    return _filter_rows_inner(w_t, L_t, w_ref, L_ref, filters=filters, weight=weight)


def print_filter_table(
    rows: list[tuple[str, float, float, float, float]],
    *,
    ref_name: str,
    title: str,
    tol: float = 0.05,
    compact: bool = False,
) -> None:
    """Print the band-by-band ratio table.

    Parameters
    ----------
    rows : list of tuple
        As returned by :func:`filter_rows`.
    ref_name : str
        Reference code's name, for the column header.
    title : str
        Heading printed above the table.
    tol : float, optional
        Flag threshold on ``|ratio - 1|``. Default 0.05.
    compact : bool, optional
        Print one summary line instead of the full table -- for the
        deliberately-wrong control configurations, where the point is that
        they are wrong and by how much, not which band. Default False.
    """
    if compact:
        finite = [r[4] for r in rows if np.isfinite(r[4])]
        if not finite:
            print(f"  {title:<52} (no band covered)")
            return
        worst = max(finite, key=lambda x: abs(x - 1.0))
        n_bad = sum(1 for x in finite if abs(x - 1.0) > tol)
        print(
            f"  {title:<52} median {np.median(finite):.3f}x  "
            f"worst {worst:.3f}x  {n_bad}/{len(finite)} outside {tol:.0%}"
        )
        return

    print(f"\n  {title}")
    print(f"  {'band':<12} {'pivot[um]':>10} {'tengri/' + ref_name:>16}")
    print("  " + "-" * 46)
    covered = 0
    for label, piv, _a, _b, ratio in rows:
        if not np.isfinite(ratio):
            print(f"  {label:<12} {piv:>10.3f} {'--':>16}   (outside SED grid)")
            continue
        covered += 1
        flag = " OK" if abs(ratio - 1.0) <= tol else "  <-- check"
        print(f"  {label:<12} {piv:>10.3f} {ratio:>15.3f}x{flag}")
    finite = [r[4] for r in rows if np.isfinite(r[4])]
    if finite:
        print(
            f"  {'':<12} {'':>10} {'median ' + f'{np.median(finite):.3f}x':>16}"
            f"   ({covered}/{len(rows)} bands covered)"
        )


def convention_sensitivity(
    w_ref: np.ndarray,
    L_t: np.ndarray,
    L_ref: np.ndarray,
    *,
    filters: tuple[tuple[str, str], ...] = BROAD_FILTERS,
) -> float:
    """Largest band-ratio shift between the photon and energy conventions.

    Returns
    -------
    float
        ``max |ratio_photon - ratio_energy|`` over covered bands, or ``nan``
        if none are covered.

    Notes
    -----
    The point of the number is to make the module docstring's claim -- that
    the bandpass convention very largely cancels from a tengri/reference ratio
    -- checkable per run rather than asserted once. A value of a few times
    :math:`10^{-3}` says the ratios below are safe to read without knowing
    which convention the reference code used internally.
    """
    a = {r[0]: r[4] for r in filter_rows(w_ref, L_t, L_ref, filters=filters, weight="photon")}
    b = {r[0]: r[4] for r in filter_rows(w_ref, L_t, L_ref, filters=filters, weight="energy")}
    d = [abs(a[k] - b[k]) for k in a if np.isfinite(a[k]) and np.isfinite(b[k])]
    return float(max(d)) if d else float("nan")


# ---------------------------------------------------------------------------
# X-ray
# ---------------------------------------------------------------------------


def _kev_to_hz() -> float:
    """``KEV_TO_HZ`` from the package, not a local literal.

    The X-ray module converts with ``nu = E_keV * KEV_TO_HZ``
    (:mod:`tengri.components.xray.xray`); going through the same constant means
    the validator lands on exactly the frequency the model was evaluated at,
    rather than a hand-carried ``12.398 / E`` that agrees to five digits and
    then drifts.
    """
    from tengri.utils.physics_constants import KEV_TO_HZ

    return float(KEV_TO_HZ)


XRAY_ENERGIES: tuple[tuple[str, float], ...] = (
    ("0.5 keV", 0.5),
    ("1 keV", 1.0),
    ("2 keV", 2.0),
    ("5 keV", 5.0),
    ("10 keV", 10.0),
    ("20 keV", 20.0),
)
"""Rest-frame energies [keV] spanning the soft and hard X-ray bands.

**2 keV is the load-bearing entry**: :math:`\\alpha_{ox}` is defined between
:math:`L_{2500}` and :math:`L_{2\\,\\mathrm{keV}}`, so once both codes are put
on one :math:`L_{2500}` the 2 keV ratio is what the matched anchor predicts.
The others test the corona's *shape* -- photon index and the high-energy
cutoff -- which the anchor does not constrain.
"""


def kev_to_angstrom(e_kev: float) -> float:
    """Photon energy [keV] to wavelength [Angstrom].

    Parameters
    ----------
    e_kev : float
        Photon energy [keV].

    Returns
    -------
    float
        Wavelength [Angstrom].
    """
    return C_AA / (e_kev * _kev_to_hz())


def xray_rows(
    w_ref: np.ndarray,
    L_t: np.ndarray,
    L_ref: np.ndarray,
    *,
    energies: tuple[tuple[str, float], ...] = XRAY_ENERGIES,
) -> list[tuple[str, float, float, float, float]]:
    """Evaluate both SEDs at each X-ray energy and form the ratio.

    Parameters
    ----------
    w_ref : array_like, shape (n_wave,)
        Shared rest-frame wavelength grid [Angstrom].
    L_t, L_ref : array_like, shape (n_wave,)
        tengri and reference-code :math:`L_\\nu` [erg/s/Hz] on ``w_ref``.
    energies : tuple of (str, float), optional
        ``(label, energy_kev)`` pairs. Defaults to :data:`XRAY_ENERGIES`.

    Returns
    -------
    list of tuple
        ``(label, energy_kev, L_t, L_ref, ratio)``; ``nan`` where the grid does
        not reach or either side is non-positive.
    """
    w_ref = np.asarray(w_ref, float)
    order = np.argsort(w_ref)
    w_o = w_ref[order]
    t_o, r_o = np.asarray(L_t, float)[order], np.asarray(L_ref, float)[order]

    rows = []
    for label, e_kev in energies:
        lam = kev_to_angstrom(e_kev)
        if lam < w_o.min() or lam > w_o.max():
            rows.append((label, e_kev, float("nan"), float("nan"), float("nan")))
            continue
        a = float(np.interp(lam, w_o, t_o))
        b = float(np.interp(lam, w_o, r_o))
        ratio = a / b if (a > 0 and b > 0) else float("nan")
        rows.append((label, e_kev, a, b, ratio))
    return rows


def print_xray_table(
    rows: list[tuple[str, float, float, float, float]],
    *,
    ref_name: str,
    title: str,
    tol: float = 0.05,
) -> None:
    """Print the X-ray ratio table and the photon-index difference.

    Parameters
    ----------
    rows : list of tuple
        As returned by :func:`xray_rows`.
    ref_name : str
        Reference code's name, for the column header.
    title : str
        Heading printed above the table.
    tol : float, optional
        Flag threshold on ``|ratio - 1|``. Default 0.05.

    Notes
    -----
    The effective photon index :math:`\\Gamma` is fitted from
    :math:`L_\\nu \\propto E^{1-\\Gamma}` across the covered energies and printed
    for both sides, for the same reason the radio table fits a spectral index:
    a flat offset is a normalization difference (one anchor, one
    :math:`\\alpha_{ox}`) while a tilt is a different corona.
    """
    print(f"\n  {title}")
    print(f"  {'energy':<12} {'keV':>7} {'tengri/' + ref_name:>16}")
    print("  " + "-" * 44)
    for label, e_kev, _a, _b, ratio in rows:
        if not np.isfinite(ratio):
            print(f"  {label:<12} {e_kev:>7.2f} {'--':>16}   (outside SED grid)")
            continue
        flag = " OK" if abs(ratio - 1.0) <= tol else "  <-- check"
        print(f"  {label:<12} {e_kev:>7.2f} {ratio:>15.3f}x{flag}")

    good = [(r[1], r[2], r[3]) for r in rows if np.isfinite(r[4])]
    if len(good) >= 2:
        lg_e = np.log10([g[0] for g in good])
        g_t = 1.0 - float(np.polyfit(lg_e, np.log10([g[1] for g in good]), 1)[0])
        g_r = 1.0 - float(np.polyfit(lg_e, np.log10([g[2] for g in good]), 1)[0])
        print(
            f"  photon index Gamma (L_nu ~ E^(1-Gamma)): "
            f"tengri {g_t:.3f}, {ref_name} {g_r:.3f}  (delta {g_t - g_r:+.3f})"
        )


# ---------------------------------------------------------------------------
# Radio
# ---------------------------------------------------------------------------

RADIO_FREQS: tuple[tuple[str, float], ...] = (
    ("LOFAR 150MHz", 0.15e9),
    ("GMRT 325MHz", 0.325e9),
    ("VLA 1.4GHz", 1.4e9),
    ("VLA 3GHz", 3.0e9),
    ("VLA 5GHz", 5.0e9),
    ("VLA 10GHz", 10.0e9),
    ("ALMA B3 100GHz", 100.0e9),
)
"""Standard survey frequencies [Hz], 150 MHz to 100 GHz.

Radio is compared at discrete frequencies rather than through bandpasses, for
the plain reason that no cm-wave transmission curves ship in ``data/filters``
(the reddest are sub-mm: LABOCA 345 GHz, SCUBA-2). It is also how the
measurement is actually quoted -- an interferometer delivers a flux at a
frequency, not a broadband magnitude -- so nothing is lost.

Spanning 150 MHz to 100 GHz matters for what it tests: the synchrotron slope is
a *power law*, so a ratio at one frequency conflates normalization with slope,
and only a lever arm separates them. 150 MHz to 10 GHz is nearly two decades.
"""


def radio_rows(
    w_ref: np.ndarray,
    L_t: np.ndarray,
    L_ref: np.ndarray,
    *,
    freqs: tuple[tuple[str, float], ...] = RADIO_FREQS,
) -> list[tuple[str, float, float, float, float]]:
    """Evaluate both SEDs at each radio frequency and form the ratio.

    Parameters
    ----------
    w_ref : array_like, shape (n_wave,)
        Shared rest-frame wavelength grid [Angstrom].
    L_t, L_ref : array_like, shape (n_wave,)
        tengri and reference-code :math:`L_\\nu` [erg/s/Hz] on ``w_ref``.
    freqs : tuple of (str, float), optional
        ``(label, frequency_hz)`` pairs. Defaults to :data:`RADIO_FREQS`.

    Returns
    -------
    list of tuple
        ``(label, freq_ghz, L_t, L_ref, ratio)``. Frequencies outside the grid,
        or where either side is non-positive, carry ``nan``.
    """
    w_ref = np.asarray(w_ref, float)
    order = np.argsort(w_ref)
    w_o = w_ref[order]
    t_o, r_o = np.asarray(L_t, float)[order], np.asarray(L_ref, float)[order]

    rows = []
    for label, nu in freqs:
        lam = C_AA / nu  # [Angstrom]
        if lam < w_o.min() or lam > w_o.max():
            rows.append((label, nu / 1e9, float("nan"), float("nan"), float("nan")))
            continue
        a = float(np.interp(lam, w_o, t_o))
        b = float(np.interp(lam, w_o, r_o))
        ratio = a / b if (a > 0 and b > 0) else float("nan")
        rows.append((label, nu / 1e9, a, b, ratio))
    return rows


def print_radio_table(
    rows: list[tuple[str, float, float, float, float]],
    *,
    ref_name: str,
    title: str,
    tol: float = 0.05,
) -> None:
    """Print the radio ratio table, plus the spectral index on both sides.

    Parameters
    ----------
    rows : list of tuple
        As returned by :func:`radio_rows`.
    ref_name : str
        Reference code's name, for the column header.
    title : str
        Heading printed above the table.
    tol : float, optional
        Flag threshold on ``|ratio - 1|``. Default 0.05.

    Notes
    -----
    The spectral index :math:`\\alpha` in :math:`L_\\nu \\propto \\nu^{-\\alpha}`
    is fitted across the covered frequencies and printed for each side, because
    it is the number the ratios cannot give. A constant offset at every
    frequency is a normalization difference -- one q_IR, one radio loudness --
    while a drift across the decade is a *slope* difference, which is a
    different model rather than a different scaling. Two codes can agree at
    1.4 GHz and disagree about everything else.
    """
    print(f"\n  {title}")
    print(f"  {'band':<16} {'GHz':>8} {'tengri/' + ref_name:>16}")
    print("  " + "-" * 46)
    for label, ghz, _a, _b, ratio in rows:
        if not np.isfinite(ratio):
            print(f"  {label:<16} {ghz:>8.3f} {'--':>16}   (outside SED grid)")
            continue
        flag = " OK" if abs(ratio - 1.0) <= tol else "  <-- check"
        print(f"  {label:<16} {ghz:>8.3f} {ratio:>15.3f}x{flag}")

    good = [(r[1], r[2], r[3]) for r in rows if np.isfinite(r[4])]
    if len(good) >= 2:
        lg_nu = np.log10([g[0] for g in good])
        a_t = -float(np.polyfit(lg_nu, np.log10([g[1] for g in good]), 1)[0])
        a_r = -float(np.polyfit(lg_nu, np.log10([g[2] for g in good]), 1)[0])
        print(
            f"  spectral index alpha (L_nu ~ nu^-alpha): "
            f"tengri {a_t:.3f}, {ref_name} {a_r:.3f}  (delta {a_t - a_r:+.3f})"
        )


# ---------------------------------------------------------------------------
# Emission lines
# ---------------------------------------------------------------------------


def safe_half_widths(
    lines: dict[str, tuple[float, ...]],
    *,
    half: float = 12.0,
    pad: float = 0.45,
) -> dict[str, float]:
    """Per-line integration half-width that cannot reach a neighboring line.

    Parameters
    ----------
    lines : dict
        ``{key: (wavelength, ...)}`` [Angstrom].
    half : float, optional
        Requested half-width [Angstrom]. Default 12.
    pad : float, optional
        Fraction of the nearest-neighbor separation allowed. Default 0.45,
        i.e. two adjacent windows leave a 10% gap between them.

    Returns
    -------
    dict
        ``{key: half_width}`` [Angstrom], each ``<= half``.

    Notes
    -----
    A fixed 12 A half-width silently blends the Halpha complex. The vacuum
    separations are 14.8 A (Halpha--[N II] 6550) and 20.7 A (Halpha--[N II]
    6585), and [S II] 6718/6733 are 14.4 A apart, so the +/-12 A windows
    overlap and ``line_lum`` -- which subtracts the in-window floor as a flat
    continuum -- integrates the *neighbor's wing*.

    Caught by injecting a Balmer pair of known ratio: both [N II] entries
    reported Halpha's ratio to three decimals, which is what a blend looks
    like when it is not looked for. Narrowing per line is not a full
    deblending (nothing measured off a spectrum can fully separate lines this
    close once they are broadened) but it stops the measurement being
    dominated by a neighbor, and :func:`line_rows` reports which lines were
    narrowed.

    A multi-component entry is treated as **one feature**, centered on the span
    of its components and wide enough to hold all of them. ``KEY_LINES`` lists
    [O II] 3727+3730 and C IV 1548+1551 as single entries precisely because
    their components are summed. Two mistakes are avoided: letting a doublet
    partner count as a contaminant (which collapsed those windows to 1.4 and
    1.2 A and measured a fraction of each) and summing two overlapping
    per-component windows (which double-counts the flux between them).

    Returns the **total** half-width, measured from the feature center, so it
    already includes the doublet's own half-span.
    """
    out = {}
    for key, cs in lines.items():
        c0 = 0.5 * (min(cs) + max(cs))
        span = 0.5 * (max(cs) - min(cs))
        foreign = [c for k, other in lines.items() if k != key for c in other]
        extra = half
        if foreign:
            gap = min(abs(x - c0) - span for x in foreign)
            extra = min(half, pad * max(gap, 0.0))
        out[key] = float(span + extra)
    return out


def line_rows(
    w_t: np.ndarray,
    L_t: np.ndarray,
    w_ref: np.ndarray,
    L_ref: np.ndarray,
    *,
    line_lum,
    lines: dict[str, tuple[float, ...]] | None = None,
    half: float = 12.0,
    min_rel: float = 1e-4,
) -> list[tuple[str, float, float, float, float, float]]:
    """Continuum-subtracted line luminosities on both sides, and their ratio.

    Each side is measured on its **own** grid -- unlike the photometry, lines
    must not be regridded first, because interpolating a resolved line onto a
    coarser grid does not conserve its flux.

    Parameters
    ----------
    w_t, L_t : array_like, shape (n_t,)
        tengri wavelength [Angstrom] and nebular-only :math:`L_\\nu` [erg/s/Hz].
    w_ref, L_ref : array_like, shape (n_r,)
        Same for the reference code.
    line_lum : callable
        The comparison's ``units.line_lum(wave, L_nu, center, half=...)``.
        Passed in rather than reimplemented so each validator measures lines
        exactly as its notebook does.
    lines : dict, optional
        ``{key: (wavelength, ...)}``; doublet components are summed. Defaults
        to the package's vacuum :data:`KEY_LINES`.
    half : float, optional
        Requested half-width [Angstrom], narrowed per line by
        :func:`safe_half_widths` so a window cannot reach its neighbor.
        Default 12.
    min_rel : float, optional
        Detection floor as a fraction of the brightest line measured on the
        same side. Default ``1e-4``.

    Returns
    -------
    list of tuple
        ``(label, wavelength_aa, L_t_line, L_ref_line, ratio, half_used)`` per
        line, in ascending wavelength. ``half_used`` is below ``half`` for
        blend-limited lines. ``ratio`` is ``nan`` unless the line clears
        ``min_rel`` on **both** sides.

    Notes
    -----
    The floor exists because the ratio of two *undetected* lines is
    numerically well-defined and physically meaningless. Injecting a Balmer
    pair with no [N II] at all, both sides return the numerical residue of
    Halpha's wing and their ratio reproduces Halpha's to three decimals -- a
    confident-looking number built from nothing. A reference code that does not
    carry a line must read as absent, not as agreement.

    The floor is relative to the brightest line rather than an equivalent
    width. EW is the more physical statistic but it divides by the local
    continuum, and a nebular-*only* spectrum (which is what these validators
    compare, the continuum having been separated out) can have essentially no
    continuum under a line, so the EW of a noise residue diverges instead of
    vanishing -- backwards for a detection test.
    """
    lines = KEY_LINES if lines is None else lines
    halves = safe_half_widths(lines, half=half)

    meas = []
    for key, centers in sorted(lines.items(), key=lambda kv: kv[1][0]):
        h = halves[key]
        # One window per feature, centered on the component span: summing
        # per-component windows would double-count an unresolved doublet.
        c0 = 0.5 * (min(centers) + max(centers))
        a = line_lum(w_t, L_t, c0, half=h)
        b = line_lum(w_ref, L_ref, c0, half=h)
        meas.append((key, centers[0], a, b, h))

    max_t = max((m[2] for m in meas), default=0.0)
    max_r = max((m[3] for m in meas), default=0.0)

    out = []
    for key, lam, a, b, h in meas:
        detected = (max_t > 0 and a >= min_rel * max_t) and (max_r > 0 and b >= min_rel * max_r)
        ratio = a / b if (detected and b > 0) else float("nan")
        out.append((_LINE_LABELS.get(key, key), lam, a, b, ratio, h))
    return out


def print_line_table(
    rows: list[tuple[str, float, float, float, float]],
    *,
    ref_name: str,
    title: str,
    normalize_to: str = "Hbeta 4863",
) -> None:
    """Print the line table, absolute and normalized to a reference line.

    Parameters
    ----------
    rows : list of tuple
        As returned by :func:`line_rows`.
    ref_name : str
        Reference code's name, for the column header.
    title : str
        Heading printed above the table.
    normalize_to : str, optional
        Label of the line to divide through by. Default ``"Hbeta 4863"``.

    Notes
    -----
    Both columns are printed because they answer different questions and only
    the second is a physics comparison.

    The **absolute** ratio folds in the whole ionizing budget -- Q_H, the
    escape fraction, the SSP's Lyman continuum -- so it mostly measures
    normalization. The ratio **normalized to Hbeta** divides that out and
    leaves the line-ratio pattern, which is what the photoionization model
    actually predicts.

    Neither is a parity check, and this table carries no OK flag by design.
    tengri's Cue emulator and the reference codes' Cloudy grids (13.x, 17,
    23.01, 25 depending on the code) are *different models*, so a residual
    here is a model difference to quantify, not a defect to fix. Reporting it
    with a pass/fail flag would invite exactly the misreading the reproduction
    notebooks' nebular sections spend paragraphs heading off.
    """
    print(f"\n  {title}")
    ref_row = next((r for r in rows if r[0] == normalize_to), None)
    hb_t, hb_r = (ref_row[2], ref_row[3]) if ref_row else (0.0, 0.0)

    norm_hdr = f"/{normalize_to.split()[0]}"
    print(f"  {'line':<14} {'lam[A]':>9} {'tengri/' + ref_name:>16} {norm_hdr:>12}  {'+/-A':>6}")
    print("  " + "-" * 64)
    blended = []
    for label, lam, a, b, ratio, hw in rows:
        mark = "" if hw >= 12.0 else " b"
        if hw < 12.0:
            blended.append(label)
        if not np.isfinite(ratio) or ratio == 0.0:
            print(f"  {label:<14} {lam:>9.1f} {'--':>16} {'--':>12}  {hw:>5.1f}{mark}")
            continue
        if hb_t > 0 and hb_r > 0:
            norm_s = f"{(a / hb_t) / (b / hb_r):.3f}x"
        else:
            norm_s = "--"
        print(f"  {label:<14} {lam:>9.1f} {ratio:>15.3f}x {norm_s:>12}  {hw:>5.1f}{mark}")
    if blended:
        print(f"  b = window narrowed below 12 A to clear a neighbor: {', '.join(blended)}")
    print("  (no pass/fail flag: Cue vs Cloudy is a model difference, not a parity check)")


# ---------------------------------------------------------------------------
# Sweep helpers for reproducibility notebooks
# ---------------------------------------------------------------------------


# Above this ratio between the largest and smallest swept value, ramp
# positions are taken in log space so the pale end does not collapse.
_RAMP_LOG_SPAN = 5.0
# The ratio panel ignores points where the reference is below this fraction of
# its own peak, and never opens wider than these bounds.
_RATIO_SIGNAL_FLOOR = 1e-3
_RATIO_YLIM_BOUND = (0.05, 20.0)

def _marker_subsample(x, n=26, window=None):
    """Indices of about ``n`` points spread evenly across ``x`` for a marker overlay.

    Parameters
    ----------
    x : array_like, shape (n_pts,)
        Ascending plot abscissa.
    n : int, optional
        Target number of markers. Default 26 -- dense enough to trace a curve,
        sparse enough that several cases can share one axis.
    window : tuple of float | None, optional
        ``(lo, hi)`` axis limits. Markers are spread across this range rather
        than across the whole array, which usually extends past the frame.

    Returns
    -------
    ndarray
        Integer indices into ``x``.
    """
    x = np.asarray(x)
    keep = np.isfinite(x)
    if window is not None:
        keep &= (x >= window[0]) & (x <= window[1])
        if not np.any(keep):
            keep = np.isfinite(x)
    idx = np.flatnonzero(keep)
    if idx.size <= n:
        return idx
    xv = x[idx]
    if np.all(xv > 0):
        coord = np.log10(xv)
    else:
        coord = xv
    targets = np.linspace(coord[0], coord[-1], n)
    picks = np.searchsorted(coord, targets)
    return np.unique(idx[np.clip(picks, 0, idx.size - 1)])


def _code_style_handles(ref_label):
    """Two proxy artists naming which mark belongs to which code."""
    return [
        Line2D(
            [],
            [],
            color="0.30",
            marker="o",
            markerfacecolor="none",
            markeredgewidth=1.3,
            markersize=6.0,
            linestyle="none",
            label=ref_label,
        ),
        Line2D([], [], color="0.30", linestyle="-", linewidth=1.8, label="tengri"),
    ]


def _ascending(x, y):
    """Return ``(x, y)`` reordered so ``x`` increases.

    Parameters
    ----------
    x : array_like, shape (n,)
        Sample axis — wavelength [Angstrom] or time [yr or Gyr].
    y : array_like, shape (n,)
        Values at ``x``.

    Returns
    -------
    x_sorted : ndarray, shape (n,)
        ``x`` in increasing order.
    y_sorted : ndarray, shape (n,)
        ``y`` permuted to match.

    Notes
    -----
    :func:`numpy.interp` requires an increasing sample axis and does not verify
    it; given a descending one it returns the fill value at every point. A
    cosmic-age axis built from a lookback-time grid descends, so both arms are
    reordered here rather than at each call site.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size > 1 and x[0] > x[-1]:
        order = np.argsort(x)
        return x[order], y[order]
    return x, y


def sweep_fig(
    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    ref_label: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    x_of_wave=None,
    xlabel: str = r"$\lambda$ [Å]",
    ylabel: str = r"$L_\nu$ [erg/s/Hz]",
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    band: tuple[float, float] = (0.9, 1.1),
    logy: bool = True,
    dyn_range: float = 1e-5,
    figsize: tuple[float, float] = (8.5, 6.0),
    cmap: str | None = None,
    values: list[float] | None = None,
    param_label: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, np.ndarray]]:
    """Overlay multiple SED comparisons with a ratio panel, one figure.

    Each case shows the reference curve as a thick translucent solid line and
    the tengri curve as a thin dashed line. Tengri is regridded onto the
    reference wavelength grid via :func:`np.interp`. The bottom panel displays
    the tengri/reference ratio for each case, with a shaded tolerance band.

    When ``cmap`` is provided, the swept parameter's value encodes the curve
    colour as a sequential ramp (light for low, dark for high), while line
    style (solid for reference, dashed for tengri) carries code identity.
    The ratio curves carry the same colour as their cases. When ``values`` and
    ``param_label`` are both given, a colourbar replaces the per-case legend.

    Parameters
    ----------
    cases : list of tuple
        Each tuple is ``(label, w_ref, L_ref, w_t, L_t)`` where:

        - ``label`` : str
            Case name for the legend.
        - ``w_ref`` : ndarray, shape (n_wave,)
            Reference wavelength grid [Å].
        - ``L_ref`` : ndarray, shape (n_wave,)
            Reference L_ν [erg/s/Hz] on w_ref.
        - ``w_t`` : ndarray, shape (n_wave,)
            Tengri wavelength grid [Å].
        - ``L_t`` : ndarray, shape (n_wave,)
            Tengri L_ν [erg/s/Hz] on w_t.
    ref_label : str
        Label for the reference curves in the legend (e.g., "CIGALE").
    title : str
        Panel title.
    xlim : tuple, optional
        x-limits in the plotted abscissa units.
    x_of_wave : callable, optional
        Maps wavelength [Å] to the plotted x-axis (e.g., ``lambda w: w / 1e4``
        for µm). Identity (wavelength in Å) when ``None``.
    xlabel : str, optional
        x-axis label. Default: "wavelength [µm]".
    ylabel : str, optional
        Top panel y-axis label. Default: "L_ν [erg/s/Hz]".
    ratio_ylim : tuple, optional
        y-limits on the ratio panel. Default (0.5, 1.5).
    band : tuple, optional
        Shaded tolerance band (lo, hi) on the ratio panel. Default (0.9, 1.1).
    dyn_range : float, optional
        Fraction of the brightest plotted value at which the y-axis floor is
        placed, so a sweep covering a few decades is not drawn on an axis
        spanning many. Default ``1e-5``. Applies only when ``logy`` is True.
    logy : bool, optional
        Use log scale on top panel y-axis. Default True.
    figsize : tuple, optional
        Figure size (width, height). Default (8.5, 6.0).
    cmap : str | None, optional
        Single-hue sequential colourmap name (e.g., ``"Blues"``, ``"Oranges"``,
        ``"Reds"``, ``"Greys"``, ``"Purples"``, ``"Greens"``), or ``None`` to
        use matplotlib's default discrete cycle. Default: ``"Blues"``. Set to
        ``None`` when cases are model *families* rather than values of one
        parameter.
    values : list of float | None, optional
        Numeric values of the swept parameter in case order, one per case.
        When given with ``param_label``, each case's colour is placed at its
        value's position in the range [min, max], so unequal steps read
        correctly. Default: ``None`` (case index used instead).
    param_label : str | None, optional
        Parameter name and units (e.g., ``"temperature [K]"``). When both
        ``values`` and ``param_label`` are given, a colourbar is drawn
        instead of a per-case legend; the colourbar becomes the legend for
        the continuous encoding. Default: ``None``.

    Returns
    -------
    fig : plt.Figure
        The figure.
    (ax, ax_ratio) : tuple of plt.Axes
        Top (main SED) and bottom (ratio) axes.
    ratios : dict
        Mapping case label to the ratio array (tengri/reference) on w_ref.

    Raises
    ------
    ValueError
        If any case has no overlapping region where both arms are positive.
    """
    from matplotlib.colors import LogNorm, Normalize

    get_cmap = plt.get_cmap

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ratios: dict[str, np.ndarray] = {}
    _plotted: list[tuple[np.ndarray, np.ndarray]] = []
    _r_sig: list[np.ndarray] = []

    # Determine colours
    n_cases = len(cases)
    if cmap is None:
        # Use default categorical colours
        colors = [f"C{i}" for i in range(n_cases)]
    else:
        # Use sequential colourmap
        colormap_obj = get_cmap(cmap)
        if values is not None:
            # Map values to colour positions in [0.25, 0.95]
            values_arr = np.asarray(values, dtype=float)
            val_min, val_max = values_arr.min(), values_arr.max()
            # A parameter spanning several factors is positioned by its logarithm:
            # linear placement crushes every small value into the pale end of the
            # ramp, where neighbours differ by less than the eye resolves.
            log_scaled = bool(val_min > 0 and val_max / val_min >= _RAMP_LOG_SPAN)
            if val_max > val_min:
                if log_scaled:
                    lo, hi = np.log10(val_min), np.log10(val_max)
                    norm_values = (np.log10(values_arr) - lo) / (hi - lo)
                else:
                    norm_values = (values_arr - val_min) / (val_max - val_min)
            else:
                # All values identical
                norm_values = np.full_like(values_arr, 0.5)
            # Map to [0.25, 0.95] ramp
            ramp_positions = 0.25 + 0.7 * norm_values
            colors = [colormap_obj(pos) for pos in ramp_positions]
        else:
            # Distribute evenly in [0.25, 0.95]
            ramp_positions = np.linspace(0.25, 0.95, n_cases)
            colors = [colormap_obj(pos) for pos in ramp_positions]

    for (label, w_ref, L_ref, w_t, L_t), color in zip(cases, colors):
        # Regrid tengri onto reference wavelength grid
        w_ref, L_ref = _ascending(w_ref, L_ref)
        w_t, L_t = _ascending(w_t, L_t)
        L_t_on_ref = np.interp(w_ref, w_t, L_t, left=0.0, right=0.0)
        ratio = np.divide(L_t_on_ref, L_ref, where=(L_ref > 0), out=np.full_like(L_ref, np.nan))
        ratios[label] = ratio
        # Where the reference is a tiny fraction of its own peak, the quotient is
        # division by noise. Those points stay in the returned array (the tables
        # window them themselves) but must not set the panel's scale.
        _sig = L_ref > (L_ref.max() * _RATIO_SIGNAL_FLOOR) if L_ref.size else np.zeros(0, bool)
        _r_sig.append(ratio[_sig & np.isfinite(ratio)])

        # Check that both arms have overlapping positive region
        pos_ref = L_ref > 0
        pos_t = L_t_on_ref > 0
        overlap_positive = pos_ref & pos_t
        if not np.any(overlap_positive):
            raise ValueError(
                f"No overlapping positive region for case {label!r}. "
                f"Reference range: [{w_ref.min():.4e}, {w_ref.max():.4e}], "
                f"Tengri range: [{w_t.min():.4e}, {w_t.max():.4e}]. "
                f"After regridding, no points have both L_ref > 0 and L_t > 0."
            )

        # Check that both arms have overlapping positive region
        pos_ref = L_ref > 0
        pos_t = L_t_on_ref > 0
        overlap_positive = pos_ref & pos_t
        if not np.any(overlap_positive):
            raise ValueError(
                f"No overlapping positive region for case {label!r}. "
                f"Reference range: [{w_ref.min():.4e}, {w_ref.max():.4e}], "
                f"Tengri range: [{w_t.min():.4e}, {w_t.max():.4e}]. "
                f"After regridding, no points have both L_ref > 0 and L_t > 0."
            )

        # Transform x-axis if needed
        x_ref = w_ref if x_of_wave is None else x_of_wave(w_ref)

        # The reference is a wide pale band and tengri a narrow saturated line on
        # top of it, both in the case colour. Two same-width lines in one colour
        # read as a single line wherever they agree, which is most of the axis and
        # is exactly where the reader needs to see that they agree.
        _xr, _yr = x_ref[pos_ref], L_ref[pos_ref]
        _mk = _marker_subsample(_xr, window=xlim)
        ax.plot(
            _xr[_mk],
            _yr[_mk],
            color=color,
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgewidth=1.3,
            markersize=5.6,
            zorder=3,
        )
        ax.plot(
            x_ref[pos_t],
            L_t_on_ref[pos_t],
            color=color,
            linestyle="-",
            linewidth=1.7,
            label=label,
            zorder=2,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()],
        )
        _plotted.append((x_ref[pos_ref], L_ref[pos_ref]))
        _plotted.append((x_ref[pos_t], L_t_on_ref[pos_t]))

        # Ratio panel, over the overlap and only where the reference carries
        # signal. At an emission-line pixel where the reference is a fraction of
        # its own peak the quotient is division by noise, and those spikes bury
        # the few-percent structure the panel exists to show.
        _r_plot = np.where(overlap_positive & _sig, ratio, np.nan)
        ax_r.plot(
            x_ref,
            _r_plot,
            color=color,
            linestyle="-",
            linewidth=1.3,
            path_effects=[pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()],
        )

    # Configure top panel
    ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if logy:
        ax.set_yscale("log")
        _vals = []
        for _x, _y in _plotted:
            _keep = np.isfinite(_y) & (_y > 0)
            if xlim is not None:
                _keep &= (_x >= xlim[0]) & (_x <= xlim[1])
            if np.any(_keep):
                _vals.append(_y[_keep])
        if _vals:
            _pos = np.concatenate(_vals)
            _ymx, _ymn = float(_pos.max()), float(_pos.min())
            # Two bounds, whichever is higher: a fixed fraction of the peak, and
            # just under the smallest value drawn. The second is what keeps a
            # sweep covering three decades off an axis spanning eighteen.
            # The peak can be a single line pixel; scale the head of the axis on
            # the 99.5th percentile so a narrow spike does not push every
            # continuum curve into the lower half of the frame.
            _ytop = float(np.percentile(_pos, 99.5))
            ax.set_ylim(max(_ymx * dyn_range, _ymn * 0.5), max(_ytop * 3.0, _ymn * 10.0))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend or colourbar
    if param_label is not None and values is not None:
        # Add a colourbar instead of a legend
        _v = np.asarray(values, dtype=float)
        _log_cb = bool(_v.min() > 0 and _v.max() / _v.min() >= _RAMP_LOG_SPAN)
        sm = plt.cm.ScalarMappable(
            cmap=get_cmap(cmap) if cmap is not None else "viridis",
            norm=(
                LogNorm(vmin=_v.min(), vmax=_v.max())
                if _log_cb
                else Normalize(vmin=_v.min(), vmax=_v.max())
            ),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=param_label, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    else:
        # Per-case legend; the code legend below is separate so that which mark
        # belongs to which code never depends on reading a colour.
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            _case_leg = ax.legend(handles, labels, fontsize=9, loc="upper left")
            ax.add_artist(_case_leg)

    fig.legend(
        handles=_code_style_handles(ref_label),
        fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.998),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    # Configure ratio panel
    ax_r.axhspan(*band, color="0.85", zorder=0)
    ax_r.axhline(1.0, color="0.5", linewidth=0.8)
    ax_r.set_xscale("log")
    _fin = [a for a in _r_sig if a.size]
    if _fin:
        # ratio_ylim is a floor on the window, not a crop: a case sitting at 0.45
        # was drawn against the bottom edge and read as "0.5 or less". Widen to
        # hold the signal-region spread, bounded so one spike cannot flatten the
        # panel everyone else lives in.
        _lo_d, _hi_d = np.percentile(np.concatenate(_fin), [2.0, 98.0])
        ax_r.set_ylim(
            float(np.clip(min(ratio_ylim[0], _lo_d * 0.95), _RATIO_YLIM_BOUND[0], 1.0)),
            float(np.clip(max(ratio_ylim[1], _hi_d * 1.05), 1.0, _RATIO_YLIM_BOUND[1])),
        )
    else:
        ax_r.set_ylim(*ratio_ylim)
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel("tengri / ref", fontsize=9)
    ax_r.grid(True, alpha=0.3)

    return fig, (ax, ax_r), ratios


def window_rows(
    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    lo: float,
    hi: float,
    rel_to: str = "point",
    peak: float | None = None,
) -> list[dict]:
    """Compute ratio statistics within a wavelength window for each case.

    For each case, tengri is regridded onto the reference wavelength grid,
    then the tengri/reference ratio is computed and filtered to the window
    [lo, hi] in wavelength. The median ratio and the maximum absolute
    deviation from unity are computed over the windowed points.

    Parameters
    ----------
    cases : list of tuple
        Each tuple is ``(label, w_ref, L_ref, w_t, L_t)`` as in :func:`sweep_fig`.
    lo, hi : float
        Wavelength window bounds [Å].
    rel_to : str, optional
        Deviation calculation mode. Either "point" (default) or "peak".

        - "point": max_abs_dev = max(|L_t/L_ref - 1|); median_ratio over all finite ratios.
        - "peak": Compute peak = max(L_ref[mask]); max_abs_dev = max(|L_t - L_ref|[mask]) / peak;
          median_ratio over points where L_ref >= 0.01 * peak.

    peak : float, optional
        Normalization scale for ``rel_to="peak"`` mode. If provided, overrides the
        window maximum as the denominator for max_abs_dev and as the reference for
        the 1% threshold of the median (points with L_ref >= 0.01 * peak).
        Must be positive. Only applies when ``rel_to="peak"``.

    Returns
    -------
    list of dict
        One dict per case with keys:

        - ``"label"`` : str
        - ``"rel_to"`` : str
            Mode used ("point" or "peak").
        - ``"median_ratio"`` : float
            Median of tengri/reference ratios (or relative difference) in the window.
        - ``"max_abs_dev"`` : float
            Maximum absolute deviation in the window.
        - ``"x_at_max"`` : float
            Wavelength where max_abs_dev occurs.
    """
    if rel_to not in ("point", "peak"):
        raise ValueError(f"rel_to must be 'point' or 'peak', got {rel_to!r}")

    if peak is not None and rel_to != "peak":
        raise ValueError("peak= applies to rel_to='peak' only")

    if peak is not None and peak <= 0:
        raise ValueError("peak must be positive")

    rows = []
    for label, w_ref, L_ref, w_t, L_t in cases:
        # Regrid tengri onto reference grid
        if w_t.shape == w_ref.shape and np.allclose(w_t, w_ref):
            L_t_on_ref = L_t
        else:
            L_t_on_ref = np.interp(w_ref, w_t, L_t, left=0.0, right=0.0)

        # Filter to wavelength window
        in_window = (w_ref >= lo) & (w_ref <= hi)
        L_ref_in_window = L_ref[in_window]
        L_t_in_window = L_t_on_ref[in_window]
        w_in_window = w_ref[in_window]

        # Compute statistics based on mode
        if rel_to == "point":
            out_array = np.full_like(L_ref_in_window, np.nan)
            ratio = np.divide(
                L_t_in_window,
                L_ref_in_window,
                where=(L_ref_in_window > 0),
                out=out_array,
            )
            finite_ratios = ratio[np.isfinite(ratio)]

            if len(finite_ratios) == 0:
                median_ratio = float("nan")
                max_abs_dev = float("nan")
                x_at_max = float("nan")
            else:
                median_ratio = float(np.median(finite_ratios))
                deviations = np.abs(finite_ratios - 1.0)
                max_abs_dev = float(np.max(deviations))
                idx_max = np.argmax(deviations)
                x_at_max = float(w_in_window[np.isfinite(ratio)][idx_max])

        else:  # rel_to == "peak"
            # Find peak in window or use provided peak
            valid_mask = L_ref_in_window > 0
            if peak is None:
                # Use window maximum
                if not np.any(valid_mask):
                    median_ratio = float("nan")
                    max_abs_dev = float("nan")
                    x_at_max = float("nan")
                else:
                    peak_val = float(np.max(L_ref_in_window[valid_mask]))

                    # Absolute deviation
                    abs_diffs = np.abs(L_t_in_window - L_ref_in_window)
                    max_abs_dev = float(np.max(abs_diffs) / peak_val)
                    idx_max = np.argmax(abs_diffs)
                    x_at_max = float(w_in_window[idx_max])

                    # Median ratio over significant points
                    threshold = 0.01 * peak_val
                    significant_mask = (L_ref_in_window >= threshold) & (
                        L_ref_in_window > 0
                    )
                    if np.any(significant_mask):
                        L_t_sig = L_t_in_window[significant_mask]
                        L_ref_sig = L_ref_in_window[significant_mask]
                        out_array = np.full_like(L_ref_sig, np.nan)
                        ratio_significant = np.divide(
                            L_t_sig, L_ref_sig, where=True, out=out_array
                        )
                        median_ratio = float(np.median(ratio_significant))
                    else:
                        median_ratio = float("nan")
            else:
                # Use provided peak
                peak_val = peak

                # Absolute deviation
                abs_diffs = np.abs(L_t_in_window - L_ref_in_window)
                max_abs_dev = float(np.max(abs_diffs) / peak_val)
                idx_max = np.argmax(abs_diffs)
                x_at_max = float(w_in_window[idx_max])

                # Median ratio over significant points
                threshold = 0.01 * peak_val
                significant_mask = (L_ref_in_window >= threshold) & (
                    L_ref_in_window > 0
                )
                if np.any(significant_mask):
                    L_t_sig = L_t_in_window[significant_mask]
                    L_ref_sig = L_ref_in_window[significant_mask]
                    out_array = np.full_like(L_ref_sig, np.nan)
                    ratio_significant = np.divide(
                        L_t_sig, L_ref_sig, where=True, out=out_array
                    )
                    median_ratio = float(np.median(ratio_significant))
                else:
                    median_ratio = float("nan")

        rows.append(
            {
                "label": label,
                "rel_to": rel_to,
                "median_ratio": median_ratio,
                "max_abs_dev": max_abs_dev,
                "x_at_max": x_at_max,
            }
        )

    return rows


def print_window_table(
    rows: list[dict],
    *,
    ref_name: str,
    title: str,
    tol: float = 0.05,
    x_unit: str = "Å",
    x_scale: float = 1.0,
) -> None:
    """Print wavelength-window statistics table.

    Displays median ratio and maximum absolute deviation for each case,
    with a ``<-- check`` flag when max_abs_dev exceeds ``tol``.

    Parameters
    ----------
    rows : list of dict
        As returned by :func:`window_rows`.
    ref_name : str
        Reference code name, for the column header.
    title : str
        Table heading.
    tol : float, optional
        Deviation threshold for the check flag. Default 0.05.
    x_unit : str, optional
        Unit label for the x-axis position column. Default "Å".
    x_scale : float, optional
        Scale factor to apply to x_at_max values. Default 1.0.
    """
    # Determine rel_to mode and validate consistency
    rel_to_modes = set()
    for row in rows:
        mode = row.get("rel_to", "point")
        rel_to_modes.add(mode)

    if len(rel_to_modes) > 1:
        raise ValueError(f"All rows must have the same rel_to mode. Found: {rel_to_modes}")

    is_peak_mode = "peak" in rel_to_modes
    dev_header = "max |Δ| [% peak]" if is_peak_mode else "max |Δ| [%]"

    print(f"\n  {title}")
    print(f"  {'case':<26} {'median ×':>12} {dev_header:>15} {'x at max [' + x_unit + ']':>15}")
    print("  " + "-" * 74)
    for row in rows:
        label = row["label"]
        median = row["median_ratio"]
        max_dev = row["max_abs_dev"]
        x_max = row["x_at_max"] * x_scale

        flag = ""
        if np.isfinite(max_dev) and max_dev > tol:
            flag = "  <-- check"

        if np.isfinite(median) and np.isfinite(max_dev):
            print(f"  {label:<26} {median:>12.3f}x {max_dev * 100:>14.1f}% {x_max:>15.2f}{flag}")
        else:
            print(f"  {label:<26} {'--':>12} {'--':>15} {'--':>15}{flag}")


def overlay_ratio_fig(
    wave_ref: np.ndarray,
    L_ref: np.ndarray,
    wave_t: np.ndarray,
    L_t: np.ndarray,
    *,
    x_of_wave=None,
    xlabel: str = r"$\lambda$ [Å]",
    title: str = "",
    ref_label: str = "reference",
    label_t: str = "tengri",
    xlim: tuple[float, float] | None = None,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    band: tuple[float, float] = (0.9, 1.1),
    dyn_range: float = 1e-5,
    figsize: tuple[float, float] = (9.0, 6.5),
) -> tuple[plt.Figure, plt.Axes, plt.Axes, np.ndarray]:
    r"""Overlay two SEDs on ONE shared axis with a ratio panel.

    Both codes are displayed on a single log-log axis, with tengri regridded
    onto the reference wavelength grid. A lower ratio panel shows tengri/reference
    with a shaded tolerance band. The x-axis can be transformed away from
    wavelength (e.g., photon energy for X-ray, frequency for radio) via
    ``x_of_wave``.

    Parameters
    ----------
    wave_ref : array_like, shape (n_wave,)
        Reference wavelength grid [Å].
    L_ref : array_like, shape (n_wave,)
        Reference L_ν [erg/s/Hz] on wave_ref.
    wave_t : array_like, shape (n_wave_t,)
        Tengri wavelength grid [Å].
    L_t : array_like, shape (n_wave_t,)
        Tengri L_ν [erg/s/Hz]; regridded onto wave_ref.
    x_of_wave : callable, optional
        Maps the Å grid to the plotted abscissa (e.g., ``lambda w: 12.398 / w``
        for keV). Identity (wavelength in Å) when ``None``.
    xlabel : str, optional
        x-axis label. Default: r"$\lambda$ [Å]".
    title : str, optional
        Panel title. Default: empty string.
    ref_label : str, optional
        Legend label for the reference curve. Default: "reference".
    label_t : str, optional
        Legend label for the tengri curve. Default: "tengri".
    xlim : tuple of float, optional
        x-limits in the plotted abscissa units. Default: None (auto).
    ratio_ylim : tuple of float, optional
        y-limits on the ratio panel. Default: (0.5, 1.5).
    band : tuple of float, optional
        Shaded tolerance band (lo, hi) on the ratio panel. Default: (0.9, 1.1).
    dyn_range : float, optional
        Top-panel y floor as a fraction of the peak. Default: 1e-5.
    figsize : tuple of float, optional
        Figure size (width, height). Default: (9.0, 6.5).

    Returns
    -------
    fig : plt.Figure
        The figure.
    ax : plt.Axes
        Top (main SED) axis.
    ax_ratio : plt.Axes
        Bottom (ratio) axis.
    ratio : ndarray, shape (n_wave,)
        The tengri/reference ratio array on wave_ref. Where L_ref <= 0,
        the ratio is ``np.nan``.

    Notes
    -----
    Tengri is regridded onto the reference wavelength grid using linear
    interpolation in (wave, L) space via :func:`np.interp`. Where the
    reference is zero or negative, the ratio is set to ``np.nan`` rather
    than ``inf``.
    """
    wave_ref = np.asarray(wave_ref)
    L_ref = np.asarray(L_ref)
    wave_t = np.asarray(wave_t)
    L_t = np.asarray(L_t)

    # Regrid tengri onto reference grid using linear interpolation
    wave_ref, L_ref = _ascending(wave_ref, L_ref)
    wave_t, L_t = _ascending(wave_t, L_t)
    L_t_on_ref = np.interp(wave_ref, wave_t, L_t, left=0.0, right=0.0)

    # Compute ratio: NaN where reference is not positive
    ratio = np.divide(
        L_t_on_ref,
        L_ref,
        where=(L_ref > 0),
        out=np.full_like(L_ref, np.nan),
    )

    # Guard: check that both arms have overlapping positive region
    pos_ref = L_ref > 0
    pos_t = L_t_on_ref > 0
    overlap_positive = pos_ref & pos_t
    if not np.any(overlap_positive):
        raise ValueError(
            f"No overlapping positive region for overlay_ratio_fig. "
            f"Reference wavelength range: [{wave_ref.min():.4e}, {wave_ref.max():.4e}], "
            f"Tengri wavelength range: [{wave_t.min():.4e}, {wave_t.max():.4e}]. "
            f"After regridding, no points have both L_ref > 0 and L_t > 0."
        )

    # Transform x-axis if needed
    x = wave_ref if x_of_wave is None else x_of_wave(wave_ref)

    # Create figure with 3:1 height ratio
    fig, (ax, ax_r) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # The reference is a wide pale band, tengri a narrow saturated line over it.
    # Two same-width curves read as one wherever they agree, which is most of the
    # axis and is where the reader needs to see that they agree.
    _xr, _yr = x[pos_ref], L_ref[pos_ref]
    _mk = _marker_subsample(_xr, n=34, window=xlim)
    ax.plot(
        _xr[_mk],
        _yr[_mk],
        color="C0",
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgewidth=1.4,
        markersize=6.0,
        zorder=3,
        label=ref_label,
    )
    ax.plot(
        x[pos_t],
        L_t_on_ref[pos_t],
        color="C3",
        linewidth=1.7,
        label=label_t,
        zorder=2,
        path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()],
    )

    # Configure top panel
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    _ymx_ref = float(np.nanmax(L_ref[pos_ref])) if pos_ref.any() else 1.0
    _ymx_t = float(np.nanmax(L_t_on_ref[pos_t])) if pos_t.any() else 1.0
    ymx = max(_ymx_ref, _ymx_t)
    _win = np.ones_like(x, dtype=bool) if xlim is None else (x >= xlim[0]) & (x <= xlim[1])
    _p = np.concatenate(
        [
            L_ref[pos_ref & _win],
            L_t_on_ref[pos_t & _win],
        ]
    )
    _p = _p[np.isfinite(_p) & (_p > 0)]
    if _p.size:
        ax.set_ylim(max(float(_p.max()) * dyn_range, float(_p.min()) * 0.5), float(_p.max()) * 2.0)
    else:
        ax.set_ylim(ymx * dyn_range, ymx * 2.0)
    ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ratio panel: shaded band, line at 1.0, ratio curve
    ax_r.axhspan(*band, color="0.85", zorder=0)
    ax_r.axhline(1.0, color="0.5", linewidth=0.8)
    ax_r.plot(x[pos_ref], ratio[pos_ref], "C1-", linewidth=1.0)
    ax_r.set_xscale("log")
    ax_r.set_ylim(*ratio_ylim)
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel(f"tengri / {ref_label}", fontsize=9)
    ax_r.grid(True, alpha=0.3)

    return fig, ax, ax_r, ratio
