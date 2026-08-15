"""Shared helpers for the matched-input validators: broadband photometry
and emission lines.

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

This module supplies both, plus the emission-line half of the same question.

Why here and not in ``_drivers/units.py``
-----------------------------------------
CONTRACT section 3 says shared helpers live in ``_drivers/units.py``,
byte-identical across comparisons. That rule is about helpers the *notebooks*
use. These are validator-only: putting filter I/O and a 26-entry bandpass ladder
into six copies of a module every notebook imports would cost the notebook path
and buy it nothing. One module, imported by the validators alone. CONTRACT
section 3 records the carve-out.

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

import numpy as np

__all__ = [
    "BROAD_FILTERS",
    "IR_BANDS",
    "KEY_LINES",
    "UV_TO_NIR",
    "band_average",
    "convention_sensitivity",
    "filter_rows",
    "line_rows",
    "load_filter",
    "pivot_wavelength",
    "print_filter_table",
    "print_line_table",
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
so past a few um the two are modelling different things and every red band
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
    rows = []
    for stem, label in filters:
        fw, ft = load_filter(stem)
        a = band_average(w_ref, L_t, fw, ft, weight=weight)
        b = band_average(w_ref, L_ref, fw, ft, weight=weight)
        ratio = a / b if (np.isfinite(a) and np.isfinite(b) and b > 0) else float("nan")
        rows.append((label, pivot_wavelength(fw, ft) / 1e4, a, b, ratio))
    return rows


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
# Emission lines
# ---------------------------------------------------------------------------


def safe_half_widths(
    lines: dict[str, tuple[float, ...]],
    *,
    half: float = 12.0,
    pad: float = 0.45,
) -> dict[str, float]:
    """Per-line integration half-width that cannot reach a neighbouring line.

    Parameters
    ----------
    lines : dict
        ``{key: (wavelength, ...)}`` [Angstrom].
    half : float, optional
        Requested half-width [Angstrom]. Default 12.
    pad : float, optional
        Fraction of the nearest-neighbour separation allowed. Default 0.45,
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
    continuum -- integrates the *neighbour's wing*.

    Caught by injecting a Balmer pair of known ratio: both [N II] entries
    reported Halpha's ratio to three decimals, which is what a blend looks
    like when it is not looked for. Narrowing per line is not a full
    deblending (nothing measured off a spectrum can fully separate lines this
    close once they are broadened) but it stops the measurement being
    dominated by a neighbour, and :func:`line_rows` reports which lines were
    narrowed.

    A multi-component entry is treated as **one feature**, centred on the span
    of its components and wide enough to hold all of them. ``KEY_LINES`` lists
    [O II] 3727+3730 and C IV 1548+1551 as single entries precisely because
    their components are summed. Two mistakes are avoided: letting a doublet
    partner count as a contaminant (which collapsed those windows to 1.4 and
    1.2 A and measured a fraction of each) and summing two overlapping
    per-component windows (which double-counts the flux between them).

    Returns the **total** half-width, measured from the feature centre, so it
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
        :func:`safe_half_widths` so a window cannot reach its neighbour.
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
        # One window per feature, centred on the component span: summing
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
        print(f"  b = window narrowed below 12 A to clear a neighbour: {', '.join(blended)}")
    print("  (no pass/fail flag: Cue vs Cloudy is a model difference, not a parity check)")
