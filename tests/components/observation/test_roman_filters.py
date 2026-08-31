# SPDX-License-Identifier: BSD-3-Clause
"""Roman Space Telescope WFI filter pack: registration, curve content, bandpass.

The pivot-wavelength check here used to allow **+/-20%**, which is wider than
the gap between adjacent Roman bands: F158 -> F184 are 16.5% apart and
F184 -> F213 are 15.8%.  A tolerance wider than the spacing between adjacent
categories cannot distinguish them, so the old assertion passed with either of
those two pairs swapped -- the one substitution a filter pack can plausibly
suffer, and the one this file exists to catch.

Measured, the curves are far better than that: the largest pivot error is 2.26%
(F062) and every other band is within 0.28%.  So the tolerance is 5% -- twice
the worst real error, and a third of the smallest band separation -- and a
nearest-nominal check states the anti-swap property directly rather than
hoping a tolerance implies it.  That second assertion also keeps working when
a band is added, which a hand-tuned tolerance does not.

The rest of the file previously asserted structure that any object satisfies:
``hasattr(curve, "wave")`` on a dataclass field is always true, ``fwhm_aa > 0``
admits a FWHM computed in the wrong units, and one ``assert curve is not None``
sat *after* ``curve.name`` had already been dereferenced, so it could never
fire.  Those are replaced with checks on the numbers.
"""

import numpy as np
import pytest

from tengri.observation.filters import (
    FILTER_REGISTRY,
    filter_info,
    load_filter,
    load_filter_set,
)

pytestmark = pytest.mark.bounds


#: ``name -> (SVO identifier, nominal pivot wavelength [Angstrom])``.
#: One table for the whole file; the band list was previously written out four
#: times, so adding a band meant four edits and the set tests could silently
#: drift from the per-filter ones.
ROMAN_BANDS: dict[str, tuple[str, float]] = {
    "roman_f062": ("Roman/WFI.F062", 0.62e4),
    "roman_f087": ("Roman/WFI.F087", 0.87e4),
    "roman_f106": ("Roman/WFI.F106", 1.06e4),
    "roman_f129": ("Roman/WFI.F129", 1.29e4),
    "roman_f158": ("Roman/WFI.F158", 1.58e4),
    "roman_f184": ("Roman/WFI.F184", 1.84e4),
    "roman_f213": ("Roman/WFI.F213", 2.13e4),
}

BAND_NAMES = tuple(ROMAN_BANDS)
NOMINAL_AA = {name: nominal for name, (_, nominal) in ROMAN_BANDS.items()}

#: Twice the largest measured deviation (F062 at 2.26%; every other band is
#: within 0.28%), and a third of the smallest adjacent separation (15.76%), so
#: it cannot be satisfied by a neighboring band's curve.
_PIVOT_RTOL = 0.05

#: Roman WFI are wide-band filters: measured FWHM/pivot spans 0.16 (F213) to
#: 0.43 (F062).  The window is generous around that but excludes a degenerate
#: near-zero width and a FWHM recorded in microns rather than Angstrom, which
#: is a factor of 1e4 and the failure ``fwhm_aa > 0`` was blind to.
_FWHM_FRAC_MIN, _FWHM_FRAC_MAX = 0.10, 0.60


# ── Registration ──────────────────────────────────────────────────
@pytest.mark.parametrize("name", BAND_NAMES)
def test_band_is_registered_with_its_svo_id(name):
    """The SVO identifier is an external data contract, so it is pinned literally.

    A wrong identifier here fetches a real curve for the wrong instrument, which
    every downstream shape and bounds check in this file would accept.
    """
    svo_id, _ = ROMAN_BANDS[name]
    assert name in FILTER_REGISTRY, f"Roman filter {name} not in FILTER_REGISTRY"
    assert FILTER_REGISTRY[name] == svo_id, (
        f"{name} maps to {FILTER_REGISTRY[name]!r}, not the SVO id {svo_id!r}"
    )


# ── Curve content ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", BAND_NAMES)
def test_curve_is_a_usable_bandpass(name):
    """Wave and transmission must be finite, aligned, ordered, and non-degenerate.

    Replaces a ``hasattr(curve, "wave")`` pair, which is true of every instance
    of the dataclass including one carrying empty arrays.
    """
    curve = load_filter(name)
    wave = np.asarray(curve.wave)
    trans = np.asarray(curve.trans)

    assert wave.shape == trans.shape, (
        f"{name}: wave {wave.shape} and trans {trans.shape} are not aligned"
    )
    assert wave.size >= 50, f"{name} has only {wave.size} wavelength points"
    assert np.all(np.isfinite(wave)) and np.all(np.isfinite(trans)), (
        f"{name} carries non-finite values"
    )
    assert np.all(np.diff(wave) > 0), f"{name} wavelength grid is not strictly increasing"
    assert np.trapezoid(trans, wave) > 0.0, (
        f"{name} integrates to zero transmission — the band passes no light"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_transmission_is_in_range(name):
    """Positive, with a real peak, and not renormalized past what SVO emits.

    SVO ships these normalized above unity (measured peaks 2.33-2.99), so the
    upper bound is a sanity ceiling rather than a physical one; the lower bound
    is what makes an all-zero or negative curve fail.
    """
    trans = np.asarray(load_filter(name).trans)

    assert trans.min() >= -0.01, f"{name} transmission dips to {trans.min()}"
    assert trans.max() > 0.3, f"{name} peak transmission {trans.max()} < 0.3"
    assert trans.max() <= 3.0, f"{name} peak transmission {trans.max()} > 3.0"


# ── Bandpass position ─────────────────────────────────────────────
@pytest.mark.parametrize("name", BAND_NAMES)
def test_pivot_is_near_its_nominal_wavelength(name):
    """Within 5% -- see :data:`_PIVOT_RTOL` for why that number and not 20%."""
    nominal = NOMINAL_AA[name]
    pivot = filter_info(name)["lambda_eff_aa"]

    assert abs(pivot - nominal) < _PIVOT_RTOL * nominal, (
        f"{name} pivot {pivot:.0f} A is more than {_PIVOT_RTOL:.0%} from its "
        f"nominal {nominal:.0f} A"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_pivot_identifies_its_own_band(name):
    """Each curve must sit closer to its own nominal than to any other band's.

    This is the assertion a tolerance only approximates, and the one that
    catches the realistic failure: two curves swapped in the pack.  It needs no
    tuning and stays correct as bands are added, whereas a tolerance has to be
    re-checked against the new spacing every time.
    """
    pivot = filter_info(name)["lambda_eff_aa"]
    nearest = min(NOMINAL_AA, key=lambda other: abs(pivot - NOMINAL_AA[other]))

    assert nearest == name, (
        f"{name} has pivot {pivot:.0f} A, which is nearer {nearest}'s nominal "
        f"{NOMINAL_AA[nearest]:.0f} A than its own {NOMINAL_AA[name]:.0f} A — "
        "the two curves are most likely swapped"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_fwhm_is_a_plausible_fraction_of_the_pivot(name):
    """``fwhm_aa > 0`` passed for a width in the wrong unit; bound the ratio."""
    info = filter_info(name)
    assert "lambda_eff_aa" in info and "fwhm_aa" in info, (
        f"{name} metadata is missing a required key: {sorted(info)}"
    )

    frac = info["fwhm_aa"] / info["lambda_eff_aa"]
    assert _FWHM_FRAC_MIN < frac < _FWHM_FRAC_MAX, (
        f"{name} FWHM/pivot = {frac:.3f} outside [{_FWHM_FRAC_MIN}, {_FWHM_FRAC_MAX}] — "
        f"fwhm_aa={info['fwhm_aa']:.1f}, lambda_eff_aa={info['lambda_eff_aa']:.1f}"
    )


# ── The pack as a set ─────────────────────────────────────────────
def test_filter_set_returns_exactly_what_was_asked_for():
    """Same bands, same order.

    The old version asserted ``curve.name in filter_names`` and then
    ``curve is not None`` -- unreachable, since the attribute access above it
    would already have raised.  Order matters because callers zip these curves
    against a photometry table.
    """
    _, _, curves = load_filter_set(list(BAND_NAMES))

    assert tuple(curve.name for curve in curves) == BAND_NAMES, (
        f"load_filter_set returned {[c.name for c in curves]}, expected {list(BAND_NAMES)}"
    )


def test_filter_set_is_ordered_by_wavelength():
    """Pivots increase across the returned pack.

    Asserted on the curves' own grids as well as the metadata: a pack whose
    metadata is sorted while the curves are not would otherwise pass.
    """
    _, _, curves = load_filter_set(list(BAND_NAMES))

    pivots = [filter_info(curve.name)["lambda_eff_aa"] for curve in curves]
    assert pivots == sorted(pivots), f"pack not in wavelength order: {pivots}"

    peaks = [float(np.asarray(curve.wave)[np.argmax(np.asarray(curve.trans))]) for curve in curves]
    assert peaks == sorted(peaks), f"peak transmission wavelengths not ordered: {peaks}"
