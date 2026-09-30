# SPDX-License-Identifier: BSD-3-Clause
"""SkyMapper Southern Survey filter pack: registration, curve content, bandpass.

SkyMapper observes in **six** bands, ``uvgriz``, not the ``ugriz`` five that
``sdss_*``/``lsst_*``/``ps1_*`` establish as the shape of an optical survey
pack.  The extra band is ``v``, a violet band at 3838 A sitting between ``u``
and ``g``; the pair is what the survey is built around, and it drives most of
what this file checks.

Two hazards are specific to this pack.

**``v`` is violet, not Johnson V.**  The registry already holds ``xmm_v`` ->
``XMM/OM.V`` and ``johnson_v``, both Johnson V near 5500 A.  ``skymapper_v`` is
3838 A, a factor 1.4 away, so the same band letter means two different things
depending on the prefix.  The alias is nonetheless spelled ``v``, because every
alias in the registry takes its band letter from the SVO identifier and an
exception here would be the arbitrary choice.  What makes that safe is a pinned
wavelength rather than a naming rule, so
:func:`test_v_is_the_violet_band_not_johnson_v` states it directly.

**``u`` and ``v`` are adjacent and overlapping.**  Their nominal pivots are
9.43% apart, by far the tightest pair here (the next tightest is 17.6%), and
their supports genuinely overlap over 3550-3850 A.  A tolerance wider than
half that separation could not tell the two curves apart, which is the one
substitution this pack can plausibly suffer.  Hence ``_PIVOT_RTOL`` below, plus
a nearest-nominal check that states the anti-swap property outright instead of
hoping a tolerance implies it.

A note for anyone extending this file from its Roman sibling: that file asserts
``wave.size >= 50``, which is **wrong here**.  SVO tabulates SkyMapper on a
coarse 50 A grid, so ``skymapper_v`` carries 16 points and ``skymapper_u`` 19.
The threshold is per-facility, not a house rule.
"""

import numpy as np
import pytest

from tengri.observation.filters import (
    FILTER_REGISTRY,
    _infer_facility,
    filter_info,
    find_cached_filter,
    load_filter,
    load_filter_set,
)

pytestmark = pytest.mark.bounds


#: ``name -> (SVO identifier, nominal pivot wavelength [Angstrom])``, in
#: wavelength order.  One table for the whole file, so adding a band is one
#: edit and the set-level tests cannot drift from the per-band ones.
#:
#: The nominals are the round published SkyMapper effective wavelengths (Wolf
#: et al. 2018, PASA 35, e010).  They are *independent* of the shipped curves,
#: which is what makes them worth asserting against: all six agree with the
#: curves tengri actually loads to within 0.21%, and six bands matching round
#: literature numbers that closely is not something a wrong or mislabeled
#: curve file would reproduce.
SKYMAPPER_BANDS: dict[str, tuple[str, float]] = {
    "skymapper_u": ("SkyMapper/SkyMapper.u", 3500.0),
    "skymapper_v": ("SkyMapper/SkyMapper.v", 3830.0),
    "skymapper_g": ("SkyMapper/SkyMapper.g", 5100.0),
    "skymapper_r": ("SkyMapper/SkyMapper.r", 6170.0),
    "skymapper_i": ("SkyMapper/SkyMapper.i", 7790.0),
    "skymapper_z": ("SkyMapper/SkyMapper.z", 9160.0),
}

BAND_NAMES = tuple(SKYMAPPER_BANDS)
NOMINAL_AA = {name: nominal for name, (_, nominal) in SKYMAPPER_BANDS.items()}

#: 1%: roughly five times the largest measured deviation from nominal (``r`` at
#: 0.206%; every other band is within 0.2%), so an SVO retabulation has room to
#: move without a spurious red, and roughly a ninth of the smallest adjacent
#: separation (u -> v at 9.43%), so it provably cannot be satisfied by the
#: neighboring band's curve.
_PIVOT_RTOL = 0.01

#: Measured FWHM/pivot runs 0.065 (``v``, a deliberately narrow violet band) to
#: 0.304 (``g``).  The window is generous around that but still excludes a
#: degenerate near-zero width and a FWHM recorded in microns rather than
#: Angstrom -- a factor of 1e4, and the failure a bare ``fwhm_aa > 0`` is blind
#: to.
_FWHM_FRAC_MIN, _FWHM_FRAC_MAX = 0.03, 0.45

#: ``skymapper_v`` carries 16 points on SVO's 50 A grid, the fewest in the
#: pack.  Set below that, so the bound catches a truncated or empty curve
#: without encoding the current tabulation as a requirement.
_MIN_POINTS = 12

#: Johnson V, for the confusability check.  Not a magic number: it is the
#: band ``skymapper_v`` must not be mistaken for.
_JOHNSON_V_AA = 5500.0


# ── Registration ──────────────────────────────────────────────────
@pytest.mark.parametrize("name", BAND_NAMES)
def test_band_is_registered_with_its_svo_id(name):
    """The SVO identifier is an external data contract, so it is pinned literally.

    A wrong identifier fetches a real curve for the wrong band, which every
    shape and finiteness check in this file would happily accept.
    """
    svo_id, _ = SKYMAPPER_BANDS[name]
    assert name in FILTER_REGISTRY, f"SkyMapper filter {name} not in FILTER_REGISTRY"
    assert FILTER_REGISTRY[name] == svo_id, (
        f"{name} maps to {FILTER_REGISTRY[name]!r}, not the SVO id {svo_id!r}"
    )


def test_the_six_aliases_map_to_six_distinct_curves():
    """No two aliases share an SVO identifier.

    The pack entered the registry as six adjacent near-identical JSON lines,
    where a copy-paste leaving two aliases on one identifier is the realistic
    slip.  It would leave ``load_filter`` returning the same curve under two
    names, which reads as success everywhere except here.
    """
    svo_ids = [FILTER_REGISTRY[name] for name in BAND_NAMES]

    assert len(set(svo_ids)) == len(BAND_NAMES), (
        f"the six SkyMapper aliases resolve to {len(set(svo_ids))} distinct SVO "
        f"ids, not 6: {dict(zip(BAND_NAMES, svo_ids))}"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_alias_band_letter_matches_the_svo_band_letter(name):
    """``skymapper_<x>`` must resolve to ``SkyMapper/SkyMapper.<x>``.

    States the naming rule the pack follows, so a future band cannot be added
    under a letter that disagrees with the curve it loads.  This is the check
    that would have caught ``u`` and ``v`` transposed in the registry even if
    their curves were somehow both plausible.
    """
    expected_letter = name.removeprefix("skymapper_")
    svo_letter = FILTER_REGISTRY[name].rsplit(".", 1)[-1]

    assert svo_letter == expected_letter, (
        f"{name} maps to SVO band {svo_letter!r}, not {expected_letter!r}"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_facility_is_inferred_as_skymapper(name):
    """The prefix must land on SkyMapper and not be swallowed by another facility.

    ``_infer_facility`` walks ``_FACILITY_FROM_PREFIX`` and returns on the
    first ``startswith`` hit, so a new prefix that is a prefix of an existing
    one silently inherits the wrong facility and the whole pack disappears
    from its own group in the discovery table.
    """
    assert _infer_facility(name) == "SkyMapper", (
        f"{name} is grouped under {_infer_facility(name)!r}, not 'SkyMapper'"
    )


# ── Offline availability ──────────────────────────────────────────
@pytest.mark.parametrize("name", BAND_NAMES)
def test_curve_is_cached_on_disk_and_needs_no_network(name):
    """Every band must ship a tracked curve, not fall back to an SVO fetch.

    A missing curve file is indistinguishable from a cold cache: the loader
    fetches, succeeds, and reports nothing, so the pack looks fine on a
    developer machine and breaks in the offline gallery build (#1798).  Asking
    :func:`find_cached_filter` where the curve is turns that silent fallback
    into a red.
    """
    svo_id, _ = SKYMAPPER_BANDS[name]
    filename = svo_id.replace("/", "_").replace(".", "_") + ".dat"

    assert find_cached_filter(filename) is not None, (
        f"{name} has no tracked curve; expected data/filters/{filename}. "
        "Without it this band only loads with network access."
    )


# ── Curve content ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", BAND_NAMES)
def test_curve_is_a_usable_bandpass(name):
    """Wave and transmission must be finite, aligned, ordered, and pass light."""
    curve = load_filter(name)
    wave = np.asarray(curve.wave)
    trans = np.asarray(curve.trans)

    assert wave.shape == trans.shape, (
        f"{name}: wave {wave.shape} and trans {trans.shape} are not aligned"
    )
    assert wave.size >= _MIN_POINTS, (
        f"{name} has only {wave.size} wavelength points (floor {_MIN_POINTS})"
    )
    assert np.all(np.isfinite(wave)) and np.all(np.isfinite(trans)), (
        f"{name} carries non-finite values"
    )
    assert np.all(np.diff(wave) > 0), (
        f"{name} wavelength grid is not strictly increasing; every downstream "
        "np.interp and searchsorted assumes it is"
    )
    assert np.trapezoid(trans, wave) > 0.0, (
        f"{name} integrates to zero transmission — the band passes no light"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_transmission_is_a_unit_normalized_throughput(name):
    """Positive, with a real peak, and normalized to unity as SVO ships it.

    Facility-specific on purpose.  SVO serves the Roman curves normalized
    *above* unity (peaks 2.33-2.99), so its sibling test can only assert a
    loose ceiling of 3.0.  SkyMapper arrives peak-normalized to 1.0 (measured
    0.992-1.000), which admits the tighter bound -- and that bound is what
    would catch a curve silently rescaled on the way in.
    """
    trans = np.asarray(load_filter(name).trans)

    assert trans.min() >= -0.01, f"{name} transmission dips to {trans.min()}"
    assert trans.max() > 0.5, f"{name} peak transmission {trans.max()} <= 0.5"
    assert trans.max() <= 1.01, (
        f"{name} peak transmission {trans.max()} > 1.01; SkyMapper curves are "
        "peak-normalized to unity, so this curve has been rescaled"
    )


# ── Bandpass position ─────────────────────────────────────────────
@pytest.mark.parametrize("name", BAND_NAMES)
def test_pivot_is_near_its_nominal_wavelength(name):
    """Within 1% — see :data:`_PIVOT_RTOL` for why that number."""
    nominal = NOMINAL_AA[name]
    pivot = filter_info(name)["lambda_eff_aa"]

    assert abs(pivot - nominal) < _PIVOT_RTOL * nominal, (
        f"{name} pivot {pivot:.0f} A is more than {_PIVOT_RTOL:.0%} from its "
        f"nominal {nominal:.0f} A"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_pivot_identifies_its_own_band(name):
    """Each curve must sit closer to its own nominal than to any other band's.

    The assertion a tolerance only approximates, and the one that catches two
    curves swapped in the pack.  It needs no tuning and stays correct as bands
    are added, whereas a tolerance has to be re-checked against the new
    spacing every time.  For ``u`` and ``v``, 9.43% apart, it is the check
    doing the real work.
    """
    pivot = filter_info(name)["lambda_eff_aa"]
    nearest = min(NOMINAL_AA, key=lambda other: abs(pivot - NOMINAL_AA[other]))

    assert nearest == name, (
        f"{name} has pivot {pivot:.0f} A, which is nearer {nearest}'s nominal "
        f"{NOMINAL_AA[nearest]:.0f} A than its own {NOMINAL_AA[name]:.0f} A — "
        "the two curves are most likely swapped"
    )


def test_v_is_the_violet_band_not_johnson_v():
    """``skymapper_v`` is 3838 A, not Johnson V near 5500 A.

    The registry spells three different bands ``*_v``: ``skymapper_v``
    (violet, 3838 A), ``johnson_v`` and ``xmm_v`` (Johnson V, ~5500 A).  The
    shared letter is the trap, and a reader importing "the v band" from the
    wrong prefix gets a curve 1.4x away in wavelength with no error.  Pinning
    the distance to Johnson V says which band this is in the only terms that
    cannot be misread.
    """
    pivot = filter_info("skymapper_v")["lambda_eff_aa"]

    assert pivot < 4500.0, (
        f"skymapper_v pivot {pivot:.0f} A is not a violet band; anything above "
        "4500 A means it has been confused with Johnson V"
    )
    assert abs(pivot - _JOHNSON_V_AA) > 1000.0, (
        f"skymapper_v pivot {pivot:.0f} A is within 1000 A of Johnson V "
        f"({_JOHNSON_V_AA:.0f} A) — it should be nowhere near it"
    )


def test_u_and_v_are_distinct_overlapping_bands():
    """The defining SkyMapper pair: separate curves, ordered, and overlapping.

    ``u`` and ``v`` are the reason this pack has six bands rather than five.
    They must be *different* curves (an identical pair would pass every
    per-band check above except the nearest-nominal one), ordered ``u`` then
    ``v``, and genuinely overlapping — the overlap is a real property of the
    filter set, so a pack of two disjoint top-hats would be the wrong data.
    """
    u_curve = load_filter("skymapper_u")
    v_curve = load_filter("skymapper_v")

    u_wave = np.asarray(u_curve.wave)
    v_wave = np.asarray(v_curve.wave)
    u_trans = np.asarray(u_curve.trans)
    v_trans = np.asarray(v_curve.trans)

    u_pivot = filter_info("skymapper_u")["lambda_eff_aa"]
    v_pivot = filter_info("skymapper_v")["lambda_eff_aa"]

    assert u_pivot < v_pivot, f"u ({u_pivot:.0f} A) must sit blueward of v ({v_pivot:.0f} A)"
    assert not (u_wave.shape == v_wave.shape and np.allclose(u_trans, v_trans)), (
        "skymapper_u and skymapper_v are the same curve under two names"
    )

    u_support = u_wave[u_trans > 0]
    v_support = v_wave[v_trans > 0]
    overlap = min(u_support.max(), v_support.max()) - max(u_support.min(), v_support.min())

    assert overlap > 0.0, (
        f"u ({u_support.min():.0f}-{u_support.max():.0f} A) and v "
        f"({v_support.min():.0f}-{v_support.max():.0f} A) do not overlap; the "
        "SkyMapper u and v bandpasses do"
    )


@pytest.mark.parametrize("name", BAND_NAMES)
def test_fwhm_is_a_plausible_fraction_of_the_pivot(name):
    """``fwhm_aa > 0`` passes for a width in the wrong unit; bound the ratio."""
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
    """Same bands, same order.  Callers zip these curves against a flux table."""
    _, _, curves = load_filter_set(list(BAND_NAMES))

    assert tuple(curve.name for curve in curves) == BAND_NAMES, (
        f"load_filter_set returned {[c.name for c in curves]}, expected {list(BAND_NAMES)}"
    )


def test_filter_set_is_ordered_by_wavelength():
    """Pivots increase across the returned pack.

    Asserted on the curves' own grids as well as on the metadata: a pack whose
    metadata is sorted while the curves are not would otherwise pass.
    """
    _, _, curves = load_filter_set(list(BAND_NAMES))

    pivots = [filter_info(curve.name)["lambda_eff_aa"] for curve in curves]
    assert pivots == sorted(pivots), f"pack not in wavelength order: {pivots}"

    peaks = [float(np.asarray(curve.wave)[np.argmax(np.asarray(curve.trans))]) for curve in curves]
    assert peaks == sorted(peaks), f"peak transmission wavelengths not ordered: {peaks}"


def test_pack_spans_near_uv_to_near_ir():
    """The six bands together cover 3050-10700 A, the survey's stated range.

    A set-level bound, so a pack that lost its bluest or reddest band and
    still passed every per-band check above is caught here.
    """
    _, _, curves = load_filter_set(list(BAND_NAMES))

    supports = [np.asarray(c.wave)[np.asarray(c.trans) > 0] for c in curves]
    blue = min(float(s.min()) for s in supports)
    red = max(float(s.max()) for s in supports)

    assert blue < 3200.0, f"bluest SkyMapper coverage starts at {blue:.0f} A, expected < 3200 A"
    assert red > 10000.0, f"reddest SkyMapper coverage ends at {red:.0f} A, expected > 10000 A"


def test_photometry_accepts_the_whole_pack():
    """The pack must reach a fit, which is the point of registering it.

    ``Photometry.from_names`` is the documented route from a filter name into
    a model, so a band that loads but cannot be assembled into an observation
    is not actually usable.
    """
    from tengri import Photometry

    phot = Photometry.from_names(list(BAND_NAMES))

    assert phot.names == BAND_NAMES, f"Photometry kept {phot.names}, expected {BAND_NAMES}"
    assert phot.n_filters == len(BAND_NAMES)
    assert all(len(w) >= _MIN_POINTS for w in phot.filter_waves), (
        f"a curve arrived truncated: {[len(w) for w in phot.filter_waves]}"
    )
