# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: the bundled 7DT curves.

These 23 curves ship inside the package rather than being fetched from SVO,
because they are total system response (detector QE and optics folded in),
which is the quantity the photometry was measured through.

What has to hold:

* every band resolves by name, offline, with no network and no cache;
* the nanometer-to-Angstrom conversion is right, checked against the pivot
  wavelengths quoted in the delivery rather than against itself;
* trimming the zero padding did not move any bandpass;
* a user can still shadow a bundled curve, and cannot do so by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract

# Central wavelengths [AA] as quoted in the delivery README. An independent
# check on the conversion: these come from the depositor, not from our files.
DELIVERED_LAM_CEN: dict[str, float] = {
    "7dt_m400": 4013.9,
    "7dt_m425": 4255.7,
    "7dt_m450": 4508.4,
    "7dt_m475": 4753.2,
    "7dt_g": 4812.2,
    "7dt_m500": 5003.6,
    "7dt_m525": 5248.9,
    "7dt_m550": 5501.4,
    "7dt_m575": 5749.2,
    "7dt_m600": 6001.4,
    "7dt_r": 6174.7,
    "7dt_m625": 6248.2,
    "7dt_m650": 6501.2,
    "7dt_m675": 6745.1,
    "7dt_m700": 6999.8,
    "7dt_m725": 7246.8,
    "7dt_m750": 7489.3,
    "7dt_i": 7599.3,
    "7dt_m775": 7752.9,
    "7dt_m800": 7992.9,
    "7dt_m825": 8241.1,
    "7dt_m850": 8485.5,
    "7dt_m875": 8732.9,
}


def test_all_twenty_three_bands_are_registered():
    """The delivery is 3 broad plus 20 medium. All of it, or the set is wrong."""
    from tengri.observation.filters import list_bundled_filters

    names = set(list_bundled_filters().names())

    assert len(names) == 23
    assert names == set(DELIVERED_LAM_CEN)


@pytest.mark.parametrize("name", sorted(DELIVERED_LAM_CEN))
def test_band_loads_and_is_well_formed(name):
    """Each curve resolves through the ordinary loader, in Angstrom, increasing."""
    from tengri.observation.filters import load_filter

    fc = load_filter(name)
    wave = np.asarray(fc.wave)
    trans = np.asarray(fc.trans)

    assert fc.name == name
    assert wave.size > 100
    assert np.all(np.diff(wave) > 0), "wavelengths must be strictly increasing"
    assert np.all(trans >= 0.0)
    assert trans.max() > 0.0
    # Optical bands in Angstrom. In nanometers these would land near 300-1000.
    assert 3000.0 < wave.min() < wave.max() < 1.0e4 + 1.0


@pytest.mark.parametrize(("name", "expected"), sorted(DELIVERED_LAM_CEN.items()))
def test_pivot_wavelength_matches_the_delivery(name, expected):
    """The conversion is checked against the depositor's own numbers.

    A unit error of the kind this whole change is about would show up here as a
    factor of 10, not as a rounding difference.
    """
    from tengri.observation.filters import load_filter

    fc = load_filter(name)
    wave, trans = np.asarray(fc.wave), np.asarray(fc.trans)
    lam_cen = np.trapezoid(trans * wave, wave) / np.trapezoid(trans, wave)

    assert lam_cen == pytest.approx(expected, abs=0.5)


def test_medium_bands_are_about_25nm_wide():
    """The 20 medium bands are a 25 nm grid. A resampling bug would show here."""
    from tengri.observation.filters import load_filter

    for name in (n for n in DELIVERED_LAM_CEN if n.startswith("7dt_m")):
        fc = load_filter(name)
        wave, trans = np.asarray(fc.wave), np.asarray(fc.trans)
        above = wave[trans >= 0.5 * trans.max()]
        fwhm_aa = above.max() - above.min()

        assert 200.0 < fwhm_aa < 300.0, f"{name}: FWHM {fwhm_aa:.0f} AA"


def test_trimming_zero_padding_did_not_move_any_bandpass():
    """Trimmed rows were exactly zero, so both bandpass integrals are unchanged.

    Re-padding with explicit zeros must reproduce the same AB zeropoint. This is
    the property the trim claimed; it is cheap to check and expensive to assume.
    """
    from tengri.observation.filters import load_filter

    for name in sorted(DELIVERED_LAM_CEN):
        fc = load_filter(name)
        wave, trans = np.asarray(fc.wave), np.asarray(fc.trans)

        pad_lo = np.arange(3000.0, wave.min(), 1.0)
        pad_hi = np.arange(wave.max() + 1.0, 10001.0, 1.0)
        wave_full = np.concatenate([pad_lo, wave, pad_hi])
        trans_full = np.concatenate([np.zeros_like(pad_lo), trans, np.zeros_like(pad_hi)])

        weight = np.trapezoid(trans / wave, wave)
        weight_full = np.trapezoid(trans_full / wave_full, wave_full)
        dmag = abs(2.5 * np.log10(weight / weight_full))

        assert dmag < 1e-6, f"{name}: trim moved the AB zeropoint by {dmag:.2e} mag"


def test_bundled_curves_load_without_network(monkeypatch):
    """No SVO call, no cache lookup. The numbers travel with the package."""
    import tengri.observation.filters as filters_mod

    def explode(*args, **kwargs):
        raise AssertionError("bundled curves must not reach for the network")

    monkeypatch.setattr(filters_mod, "download_filter", explode)

    fc = filters_mod.load_filter("7dt_m400")
    assert np.asarray(fc.wave).size > 100


def test_user_registration_shadows_a_bundled_curve():
    """Precedence: the two user routes still win, as they do over SVO names."""
    from tengri.observation.filters import load_filter
    from tengri.observation.filters.custom import register_filter, unregister_filter

    try:
        register_filter("7dt_g", np.linspace(5000.0, 6000.0, 50), np.ones(50), overwrite=True)
        assert float(load_filter("7dt_g").wave.min()) == 5000.0
    finally:
        unregister_filter("7dt_g")

    assert float(load_filter("7dt_g").wave.min()) != 5000.0


def test_shadowing_a_bundled_curve_needs_overwrite():
    """Accidental collision raises, and the message says where the name came from."""
    from tengri.observation.filters.custom import register_filter

    with pytest.raises(KeyError, match="bundled"):
        register_filter("7dt_m400", np.linspace(4000.0, 4100.0, 10), np.ones(10))


def test_bundled_curves_appear_in_the_filter_menu():
    """Loadable but unlisted is undiscoverable.

    ``tengri.list_filters()`` scans the SVO cache directory, which these curves
    are not in. The unknown-filter error points users at that menu, so a band
    missing from it cannot be found by anyone who does not already know its
    name.
    """
    import tengri

    rows = tengri.list_filters(survey="7dt")
    listed = {row["name"] for row in rows}

    assert listed == set(DELIVERED_LAM_CEN)


def test_provenance_file_ships_with_the_curves():
    """Instrument data without provenance is not reproducible, so pin its presence."""
    from importlib import resources

    text = resources.files("tengri.data.filters_7dt").joinpath("PROVENANCE.md").read_text()

    assert "sha256" in text
    assert "nanometer" in text.lower()
    # The curves include QE; a reader must not mistake them for filter glass.
    assert "QE" in text
