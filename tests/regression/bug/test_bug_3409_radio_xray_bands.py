# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #3409: radio bands below 3 GHz, frequency and energy wave units.

``register_filter`` refused a 1.4 GHz top-hat because the Angstrom ceiling stopped
at 1e9 A, while the forward model extends to 3e11 A. ``wave_unit`` rejected Hz, GHz
and keV. A keV-valued band passed in the default unit registered silently as 2-10 A.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri.observation.filters import custom, load_filter
from tengri.observation.filters.custom import register_filter, unregister_filter
from tengri.utils.physics_constants import C_AA, HC_KEV_ANGSTROM
from tengri.utils.wave_unit import WAVE_CEILING_AA, to_angstrom
from tengri.utils.wavelength import RADIO_WAVE_MAX


@pytest.fixture
def name():
    """Yield a unique user-filter name and unregister it afterwards."""
    probe = "bug3409_probe"
    yield probe
    unregister_filter(probe)


def _registered(name: str) -> tuple[np.ndarray, np.ndarray]:
    fc = custom._USER_FILTER_REGISTRY[name]
    return np.asarray(fc.wave), np.asarray(fc.trans)


# ── The ceiling is the forward model's radio extent ──────────────────


def test_ceiling_is_the_forward_model_radio_extent():
    assert WAVE_CEILING_AA == RADIO_WAVE_MAX == pytest.approx(3.0e11)


def test_1p4ghz_tophat_registers_in_ghz(name):
    nu_ghz = np.linspace(1.0, 2.0, 50)
    register_filter(name, nu_ghz, np.ones(50), wave_unit="GHz")
    wave, _ = _registered(name)
    assert np.all(np.diff(wave) > 0)
    assert wave[0] == pytest.approx(C_AA / 2.0e9, rel=1e-12)
    assert wave[-1] == pytest.approx(C_AA / 1.0e9, rel=1e-12)


def test_range_above_old_ceiling_but_below_radio_extent_registers(name):
    register_filter(name, np.linspace(2.0e9, 3.0e9, 20), np.ones(20))
    assert name in custom._USER_FILTER_REGISTRY


def test_range_above_forward_model_extent_is_still_refused(name):
    with pytest.raises(ValueError, match="outside the plausible Angstrom range"):
        register_filter(name, np.linspace(4.0e11, 6.0e11, 20), np.ones(20))
    assert name not in custom._USER_FILTER_REGISTRY


# ── Frequency and energy units ───────────────────────────────────────


@pytest.mark.parametrize(
    ("unit", "hz_per_unit"),
    [("Hz", 1.0), ("kHz", 1.0e3), ("MHz", 1.0e6), ("GHz", 1.0e9)],
)
def test_frequency_units_convert_to_angstrom(unit, hz_per_unit):
    nu = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(to_angstrom(nu, unit), C_AA / (nu * hz_per_unit), rtol=1e-14)


def test_energy_units_convert_with_package_hc_constant():
    e_kev = np.array([2.0, 10.0])
    np.testing.assert_allclose(to_angstrom(e_kev, "keV"), HC_KEV_ANGSTROM / e_kev, rtol=1e-14)
    np.testing.assert_allclose(
        to_angstrom(e_kev * 1.0e3, "eV"), HC_KEV_ANGSTROM / e_kev, rtol=1e-14
    )


def test_new_unit_names_are_case_insensitive():
    np.testing.assert_allclose(
        to_angstrom(np.array([1.0]), "ghz"), to_angstrom(np.array([1.0]), "GHz")
    )
    np.testing.assert_allclose(
        to_angstrom(np.array([2.0]), "KEV"), to_angstrom(np.array([2.0]), "keV")
    )


def test_unknown_unit_message_lists_new_units():
    with pytest.raises(ValueError, match="GHz"):
        to_angstrom(np.array([1.0]), "parsec")


def test_registering_in_ghz_reverses_transmission_with_wavelength(name):
    # Transmission rises with frequency; after conversion the wavelength axis is
    # increasing, so the transmission must be reversed along with it.
    nu_ghz = np.linspace(1.0, 2.0, 50)
    trans = np.linspace(0.0, 1.0, 50)
    register_filter(name, nu_ghz, trans, wave_unit="GHz")
    wave, stored = _registered(name)
    assert np.all(np.diff(wave) > 0)
    # Lowest frequency is the longest wavelength, so it carries trans[0] = 0.
    assert stored[-1] == pytest.approx(0.0, abs=1e-12)
    assert stored[0] == pytest.approx(1.0, abs=1e-12)


def test_registering_in_kev_gives_xray_wavelengths(name):
    e_kev = np.linspace(2.0, 10.0, 50)
    register_filter(name, e_kev, np.ones(50), wave_unit="keV")
    wave, _ = _registered(name)
    assert wave[0] == pytest.approx(HC_KEV_ANGSTROM / 10.0, rel=1e-12)
    assert wave[-1] == pytest.approx(HC_KEV_ANGSTROM / 2.0, rel=1e-12)


# ── Unit-less keV-looking band ───────────────────────────────────────


def test_unitless_band_in_angstrom_xray_range_warns_about_keV(name):
    with pytest.warns(UserWarning, match="wave_unit='keV'"):
        register_filter(name, np.linspace(2.0, 10.0, 50), np.ones(50))
    # Still legal: it is a valid X-ray wavelength range.
    assert name in custom._USER_FILTER_REGISTRY


def test_stated_keV_band_does_not_warn(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        register_filter(name, np.linspace(2.0, 10.0, 50), np.ones(50), wave_unit="keV")


def test_stated_angstrom_band_does_not_warn_about_keV(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register_filter(name, np.linspace(2.0, 10.0, 50), np.ones(50), wave_unit="AA")
    assert not [w for w in caught if "keV" in str(w.message)]


# ── Shipped radio bands (data/filters, same route as PLANCK_HFI) ─────

_SHIPPED_RADIO_BANDS = [
    ("VLA_L_1p4GHz", 1.0, 2.0),
    ("VLA_S_3GHz", 2.0, 4.0),
    ("MeerKAT_L_1p28GHz", 0.856, 1.712),
    ("LOFAR_HBA_144MHz", 0.120, 0.168),
]


@pytest.mark.parametrize(("name", "lo_ghz", "hi_ghz"), _SHIPPED_RADIO_BANDS)
def test_shipped_radio_band_has_its_edges(name, lo_ghz, hi_ghz):
    fc = load_filter(name)
    wave = np.asarray(fc.wave)
    trans = np.asarray(fc.trans)
    assert np.all(np.diff(wave) > 0)
    inside = wave[trans > 0.5]
    assert inside.min() == pytest.approx(C_AA / (hi_ghz * 1.0e9), rel=1e-3)
    assert inside.max() == pytest.approx(C_AA / (lo_ghz * 1.0e9), rel=1e-3)
    assert float(np.max(trans)) == pytest.approx(1.0)


@pytest.mark.parametrize(("name", "lo_ghz", "hi_ghz"), _SHIPPED_RADIO_BANDS)
def test_shipped_radio_band_resolves_by_lowercase_alias(name, lo_ghz, hi_ghz):
    alias = name.lower()
    assert load_filter(alias).name == alias
