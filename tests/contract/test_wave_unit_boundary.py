# SPDX-License-Identifier: BSD-3-Clause
"""Contract: one ``wave_unit`` spelling at every raw-array boundary, and refusals.

Every public entry that takes a raw wavelength array converts it to Angstrom once,
at ingestion. A nanometer or micron input must therefore produce arrays bit-identical
to the same grid stated in Angstrom. A range that is nonsense for Angstrom is refused.
Timescale parameters are bounded by suffix when a Parameters object is built.
"""

from __future__ import annotations

import collections
import contextlib
import warnings

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Parameters
from tengri.config.exceptions import ParameterError
from tengri.observation.filters import custom
from tengri.observation.filters.custom import (
    load_filter_from_dsps_file,
    load_filter_from_dsps_transmission_curve,
    register_filter,
    register_filter_from_file,
    unregister_filter,
)
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.priors import Fixed
from tengri.utils.wave_unit import (
    WAVE_CEILING_AA,
    WAVE_FLOOR_AA,
    check_timescale,
    to_angstrom,
)

pytestmark = pytest.mark.contract

#: A nm grid spanning the optical, with a plain triangular transmission.
WAVE_NM = np.linspace(300.0, 700.0, 81)
TRANS = np.clip(1.0 - np.abs(np.linspace(-1.0, 1.0, 81)), 0.0, None)

DspsCurve = collections.namedtuple("DspsCurve", ["wave", "transmission"])


@pytest.fixture
def filter_name():
    """Yield a unique user-filter name and unregister it afterwards."""
    name = "wu_contract_probe"
    yield name
    with contextlib.suppress(KeyError):
        unregister_filter(name)


# ── to_angstrom: the one conversion ───────────────────────────────────


@pytest.mark.parametrize("unit", ["nm", "NM"])
def test_nm_converts_bit_identical_to_angstrom_times_ten(unit):
    chex.assert_trees_all_equal(to_angstrom(WAVE_NM, unit), WAVE_NM * 10.0)


@pytest.mark.parametrize("unit", ["um", "micron", "Micron"])
def test_um_converts_bit_identical_to_angstrom_times_1e4(unit):
    wave_um = WAVE_NM / 1000.0
    chex.assert_trees_all_equal(to_angstrom(wave_um, unit), wave_um * 1.0e4)


@pytest.mark.parametrize("unit", ["AA", "aa", "angstrom", "Angstrom"])
def test_angstrom_names_are_identity(unit):
    wave = np.linspace(4000.0, 8000.0, 5)
    assert to_angstrom(wave, unit) is wave


def test_unknown_unit_raises_listing_accepted_names():
    with pytest.raises(ValueError) as info:
        to_angstrom(WAVE_NM, "pc")
    message = str(info.value)
    assert "'pc'" in message
    for accepted in ("'AA'", "'angstrom'", "'nm'", "'um'", "'micron'"):
        assert accepted in message


# ── filter intake: register_filter / register_filter_from_file ────────


def test_register_filter_nm_matches_angstrom(filter_name):
    register_filter(filter_name, WAVE_NM, TRANS, wave_unit="nm")
    nm_curve = custom._USER_FILTER_REGISTRY[filter_name]
    unregister_filter(filter_name)

    register_filter(filter_name, WAVE_NM * 10.0, TRANS)
    aa_curve = custom._USER_FILTER_REGISTRY[filter_name]

    chex.assert_trees_all_equal(nm_curve.wave, aa_curve.wave)
    chex.assert_trees_all_equal(nm_curve.trans, aa_curve.trans)


def test_register_filter_um_matches_angstrom(filter_name):
    wave_um = WAVE_NM / 1000.0
    register_filter(filter_name, wave_um, TRANS, wave_unit="um")
    um_curve = custom._USER_FILTER_REGISTRY[filter_name]
    unregister_filter(filter_name)

    register_filter(filter_name, wave_um * 1.0e4, TRANS)
    aa_curve = custom._USER_FILTER_REGISTRY[filter_name]

    chex.assert_trees_all_equal(um_curve.wave, aa_curve.wave)


def test_register_filter_from_file_nm_matches_angstrom(tmp_path, filter_name):
    path = tmp_path / "band.csv"
    rows = np.column_stack([WAVE_NM, TRANS])
    np.savetxt(path, rows, delimiter=",", header="lam,trans", comments="")

    register_filter_from_file(filter_name, path, wave_unit="nm")
    nm_curve = custom._USER_FILTER_REGISTRY[filter_name]
    unregister_filter(filter_name)

    register_filter_from_file(filter_name, path, wave_unit="NM")
    chex.assert_trees_all_equal(nm_curve.wave, custom._USER_FILTER_REGISTRY[filter_name].wave)
    chex.assert_trees_all_equal(nm_curve.wave, jnp.asarray(WAVE_NM * 10.0))


def test_dsps_transmission_curve_nm_matches_angstrom():
    nm = load_filter_from_dsps_transmission_curve(
        DspsCurve(wave=WAVE_NM, transmission=TRANS), name="wu_dsps", wave_unit="nm"
    )
    aa = load_filter_from_dsps_transmission_curve(
        DspsCurve(wave=WAVE_NM * 10.0, transmission=TRANS), name="wu_dsps"
    )
    chex.assert_trees_all_equal(nm.wave, aa.wave)


def test_dsps_file_nm_matches_angstrom(tmp_path, monkeypatch):
    import dsps.data_loaders as loaders

    path = tmp_path / "band.h5"
    path.write_bytes(b"placeholder")

    def fake_loader(fn):
        assert fn == str(path)
        return DspsCurve(wave=WAVE_NM, transmission=TRANS)

    monkeypatch.setattr(loaders, "load_transmission_curve", fake_loader)
    nm = load_filter_from_dsps_file(path, name="wu_dsps_file", wave_unit="nm")
    monkeypatch.setattr(
        loaders,
        "load_transmission_curve",
        lambda fn: DspsCurve(wave=WAVE_NM * 10.0, transmission=TRANS),
    )
    aa = load_filter_from_dsps_file(path, name="wu_dsps_file")
    chex.assert_trees_all_equal(nm.wave, aa.wave)


# ── Spectroscopy and its factories ────────────────────────────────────


def test_spectroscopy_nm_stores_angstrom_bit_identical():
    nm = Spectroscopy(wave_obs=jnp.asarray(WAVE_NM * 1.0e3), wave_unit="nm")
    aa = Spectroscopy(wave_obs=jnp.asarray(WAVE_NM * 1.0e4))
    chex.assert_trees_all_equal(nm.wave_obs, aa.wave_obs)


def test_spectroscopy_um_stores_angstrom_bit_identical():
    wave_um = np.linspace(0.8, 2.5, 60)
    um = Spectroscopy(wave_obs=jnp.asarray(wave_um), wave_unit="um")
    aa = Spectroscopy(wave_obs=jnp.asarray(wave_um * 1.0e4))
    chex.assert_trees_all_equal(um.wave_obs, aa.wave_obs)


def test_spectroscopy_default_unit_is_angstrom():
    grid = np.linspace(4000.0, 6000.0, 50)
    implicit = Spectroscopy(wave_obs=jnp.asarray(grid))
    explicit = Spectroscopy(wave_obs=jnp.asarray(grid), wave_unit="AA")
    chex.assert_trees_all_equal(implicit.wave_obs, explicit.wave_obs)


def test_spectroscopy_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown wave_unit"):
        Spectroscopy(wave_obs=jnp.asarray(WAVE_NM), wave_unit="parsec")


def test_nirspec_prism_nm_matches_angstrom():
    nm = Spectroscopy.nirspec_prism(jnp.asarray(WAVE_NM * 1.0e2), wave_unit="nm")
    aa = Spectroscopy.nirspec_prism(jnp.asarray(WAVE_NM * 1.0e3))
    chex.assert_trees_all_equal(nm.wave_obs, aa.wave_obs)
    chex.assert_trees_all_equal(nm.resolution, aa.resolution)


def test_nirspec_g140m_um_matches_angstrom():
    wave_um = np.linspace(1.0, 2.0, 40)
    um = Spectroscopy.nirspec_g140m(jnp.asarray(wave_um), wave_unit="um")
    aa = Spectroscopy.nirspec_g140m(jnp.asarray(wave_um * 1.0e4))
    chex.assert_trees_all_equal(um.wave_obs, aa.wave_obs)
    chex.assert_trees_all_equal(um.resolution, aa.resolution)


def test_constant_r_nm_matches_angstrom():
    nm = Spectroscopy.constant_r(jnp.asarray(WAVE_NM), 1000.0, wave_unit="nm")
    aa = Spectroscopy.constant_r(jnp.asarray(WAVE_NM * 10.0), 1000.0)
    chex.assert_trees_all_equal(nm.wave_obs, aa.wave_obs)


def test_desi_like_nm_matches_angstrom():
    nm = Spectroscopy.desi_like(jnp.asarray(WAVE_NM), wave_unit="nm")
    aa = Spectroscopy.desi_like(jnp.asarray(WAVE_NM * 10.0))
    chex.assert_trees_all_equal(nm.wave_obs, aa.wave_obs)


# ── Refusals and warnings ─────────────────────────────────────────────


def test_range_below_floor_is_refused_for_filter(filter_name):
    wave_below = np.linspace(1.0e-4, 1.0e-3, 20)
    with pytest.raises(ValueError, match="wave_unit='um'"):
        register_filter(filter_name, wave_below, np.ones(20))
    assert filter_name not in custom._USER_FILTER_REGISTRY


def test_range_above_ceiling_is_refused_for_filter(filter_name):
    wave_above = np.linspace(2.0e9, 3.0e9, 20)
    with pytest.raises(ValueError, match="outside the plausible Angstrom range"):
        register_filter(filter_name, wave_above, np.ones(20))


def test_range_below_floor_is_refused_for_spectroscopy():
    with pytest.raises(ValueError, match="wave_unit='um'"):
        Spectroscopy(wave_obs=jnp.asarray(np.linspace(1.0e-4, 1.0e-3, 20)))


def test_extreme_uv_range_warns_for_spectroscopy_in_angstrom():
    with pytest.warns(UserWarning, match="extreme-UV"):
        Spectroscopy(wave_obs=jnp.asarray(np.linspace(380.0, 980.0, 100)))


def test_extreme_uv_range_does_not_warn_when_unit_is_stated():
    # 100-134 nm is 1000-1340 Å. Stated as nm it is settled, so it must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Spectroscopy(wave_obs=jnp.asarray(np.linspace(100.0, 134.0, 50)), wave_unit="nm")


def test_optical_nm_looking_range_as_filter_warns_only_when_inferred(filter_name):
    # Values that look like nm are only a warning in Angstrom, never a refusal.
    with pytest.warns(UserWarning, match="extreme-UV"):
        register_filter(filter_name, np.linspace(300.0, 700.0, 81), TRANS)
    unregister_filter(filter_name)


def test_shipped_filters_are_not_refused():
    # The measured floor is set below every shipped curve, so none is refused.
    from tengri.observation.filters import list_available_filters, load_filter

    names = list_available_filters().names()
    loaded = [load_filter(n) for n in names[:40]]
    assert loaded
    for fc in loaded:
        assert float(np.min(np.asarray(fc.wave))) >= WAVE_FLOOR_AA
        assert float(np.max(np.asarray(fc.wave))) <= WAVE_CEILING_AA


# ── Timescales ────────────────────────────────────────────────────────


def test_fixed_gyr_value_out_of_range_raises():
    with pytest.raises(ParameterError, match="sfh_dpl_tau_gyr"):
        Parameters(sfh_dpl_tau_gyr=Fixed(5000.0))


def test_fixed_myr_value_out_of_range_raises():
    # The _myr parameter is present only alongside a tsnorm SFH width, so pair it.
    with pytest.raises(ParameterError, match="sfh_field_psd_tau_myr"):
        Parameters(
            sfh_tsnorm_width_gyr=Fixed(1.0),
            sfh_field_psd_tau_myr=Fixed(1.0e-6),
        )


def test_check_timescale_names_parameter_value_and_unit():
    with pytest.raises(ParameterError) as info:
        check_timescale(5000.0, "sfh_dpl_age_gyr")
    assert "sfh_dpl_age_gyr" in str(info.value)
    assert "gyr" in str(info.value)


def test_check_timescale_ignores_other_suffixes():
    check_timescale(1.0e12, "redshift")


def test_declared_defaults_pass_timescale_bounds():
    spec = Parameters()
    assert "sfh_field_psd_tau_myr" in spec._distributions
    assert spec._distributions["sfh_field_psd_tau_myr"].bounds[0] >= 1.0e-3


def test_micron_looking_optical_range_is_accepted_as_angstrom_not_refused(filter_name):
    # 0.3-1.0 Å is a real NuSTAR band (0.157 Å is the shipped minimum). Range alone
    # cannot tell it from micron input, so it is accepted; the unit must be stated.
    register_filter(filter_name, np.linspace(0.3, 1.0, 40), np.ones(40))
    unregister_filter(filter_name)
    register_filter(filter_name, np.linspace(0.3, 1.0, 40), np.ones(40), wave_unit="um")
    stated = custom._USER_FILTER_REGISTRY[filter_name].wave
    chex.assert_trees_all_equal(stated, jnp.asarray(np.linspace(0.3, 1.0, 40) * 1.0e4))
