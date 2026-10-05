# SPDX-License-Identifier: BSD-3-Clause
"""Contract: SSP mass-remaining tables (#2614).

The reference table (packaged for FSPS MIST + Chabrier) provides
metallicity-dependent surviving-mass fractions. Tests ensure:

* The packaged `.dat` file matches the tracked h5 file exactly;
* Pins from #2614 hold at matching Z nodes and ages;
* Each attach condition (first filename token, isochrone token, IMF, age
  ladder, metallicity ladder, finite ages) is isolated: every negative case
  sits on the exact reference ladder unless the case is about the ladder;
* Matching grids use the table (Z-dependent), non-matching fall back to
  the sigmoid (Z-independent);
* dtype parity and a read-only table; the file's own table is preferred.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.contract


def _write_tiny_ssp(path: Path, age_nodes, met_nodes, **attrs) -> Path:
    """Write a minimal valid DSPS-format SSP file."""
    n_met = len(met_nodes)
    n_age = len(age_nodes)
    n_wave = 50
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, n_wave)
        f["ssp_flux"] = np.full((n_met, n_age, n_wave), 1e-4)
        f["ssp_lg_age_gyr"] = age_nodes
        f["ssp_lgmet"] = met_nodes
        for key, value in attrs.items():
            f.attrs[key] = value
    return path


def test_packaged_dat_matches_h5(tmp_path):
    """Packaged `.dat` == tracked `data/fsps_mass_remaining_chabrier.h5` exactly."""
    # Load from h5 file (repo root via parents)
    repo_root = Path(__file__).resolve().parents[3]
    h5_path = repo_root / "data" / "fsps_mass_remaining_chabrier.h5"
    assert h5_path.exists(), f"Tracked h5 file not found: {h5_path}"

    with h5py.File(h5_path, "r") as f:
        log_age_yr_h5 = f["log_age_yr"][:]
        z_absolute_h5 = f["z_absolute"][:]
        mass_remaining_h5 = f["mass_remaining"][:]

    # Load from packaged .dat via load_ssp_data mechanism
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    log_age_yr_dat, lgmet_absolute_dat, table_dat = _load_mass_remaining_reference(
        ("mist", "chabrier")
    )

    # Compare arrays exactly
    assert np.array_equal(log_age_yr_h5, log_age_yr_dat)
    assert np.array_equal(np.log10(z_absolute_h5), lgmet_absolute_dat)
    assert np.array_equal(mass_remaining_h5, table_dat)


def test_pins_from_2614(tmp_path):
    """Table at Z☉ and ages 1e8/1e9/1e10 yr match pins from #2614."""
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    log_age_yr, lgmet, table = _load_mass_remaining_reference(("mist", "chabrier"))

    # Find Z☉ (absolute Z = 0.0142) node, which is log10(0.0142) = -1.848
    z_sun = 0.0142
    lgmet_zsun = np.log10(z_sun)
    i_z_sun = np.argmin(np.abs(lgmet - lgmet_zsun))

    # Find age nodes: 1e8, 1e9, 1e10 yr = log age 8, 9, 10
    ages_target = [8, 9, 10]
    i_ages = [np.argmin(np.abs(log_age_yr - age)) for age in ages_target]

    # Expected pins from #2614
    pins = [0.7769, 0.6716, 0.5742]

    for i_age, age_log, pin in zip(i_ages, ages_target, pins):
        value = table[i_z_sun, i_age]
        assert np.abs(value - pin) < 5e-4, (
            f"Age 1e{age_log} yr: expected {pin}, got {value} (diff {abs(value - pin)})"
        )


def test_pins_min_max_at_1e10_yr(tmp_path):
    """Min and max over Z at 1e10 yr = 0.5544 / 0.5833 from #2614."""
    from tengri.components.stellar.sps.dsps_wrapper import (
        _load_mass_remaining_reference,
    )

    log_age_yr, _, table = _load_mass_remaining_reference(("mist", "chabrier"))

    # 1e10 yr = log age 10
    i_age_1e10 = np.argmin(np.abs(log_age_yr - 10))

    # Min and max over metallicity at that age
    min_val = np.min(table[:, i_age_1e10])
    max_val = np.max(table[:, i_age_1e10])

    assert np.abs(min_val - 0.5544) < 5e-4
    assert np.abs(max_val - 0.5833) < 5e-4


_MIST_CHAB = "fsps_mist_miles_chabrier.h5"
_CHAB_ATTR = "Chabrier (2003)"


@pytest.fixture(scope="module")
def reference():
    """Exact reference ladder from the packaged table: (age_gyr, lgmet, table)."""
    from tengri.components.stellar.sps.dsps_wrapper import _load_mass_remaining_reference

    log_age_yr, lgmet, table = _load_mass_remaining_reference(("mist", "chabrier"))
    return (log_age_yr - 9.0).astype(float), lgmet.astype(float), np.array(table)


def _load(path: Path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_ssp_data(str(path), **kw)


def _assert_sigmoid(mr):
    mr = np.asarray(mr)
    np.testing.assert_array_equal(mr, np.broadcast_to(mr[0], mr.shape))


@pytest.mark.parametrize(
    ("name", "attrs"),
    [
        pytest.param(_MIST_CHAB, {}, id="fsps_miles_name"),
        pytest.param("fsps_mist_c3k_a_chabrier.h5", {}, id="fsps_c3k_name"),
        pytest.param(
            "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5", {}, id="ssp_wne_name"
        ),
        pytest.param("fsps_mist_miles_noimftoken.h5", {"imf": _CHAB_ATTR}, id="chabrier_by_attr"),
    ],
)
def test_positive_uses_table(tmp_path, reference, name, attrs):
    """Each matching grid takes the table: equal to it and Z-dependent."""
    age, lgmet, table = reference
    ssp = _load(_write_tiny_ssp(tmp_path / name, age, lgmet, **attrs))
    np.testing.assert_allclose(np.asarray(ssp.ssp_mass_remaining), table, rtol=1e-12, atol=0)
    spread = np.ptp(np.asarray(ssp.ssp_mass_remaining), axis=0)
    assert spread[-1] > 0.02


def _shift_one(arr, i, d):
    out = np.array(arr)
    out[i] += d
    return out


# Each case: (name, attrs, ladder transform (age, lgmet) -> (age, lgmet), same_shape)
_NEGATIVE = [
    pytest.param("fsps_mist_miles_kroupa.h5", {}, None, True, id="name_kroupa"),
    pytest.param("fsps_mist_miles_salpeter.h5", {}, None, True, id="name_salpeter"),
    pytest.param(_MIST_CHAB, {"imf": "kroupa"}, None, True, id="attr_kroupa_wins"),
    pytest.param("pgny_mist_c3k_chabrier.h5", {}, None, True, id="first_token_pgny"),
    pytest.param("bpss_mist_c3k_chabrier.h5", {}, None, True, id="first_token_bpss"),
    pytest.param("fsps_prsc_miles_chabrier.h5", {}, None, True, id="isochrone_prsc"),
    pytest.param(_MIST_CHAB, {}, lambda a, z: (a + 0.01, z), True, id="age_shift"),
    pytest.param(_MIST_CHAB, {}, lambda a, z: (a, _shift_one(z, 5, 0.01)), True, id="met_shift"),
    pytest.param(_MIST_CHAB, {}, lambda a, z: (a + 1e-4, z), True, id="age_shift_1e-4"),
    pytest.param(
        _MIST_CHAB, {}, lambda a, z: (a, _shift_one(z, 5, 1e-4)), True, id="met_shift_1e-4"
    ),
    pytest.param(_MIST_CHAB, {}, lambda a, z: (np.delete(a, 40), z), False, id="age_node_removed"),
    pytest.param(_MIST_CHAB, {}, lambda a, z: (np.r_[-np.inf, a[1:]], z), True, id="age0_anchor"),
]


@pytest.mark.parametrize(("name", "attrs", "ladder", "same_shape"), _NEGATIVE)
def test_negative_falls_back_to_sigmoid(tmp_path, reference, name, attrs, ladder, same_shape):
    """Exactly one condition fails: the sigmoid (Z-independent) is used."""
    age, lgmet, table = reference
    if ladder is not None:
        age, lgmet = ladder(age, lgmet)
    ssp = _load(_write_tiny_ssp(tmp_path / name, age, lgmet, **attrs))
    mr = np.asarray(ssp.ssp_mass_remaining)
    _assert_sigmoid(mr)
    if same_shape:
        assert not np.allclose(mr, table, equal_nan=False)


def test_attr_imf_is_authoritative_over_name():
    """`_detect_imf` returns the HDF5 attribute when present, ignoring the name."""
    from tengri.components.stellar.sps.dsps_wrapper import _detect_imf

    handle = SimpleNamespace(attrs={"imf": "Kroupa (2001)"})
    assert _detect_imf(handle, _MIST_CHAB) == "kroupa (2001)"


@pytest.mark.parametrize("imf", ["", "   ", None])
def test_direct_call_empty_imf_returns_none(reference, imf):
    from tengri.components.stellar.sps.dsps_wrapper import _reference_mass_remaining

    age, lgmet, _ = reference
    assert _reference_mass_remaining("fsps_mist_miles_chabrier", imf, age, lgmet) is None


@pytest.mark.parametrize("dtype", [None, jnp.float32], ids=["default", "float32"])
def test_dtype_parity_and_table_untouched(tmp_path, reference, dtype):
    """mass_remaining shares ssp_flux's dtype, both JAX arrays; table not mutated."""
    from tengri.components.stellar.sps.dsps_wrapper import _load_mass_remaining_reference

    age, lgmet, _ = reference
    path = _write_tiny_ssp(tmp_path / _MIST_CHAB, age, lgmet)
    before = tuple(np.array(a) for a in _load_mass_remaining_reference(("mist", "chabrier")))
    for _ in range(2):
        ssp = _load(path, dtype=dtype)
        assert isinstance(ssp.ssp_mass_remaining, jax.Array)
        assert isinstance(ssp.ssp_flux, jax.Array)
        assert ssp.ssp_mass_remaining.dtype == ssp.ssp_flux.dtype
        if dtype is not None:
            assert ssp.ssp_mass_remaining.dtype == dtype
    fresh = _load_mass_remaining_reference.__wrapped__(("mist", "chabrier"))
    for a, b, c in zip(before, _load_mass_remaining_reference(("mist", "chabrier")), fresh):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)


def test_file_own_table_preferred(tmp_path, reference):
    """File's own ssp_mass_remaining is preferred over the reference."""
    age, lgmet, _ = reference
    path = _write_tiny_ssp(
        tmp_path / _MIST_CHAB,
        age,
        lgmet,
        imf=_CHAB_ATTR,
        isochrone="mist",
        spectral_library="miles",
    )
    with h5py.File(path, "a") as f:
        f["ssp_mass_remaining"] = np.full((len(lgmet), len(age)), 0.5)
    assert np.allclose(np.asarray(_load(path).ssp_mass_remaining), 0.5)


def test_alpha_axis_grid_does_not_get_the_2d_table(tmp_path, reference):
    """A grid with an [alpha/Fe] axis never takes the (n_met, n_age) table.

    The loader gives such a grid the metallicity-independent DSPS sigmoid,
    shape (n_met, n_age), identical across metallicity.
    """
    age, lgmet, table = reference
    n_alpha, n_wave = 3, 50
    path = tmp_path / _MIST_CHAB
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, n_wave)
        f["ssp_flux"] = np.full((len(lgmet), n_alpha, len(age), n_wave), 1e-4)
        f["ssp_lg_age_gyr"] = age
        f["ssp_lgmet"] = lgmet
        f["ssp_alpha_fe"] = np.array([0.0, 0.2, 0.4])
        f.attrs["imf"] = _CHAB_ATTR
    mr = np.asarray(_load(path).ssp_mass_remaining)
    assert mr.shape == table.shape
    _assert_sigmoid(mr)
    assert not np.allclose(mr, table)


def test_reference_arrays_are_read_only():
    """The cached reference arrays cannot be written through (all three)."""
    from tengri.components.stellar.sps.dsps_wrapper import _load_mass_remaining_reference

    for arr in _load_mass_remaining_reference(("mist", "chabrier")):
        with pytest.raises(ValueError, match="read-only"):
            arr[(0,) * arr.ndim] = 0.0
