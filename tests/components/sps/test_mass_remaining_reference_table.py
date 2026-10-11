# SPDX-License-Identifier: BSD-3-Clause
"""Contract: companion surviving-mass tables and how the loader attaches them (#2614, #2751).

* Pins from #2614 hold on the MIST Chabrier table (shipped as package data);
* a grid whose registry entry names a companion takes it (Z-dependent), with
  dtype parity with ``ssp_flux`` and read-only cached arrays;
* a grid's own table, when it agrees with the companion, is the one used;
* a grid the registry does not name, or names without a table, never falls to the
  fit silently (the full refusal matrix is in
  ``tests/regression/bug/test_bug_2751_mass_remaining_source.py``).
"""

from __future__ import annotations

import warnings
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.stellar.sps.mass_remaining_tables import load_companion_table

pytestmark = pytest.mark.contract

_MIST_CHAB = "fsps_mist_miles_chabrier.h5"


def _write_tiny_ssp(path: Path, age_gyr, met_nodes, **attrs) -> Path:
    n_wave = 50
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, n_wave)
        f["ssp_flux"] = np.full((len(met_nodes), len(age_gyr), n_wave), 1e-4)
        f["ssp_lg_age_gyr"] = age_gyr
        f["ssp_lgmet"] = met_nodes
        for key, value in attrs.items():
            f.attrs[key] = value
    return path


@pytest.fixture(scope="module")
def reference():
    """(log10 age [Gyr], log10 Z, table) of the MIST Chabrier companion."""
    age_yr, lgmet, table = load_companion_table(
        "mass_remaining_mist_chabrier.h5", "mist", "chabrier"
    )
    return age_yr - 9.0, np.array(lgmet), np.array(table)


def _load(path: Path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_ssp_data(str(path), **kw)


def test_pins_from_2614(reference):
    """Solar-metallicity node at 1e8/1e9/1e10 yr matches the #2614 pins."""
    age_gyr, lgmet, table = reference
    i_z = np.argmin(np.abs(lgmet - np.log10(0.0142)))
    for log_age_yr, pin in zip([8, 9, 10], [0.7769, 0.6716, 0.5742]):
        i_age = np.argmin(np.abs(age_gyr + 9.0 - log_age_yr))
        assert abs(table[i_z, i_age] - pin) < 5e-4


def test_pins_min_max_over_z_at_1e10_yr(reference):
    """Min and max over Z at 1e10 yr: 0.5544 / 0.5833 (#2614)."""
    age_gyr, _, table = reference
    column = table[:, np.argmin(np.abs(age_gyr + 9.0 - 10.0))]
    assert abs(column.min() - 0.5544) < 5e-4
    assert abs(column.max() - 0.5833) < 5e-4


@pytest.mark.parametrize(
    "name",
    [
        _MIST_CHAB,
        "fsps_mist_c3k_a_chabrier.h5",
        "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    ],
)
def test_registered_names_take_the_table(tmp_path, reference, name):
    """Each registered MIST Chabrier name (also the wNE variant) equals the table, Z-dependent."""
    age, lgmet, table = reference
    ssp = _load(_write_tiny_ssp(tmp_path / name, age, lgmet))
    np.testing.assert_allclose(np.asarray(ssp.ssp_mass_remaining), table, rtol=1e-12, atol=0)
    assert np.ptp(np.asarray(ssp.ssp_mass_remaining), axis=0)[-1] > 0.02


def test_attr_imf_is_authoritative_over_name():
    """`_detect_imf` returns the HDF5 attribute when present, ignoring the name."""
    from tengri.components.stellar.sps.dsps_wrapper import _detect_imf

    handle = SimpleNamespace(attrs={"imf": "Kroupa (2001)"})
    assert _detect_imf(handle, _MIST_CHAB) == "kroupa (2001)"


def test_attr_imf_cannot_contradict_the_registry(tmp_path, reference):
    """A grid attribute naming another IMF than its registry row is refused."""
    age, lgmet, _ = reference
    path = _write_tiny_ssp(tmp_path / _MIST_CHAB, age, lgmet, imf="Kroupa (2001)")
    with pytest.raises(ValueError, match="declares IMF"):
        _load(path)


@pytest.mark.parametrize("dtype", [None, jnp.float32], ids=["default", "float32"])
def test_dtype_parity_and_table_untouched(tmp_path, reference, dtype):
    """mass_remaining shares ssp_flux's dtype, both JAX arrays; the cached table is not mutated."""
    age, lgmet, _ = reference
    path = _write_tiny_ssp(tmp_path / _MIST_CHAB, age, lgmet)
    args = ("mass_remaining_mist_chabrier.h5", "mist", "chabrier")
    before = tuple(np.array(a) for a in load_companion_table(*args))
    for _ in range(2):
        ssp = _load(path, dtype=dtype)
        assert isinstance(ssp.ssp_mass_remaining, jax.Array)
        assert ssp.ssp_mass_remaining.dtype == ssp.ssp_flux.dtype
        if dtype is not None:
            assert ssp.ssp_mass_remaining.dtype == dtype
    fresh = load_companion_table.__wrapped__(*args)
    for a, b, c in zip(before, load_companion_table(*args), fresh):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)


def test_reference_arrays_are_read_only():
    """The cached companion arrays cannot be written through (all three)."""
    for arr in load_companion_table("mass_remaining_mist_chabrier.h5", "mist", "chabrier"):
        with pytest.raises(ValueError, match="read-only"):
            arr[(0,) * arr.ndim] = 0.0


def test_package_data_directory_holds_every_registered_companion():
    """Every companion the registry names ships in the package data directory."""
    from tengri.components.stellar.sps.mass_remaining_tables import (
        EMBEDDED,
        MASS_REMAINING_REGISTRY,
        PENDING,
    )

    shipped = {p.name for p in files("tengri.data.ssp_mass_remaining").iterdir()}
    named = {
        e.source for e in MASS_REMAINING_REGISTRY.values() if e.source not in (PENDING, EMBEDDED)
    }
    assert named <= shipped
