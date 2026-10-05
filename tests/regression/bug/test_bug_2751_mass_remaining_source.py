"""Surviving mass comes from each grid's own isochrones, or the load refuses (#2751).

``ssp_mass_remaining[Z, age]`` is the fraction of the formed mass in living stars plus
remnants (Conroy, Gunn & White 2009; FSPS remnants after Renzini & Ciotti 1993; BC03
after Bruzual & Charlot 2003). Before the fix every grid without its own table silently
received DSPS's metallicity-independent sigmoid fit, whatever its isochrones: the
surviving mass of a MIST Kroupa grid ignored the 0.03 spread across Z at 10 Gyr, and
a BC03 grid read an FSPS fit.

Measured on the committed tables at log10(age/yr) = 10, solar-ish node (index 9,
log10 Z = -1.848):

    Salpeter 0.7468 > Kroupa 0.5998 > Chabrier 0.5742

(FSPS ordering: a bottom-heavier IMF locks more mass in surviving low-mass stars).
"""

from __future__ import annotations

import warnings
from importlib.resources import files
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

REPO = Path(__file__).resolve().parents[3]
LOG_AGE_10 = 10.0


def _load_ssp_data(*args, **kwargs):
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_ssp_data(*args, **kwargs)


def _table(name: str):
    """(log10 age [yr], log10 Z, table) of a shipped companion file, read directly."""
    path = files("tengri.data.ssp_mass_remaining") / name
    with h5py.File(str(path), "r") as f:
        return f["log10_age_yr"][:], f["log10_z_abs"][:], f["mass_remaining"][:]


def _write_grid(path: Path, lg_age_gyr, lgmet, mass_remaining=None, **attrs) -> Path:
    lg_age_gyr = np.asarray(lg_age_gyr, dtype=float)
    lgmet = np.asarray(lgmet, dtype=float)
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, 20)
        f["ssp_flux"] = np.full((len(lgmet), len(lg_age_gyr), 20), 1e-4)
        f["ssp_lg_age_gyr"] = lg_age_gyr
        f["ssp_lgmet"] = lgmet
        if mass_remaining is not None:
            f["ssp_mass_remaining"] = mass_remaining
        for key, value in attrs.items():
            f.attrs[key] = value
    return path


def _grid_on_table(tmp_path, name, table_file, **kw):
    age, logz, tab = _table(table_file)
    return _write_grid(tmp_path / name, age - 9.0, logz, **kw), age, logz, tab


# ---------------------------------------------------------------------------
# The fix: each MIST grid takes its own IMF's table, which varies with Z.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("imf", ["chabrier", "kroupa", "salpeter"])
@pytest.mark.parametrize("library", ["miles", "c3k_a"])
def test_mist_grid_resolves_to_its_companion_table(tmp_path, imf, library):
    """MIST grids equal the companion table at the nodes to 1e-12, and say so."""
    table_file = f"mass_remaining_mist_{imf}.h5"
    path, _, _, tab = _grid_on_table(tmp_path, f"fsps_mist_{library}_{imf}.h5", table_file)
    ssp = _load_ssp_data(str(path))
    assert ssp.mass_remaining_source == f"companion:{table_file}"
    np.testing.assert_allclose(np.asarray(ssp.ssp_mass_remaining), tab, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("imf", ["chabrier", "kroupa", "salpeter"])
def test_surviving_mass_varies_with_metallicity(tmp_path, imf):
    """Spread over Z at log10 age 10 is >= 0.01 (measured ~0.03); the fit gave 0."""
    path, age, _, _ = _grid_on_table(
        tmp_path, f"fsps_mist_miles_{imf}.h5", f"mass_remaining_mist_{imf}.h5"
    )
    mr = np.asarray(_load_ssp_data(str(path)).ssp_mass_remaining)
    column = mr[:, np.argmin(np.abs(age - LOG_AGE_10))]
    assert column.max() - column.min() >= 0.01


def test_imf_ordering_at_log_age_10(tmp_path):
    """Salpeter > Kroupa > Chabrier at the solar-ish node (values in the module docstring)."""
    values = {}
    for imf in ("chabrier", "kroupa", "salpeter"):
        path, age, _, _ = _grid_on_table(
            tmp_path, f"fsps_mist_miles_{imf}.h5", f"mass_remaining_mist_{imf}.h5"
        )
        mr = np.asarray(_load_ssp_data(str(path)).ssp_mass_remaining)
        values[imf] = mr[9, np.argmin(np.abs(age - LOG_AGE_10))]
    assert values["salpeter"] > values["kroupa"] > values["chabrier"]
    assert values["chabrier"] == pytest.approx(0.5742, abs=5e-4)
    assert values["kroupa"] == pytest.approx(0.5998, abs=5e-4)
    assert values["salpeter"] == pytest.approx(0.7468, abs=5e-4)


def test_prsc_chabrier_companion_matches_the_grid_own_table(tmp_path):
    """The repackaged PARSEC table is the one the PARSEC grid carries (cross-check passes)."""
    age, logz, tab = _table("mass_remaining_prsc_chabrier.h5")
    path = _write_grid(tmp_path / "fsps_prsc_c3k_a_chabrier.h5", age - 9.0, logz)
    wne = _write_grid(
        tmp_path / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5", age - 9.0, logz
    )
    for p in (path, wne):
        ssp = _load_ssp_data(str(p))
        assert ssp.mass_remaining_source == "companion:mass_remaining_prsc_chabrier.h5"
        np.testing.assert_allclose(np.asarray(ssp.ssp_mass_remaining), tab, rtol=1e-12, atol=0)
    emb = _write_grid(
        tmp_path / "fsps_prsc_miles_chabrier.h5", age - 9.0, logz, mass_remaining=tab
    )
    assert _load_ssp_data(str(emb)).mass_remaining_source == "embedded"


# ---------------------------------------------------------------------------
# Age placement: node tolerance, t = 0, one-node clamp.
# ---------------------------------------------------------------------------
def test_ages_within_1e5_dex_of_a_node_take_the_node_value(tmp_path):
    age, logz, tab = _table("mass_remaining_mist_chabrier.h5")
    jitter = np.where(np.arange(age.size) % 2 == 0, 4e-6, -4e-6)
    path = _write_grid(tmp_path / "fsps_mist_miles_chabrier.h5", age + jitter - 9.0, logz)
    mr = np.asarray(_load_ssp_data(str(path)).ssp_mass_remaining)
    np.testing.assert_allclose(mr, tab, rtol=1e-12, atol=0.0)


def test_t_zero_node_is_one_not_extrapolated(tmp_path):
    age, logz, tab = _table("mass_remaining_bc03pdva94_chabrier.h5")
    ages = np.concatenate([[-np.inf], age - 9.0])
    path = _write_grid(tmp_path / "bc03_pdva_stelib_chabrier.h5", ages, logz)
    ssp = _load_ssp_data(str(path))
    mr = np.asarray(ssp.ssp_mass_remaining)
    assert ssp.mass_remaining_source == "companion:mass_remaining_bc03pdva94_chabrier.h5"
    assert np.all(mr[:, 0] == 1.0)
    np.testing.assert_allclose(mr[:, 1:], tab, rtol=1e-12, atol=0.0)


def test_age_within_one_node_beyond_the_table_is_clamped(tmp_path):
    age, logz, tab = _table("mass_remaining_mist_chabrier.h5")
    extra = age[-1] + 0.5 * (age[-1] - age[-2])
    path = _write_grid(tmp_path / "fsps_mist_miles_chabrier.h5", np.r_[age, extra] - 9.0, logz)
    mr = np.asarray(_load_ssp_data(str(path)).ssp_mass_remaining)
    np.testing.assert_array_equal(mr[:, -1], tab[:, -1])


def test_age_beyond_one_node_raises(tmp_path):
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")
    extra = age[-1] + 3.0 * (age[-1] - age[-2])
    path = _write_grid(tmp_path / "fsps_mist_miles_chabrier.h5", np.r_[age, extra] - 9.0, logz)
    with pytest.raises(ValueError, match="more than one table node beyond"):
        _load_ssp_data(str(path))


# ---------------------------------------------------------------------------
# Refusals: every error path, exact exception.
# ---------------------------------------------------------------------------
def test_pending_grid_without_opt_in_raises_naming_table_and_opt_in(tmp_path):
    age = np.linspace(5.0, 10.0, 20) - 9.0
    path = _write_grid(tmp_path / "fsps_pdva_miles_chabrier.h5", age, [-4.0, -2.0])
    with pytest.raises(ValueError, match=r"mass_remaining_pdva_chabrier\.h5.*dsps_fit"):
        _load_ssp_data(str(path))


def test_metallicity_mismatch_raises_listing_both(tmp_path):
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")
    shifted = logz.copy()
    shifted[5] += 1e-3
    path = _write_grid(tmp_path / "fsps_mist_miles_chabrier.h5", age - 9.0, shifted)
    with pytest.raises(ValueError, match=r"(?s)grid  log10\(Z\).*table log10\(Z\)"):
        _load_ssp_data(str(path))


def test_unreadable_companion_propagates_the_io_error(tmp_path, monkeypatch):
    from tengri.components.stellar.sps import mass_remaining_tables as mrt

    monkeypatch.setitem(
        mrt.MASS_REMAINING_REGISTRY,
        "fsps_testonly_miles_chabrier",
        mrt.MassRemainingEntry("mist", "chabrier", "mass_remaining_does_not_exist.h5"),
    )
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")
    path = _write_grid(tmp_path / "fsps_testonly_miles_chabrier.h5", age - 9.0, logz)
    with pytest.raises(FileNotFoundError):
        _load_ssp_data(str(path))


def test_embedded_table_disagreeing_with_companion_raises(tmp_path):
    age, logz, tab = _table("mass_remaining_mist_chabrier.h5")
    path = _write_grid(
        tmp_path / "fsps_mist_miles_chabrier.h5", age - 9.0, logz, mass_remaining=tab + 1e-3
    )
    with pytest.raises(ValueError, match="differs from mass_remaining_mist_chabrier.h5"):
        _load_ssp_data(str(path))


def test_embedded_table_agreeing_with_companion_is_used(tmp_path):
    age, logz, tab = _table("mass_remaining_mist_chabrier.h5")
    path = _write_grid(
        tmp_path / "fsps_mist_miles_chabrier.h5", age - 9.0, logz, mass_remaining=tab
    )
    ssp = _load_ssp_data(str(path))
    assert ssp.mass_remaining_source == "embedded"
    np.testing.assert_array_equal(np.asarray(ssp.ssp_mass_remaining), tab)


def test_grid_imf_contradicting_registry_raises(tmp_path):
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")
    path = _write_grid(
        tmp_path / "fsps_mist_miles_chabrier.h5", age - 9.0, logz, imf="Kroupa (2001)"
    )
    with pytest.raises(ValueError, match="declares IMF"):
        _load_ssp_data(str(path))


def test_unknown_mode_raises(tmp_path):
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")
    path = _write_grid(tmp_path / "fsps_mist_miles_chabrier.h5", age - 9.0, logz)
    with pytest.raises(ValueError, match="mass_remaining must be one of"):
        _load_ssp_data(str(path), mass_remaining="fit")


# ---------------------------------------------------------------------------
# The opt-in and user grids.
# ---------------------------------------------------------------------------
def _expected_fit(lg_age_gyr, n_met, params):
    import jax.numpy as jnp
    from dsps.imf.surviving_mstar import surviving_mstar

    from tengri.utils.ssp_anchor import ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR

    lg_age_yr = jnp.maximum(jnp.asarray(lg_age_gyr) + 9.0, ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR)
    return np.broadcast_to(
        np.asarray(surviving_mstar(lg_age_yr, **params)), (n_met, lg_age_yr.size)
    )


def test_opt_in_reproduces_the_dsps_fit_bit_exactly(tmp_path):
    from dsps.imf.surviving_mstar import KROUPA_PARAMS

    age = np.linspace(5.0, 10.1, 40) - 9.0
    path = _write_grid(tmp_path / "fsps_pdva_miles_kroupa.h5", age, [-4.0, -3.0, -2.0])
    ssp = _load_ssp_data(str(path), mass_remaining="dsps_fit")
    assert ssp.mass_remaining_source == "dsps_fit"
    np.testing.assert_array_equal(
        np.asarray(ssp.ssp_mass_remaining), _expected_fit(age, 3, KROUPA_PARAMS)
    )


def test_opt_in_overrides_even_a_table_it_could_have_used(tmp_path):
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")
    path = _write_grid(tmp_path / "fsps_mist_miles_chabrier.h5", age - 9.0, logz)
    ssp = _load_ssp_data(str(path), mass_remaining="dsps_fit")
    assert ssp.mass_remaining_source == "dsps_fit"
    mr = np.asarray(ssp.ssp_mass_remaining)
    np.testing.assert_array_equal(mr, np.broadcast_to(mr[0], mr.shape))


def test_user_grid_without_table_warns_naming_the_opt_in(tmp_path):
    from dsps.imf.surviving_mstar import CHABRIER_PARAMS

    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    age = np.linspace(5.0, 10.1, 40) - 9.0
    path = _write_grid(tmp_path / "my_own_grid.h5", age, [-3.0, -2.0], imf="chabrier")
    with pytest.warns(UserWarning, match="dsps_fit"):
        ssp = load_ssp_data(str(path))
    assert ssp.mass_remaining_source == "dsps_fit"
    np.testing.assert_array_equal(
        np.asarray(ssp.ssp_mass_remaining), _expected_fit(age, 2, CHABRIER_PARAMS)
    )


def test_user_grid_with_its_own_table_keeps_it_silently(tmp_path):
    age = np.linspace(5.0, 10.1, 10) - 9.0
    mr = np.full((2, 10), 0.7)
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    path = _write_grid(tmp_path / "my_own_grid.h5", age, [-3.0, -2.0], mass_remaining=mr)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ssp = load_ssp_data(str(path))
    assert ssp.mass_remaining_source == "embedded"
    np.testing.assert_array_equal(np.asarray(ssp.ssp_mass_remaining), mr)


def test_alpha_axis_grid_without_table_never_takes_the_companion(tmp_path):
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")
    path = tmp_path / "fsps_mist_miles_chabrier.h5"
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, 20)
        f["ssp_flux"] = np.full((len(logz), 3, len(age), 20), 1e-4)
        f["ssp_lg_age_gyr"] = age - 9.0
        f["ssp_lgmet"] = logz
        f["ssp_alpha_fe"] = np.array([0.0, 0.2, 0.4])
    with pytest.raises(ValueError, match=r"alpha/Fe"):
        _load_ssp_data(str(path))


def test_hand_built_ssp_declares_no_source_and_keeps_it_through_pytrees():
    import jax.numpy as jnp

    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    ssp = SSPData(
        ssp_wave=jnp.ones(3),
        ssp_flux=jnp.ones((1, 2, 3)),
        ssp_lg_age_gyr=jnp.zeros(2),
        ssp_lgmet=jnp.zeros(1),
        ssp_mass_remaining=jnp.ones((1, 2)),
        mass_remaining_source="embedded",
    )
    assert SSPData(*[None] * 4).mass_remaining_source == "unspecified"
    back = jax.tree_util.tree_map(lambda x: x, ssp)
    assert back.mass_remaining_source == "embedded"


# ---------------------------------------------------------------------------
# Registry contract.
# ---------------------------------------------------------------------------
def test_every_catalog_grid_is_registered_with_a_table_or_pending():
    from tengri._data_setup import _KNOWN_SSPS
    from tengri.components.stellar.sps import mass_remaining_tables as mrt

    for name in _KNOWN_SSPS:
        entry = mrt.MASS_REMAINING_REGISTRY.get(mrt.canonical_grid_name(name))
        assert entry is not None, f"{name} has no registry entry"
        assert entry.imf in ("chabrier", "kroupa", "salpeter")
        if entry.source in (mrt.PENDING, mrt.EMBEDDED):
            continue
        age, logz, tab = mrt.load_companion_table(entry.source, entry.isoc, entry.imf)
        assert tab.shape == (logz.size, age.size)


def test_registry_pending_set_is_exactly_the_declared_one():
    from tengri.components.stellar.sps import mass_remaining_tables as mrt

    pending_isoc = {
        e.isoc for e in mrt.MASS_REMAINING_REGISTRY.values() if e.source == mrt.PENDING
    }
    assert pending_isoc == {"prsc", "pdva", "bsti", "bpss", "pgny_mist"}
    pending_prsc = {
        e.imf
        for e in mrt.MASS_REMAINING_REGISTRY.values()
        if e.source == mrt.PENDING and e.isoc == "prsc"
    }
    assert pending_prsc == {"kroupa", "salpeter"}
    bc03 = mrt.MASS_REMAINING_REGISTRY["bc03_pdva_stelib_chabrier"]
    assert bc03.source == "mass_remaining_bc03pdva94_chabrier.h5"


def test_no_locally_present_grid_silently_uses_the_fit():
    """Each SSP grid on disk resolves to a table or refuses; none lands on 'dsps_fit'."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    grids = sorted((REPO / "data").glob("fsps_*.h5")) + sorted((REPO / "data").glob("ssp_*.h5"))
    grids += sorted((REPO / "data").glob("bc03_*.h5")) + sorted((REPO / "data").glob("pgny_*.h5"))
    grids = [g for g in grids if "mass_remaining" not in g.name and "nebular" not in g.name]
    if not grids:
        pytest.skip("no SSP grid present locally")
    for grid in grids:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                ssp = load_ssp_data(str(grid))
        except ValueError as err:
            assert "dsps_fit" in str(err), f"{grid.name}: {err}"
            continue
        assert ssp.mass_remaining_source != "dsps_fit", grid.name
        assert ssp.mass_remaining_source == "embedded" or ssp.mass_remaining_source.startswith(
            "companion:"
        )


def test_bc03_grid_resolves_to_its_companion_with_t0_equal_one():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    for root in (REPO / "data", Path("/Users/suchethacooray/Projects/tengri/data")):
        grid = root / "bc03_pdva_stelib_chabrier.h5"
        if grid.is_file():
            break
    else:
        pytest.skip("BC03 grid not present locally")
    ssp = load_ssp_data(str(grid))
    assert ssp.mass_remaining_source == "companion:mass_remaining_bc03pdva94_chabrier.h5"
    lg_age = np.asarray(ssp.ssp_lg_age_gyr)
    mr = np.asarray(ssp.ssp_mass_remaining)
    assert np.isneginf(lg_age[0]) and np.all(mr[:, 0] == 1.0)


def test_synthetic_fixture_under_a_catalog_name_is_not_held_to_the_registry(tmp_path):
    """A file declaring ``synthetic`` (test fixture) keeps its own table, unchecked."""
    age, logz, _ = _table("mass_remaining_mist_chabrier.h5")[0], [-4.0, -2.0], None
    mr = np.full((2, age.size), 0.5)
    path = _write_grid(
        tmp_path / "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
        age - 9.0,
        logz,
        mass_remaining=mr,
        synthetic=True,
    )
    ssp = _load_ssp_data(str(path))
    assert ssp.mass_remaining_source == "embedded"
    np.testing.assert_array_equal(np.asarray(ssp.ssp_mass_remaining), mr)
