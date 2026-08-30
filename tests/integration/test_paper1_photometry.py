"""Integration tests for the CANDELS photometry pipeline of the framework paper (#2089).

Bug: ``fit_one.extract_photometry`` carried a private AB zero point of
3.63e-23 (3.63 Jy: 1000x too small) and a private column map whose five ACS
keys were spelled ``WFC3_*``; a ``continue`` guard hid the mismatch, so
galaxy 13097 fit 8 bands of fluxes 1000x too faint and every NUTS transition
diverged.

Mutation checks (each test names the mutant that kills it):
- zero point back to 3.63e-23: ``test_ab_mag_to_fnu_matches_the_ab_definition``,
  ``test_extract_photometry_13097_returns_every_usable_band``.
- ``ACS_F435W`` key renamed back to ``WFC3_F435W``:
  ``test_the_map_is_exactly_the_documented_one``,
  ``test_every_value_is_a_tengri_filter_and_every_key_a_catalog_column``,
  ``test_extract_photometry_13097_returns_every_usable_band``.
- ``raise KeyError`` -> ``continue``:
  ``test_a_missing_mapped_column_raises_instead_of_dropping_the_band``.
- drop the ``ks_taken`` rule: ``test_one_ks_band_isaac_first_hawki_only_as_fallback``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PAPER1 = REPO / "analysis" / "paper1"
# fit_one.py does a bare ``from candels_io import ...``: the directory must be importable.
if str(PAPER1) not in sys.path:
    sys.path.insert(0, str(PAPER1))

import candels_io
import fit_one

import tengri

EXPECTED_MAP = {
    "ACS_F435W": "hst_f435w",
    "ACS_F606W": "hst_f606w",
    "ACS_F775W": "hst_f775w",
    "ACS_F814W": "hst_f814w",
    "ACS_F850LP": "hst_f850lp",
    "WFC3_F098M": "hst_f098m",
    "WFC3_F105W": "hst_f105w",
    "WFC3_F125W": "hst_f125w",
    "WFC3_F160W": "hst_f160w",
    "ISAAC_KS": "vista_ks",
    "HAWKI_KS": "vista_ks",
    "IRAC_CH1": "irac_36",
    "IRAC_CH2": "irac_45",
    "IRAC_CH3": "irac_58",
    "IRAC_CH4": "irac_80",
}


@pytest.fixture(scope="module")
def catalog():
    try:
        return candels_io.load_candels_z1()
    except FileNotFoundError:
        pytest.skip("CANDELS catalog not found")


@pytest.fixture(scope="module")
def registry_names():
    return {row["alias"] for row in tengri.list_filters()}


def test_ab_mag_to_fnu_matches_the_ab_definition():
    fnu, err = candels_io.ab_mag_to_fnu(21.342, 0.001)
    expected = 3.631e-20 * 10 ** (-21.342 / 2.5)
    assert float(fnu) == pytest.approx(expected, rel=1e-3, abs=0)
    assert float(fnu) == pytest.approx(1.055e-28, rel=2e-3, abs=0)
    assert float(err) == pytest.approx(expected * np.log(10) / 2.5 * 0.001, rel=1e-3, abs=0)


def test_ab_mag_to_fnu_is_elementwise():
    fnu, err = candels_io.ab_mag_to_fnu(np.array([20.0, 25.0]), np.array([0.01, 0.1]))
    assert fnu.shape == (2,) and err.shape == (2,)
    assert fnu[0] / fnu[1] == pytest.approx(100.0)


def test_the_map_is_exactly_the_documented_one():
    assert candels_io.CANDELS_TO_TENGRI == EXPECTED_MAP
    assert list(candels_io.CANDELS_TO_TENGRI) == list(EXPECTED_MAP)
    assert candels_io.KS_COLUMNS == ("ISAAC_KS", "HAWKI_KS")
    assert "CTIO_U" not in candels_io.CANDELS_TO_TENGRI
    assert "VIMOS_U" not in candels_io.CANDELS_TO_TENGRI


def test_every_value_is_a_tengri_filter_and_every_key_a_catalog_column(catalog, registry_names):
    for column, name in candels_io.CANDELS_TO_TENGRI.items():
        assert name in registry_names, name
        assert column in catalog["header"], column
        assert f"e{column}" in catalog["header"], column


def test_load_candels_z1_carries_the_data_matrix(catalog):
    assert catalog["data"].shape == (len(catalog["id"]), len(catalog["header"]))


def _row_from(mags: dict[str, tuple[float, float]]):
    header = ["ID"] + [x for c in candels_io.CANDELS_TO_TENGRI for x in (c, f"e{c}")]
    row = np.zeros(len(header))
    for column in candels_io.CANDELS_TO_TENGRI:
        mag, err = mags.get(column, (22.0, 0.05))
        row[header.index(column)] = mag
        row[header.index(f"e{column}")] = err
    return header, row


def test_a_missing_mapped_column_raises_instead_of_dropping_the_band():
    header, row = _row_from({})
    i = header.index("ACS_F435W")
    header_without = header[:i] + header[i + 1 :]
    row_without = np.delete(row, i)
    with pytest.raises(KeyError, match="ACS_F435W"):
        candels_io.photometry_for_row(header_without, row_without)


def test_one_ks_band_isaac_first_hawki_only_as_fallback():
    header, row = _row_from({"ISAAC_KS": (21.0, 0.01), "HAWKI_KS": (22.0, 0.01)})
    names, fnu, _ = candels_io.photometry_for_row(header, row)
    assert names.count("vista_ks") == 1
    isaac_flux = float(candels_io.ab_mag_to_fnu(21.0, 0.01)[0])
    assert fnu[names.index("vista_ks")] == pytest.approx(isaac_flux, rel=1e-12, abs=0)

    header, row = _row_from({"ISAAC_KS": (98.992, -99.0), "HAWKI_KS": (22.0, 0.01)})
    names, fnu, _ = candels_io.photometry_for_row(header, row)
    assert names.count("vista_ks") == 1
    hawki_flux = float(candels_io.ab_mag_to_fnu(22.0, 0.01)[0])
    assert fnu[names.index("vista_ks")] == pytest.approx(hawki_flux, rel=1e-12, abs=0)


def test_photometry_for_row_skips_sentinels_and_orders_by_the_map():
    header, row = _row_from({"WFC3_F098M": (98.992, -99.0)})
    names, fnu, err = candels_io.photometry_for_row(header, row)
    assert "hst_f098m" not in names
    assert names[:5] == ["hst_f435w", "hst_f606w", "hst_f775w", "hst_f814w", "hst_f850lp"]
    assert len(names) == len(fnu) == len(err) == 13
    assert np.all(err > 0)


def test_extract_photometry_13097_returns_every_usable_band(catalog, registry_names):
    z = float(catalog["z"][np.where(catalog["id"] == 13097)[0][0]])
    gal_id, z_out, names, fnu, err = fit_one.extract_photometry(13097, catalog, z)
    assert gal_id == 13097 and z_out == z
    # Every usable band: 15 map keys -> 14 distinct filters (both Ks columns share
    # ``vista_ks``) minus ``WFC3_F098M``, a sentinel for this galaxy. The buggy
    # private map returned 8 (the five ACS bands were spelled ``WFC3_*``).
    assert len(names) == 13
    assert "hst_f435w" in names
    assert "hst_f098m" not in names  # sentinel in the catalog for this galaxy
    assert names.count("vista_ks") == 1
    assert set(names) <= registry_names
    assert np.all((fnu > 5e-30) & (fnu < 5e-28)), fnu
    assert np.all(err > 0) and np.all(err < fnu)


def test_thin_samples_and_iter_draws_use_flattened_draws():
    rng = np.random.default_rng(0)
    samples = {"a": rng.normal(size=9000), "b": rng.normal(size=9000)}
    thin = fit_one.thin_samples(samples)
    assert thin["a"].ndim == 1
    assert 0 < thin["a"].shape[0] <= fit_one.MAX_SAVED_DRAWS
    assert thin["a"][1] == samples["a"][3]  # 9000 draws -> every third one
    assert fit_one.thin_samples({"a": np.arange(2400.0)})["a"].shape == (2400,)

    draws = list(fit_one.iter_draws(thin, {"fixed": 1.5}, 5))
    assert len(draws) == 5
    assert draws[0] == {"fixed": 1.5, "a": float(thin["a"][0]), "b": float(thin["b"][0])}
    assert list(fit_one.iter_draws({"a": np.arange(3.0)}, {}, 10)) == [
        {"a": 0.0},
        {"a": 1.0},
        {"a": 2.0},
    ]


def test_driver_timeout_covers_a_retune():
    import inspect

    import run_candels_fits

    assert run_candels_fits.DEFAULT_FIT_TIMEOUT_S >= 1800
    default = inspect.signature(run_candels_fits.run_fit_subprocess).parameters["timeout"].default
    assert default == run_candels_fits.DEFAULT_FIT_TIMEOUT_S
