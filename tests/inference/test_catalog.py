"""Tests for photometric catalog reader.

Tests Feature 4 — the ``read_catalog`` function and ``Catalog`` dataclass
in ``tengri.observation.catalog``.
"""

import pytest

pytestmark = pytest.mark.contract

import numpy as np
import numpy.testing as npt
import pytest

from tengri.observation.catalog import (
    CANDELS_TO_TENGRI,
    CIGALE_TO_TENGRI,
    Catalog,
    read_catalog,
)
from tengri.observation.noise import DETECTED, LOWER_LIMIT, UPPER_LIMIT

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def basic_csv(tmp_path):
    """CSV with 3 galaxies, 3 SDSS filters, all detected."""
    path = tmp_path / "basic.csv"
    path.write_text(
        "id,redshift,sdss_u,sdss_u_err,sdss_g,sdss_g_err,sdss_r,sdss_r_err\n"
        "gal1,0.5,0.10,0.01,0.20,0.02,0.30,0.03\n"
        "gal2,1.0,0.15,0.015,0.25,0.025,0.35,0.035\n"
        "gal3,1.5,0.05,0.005,0.10,0.01,0.15,0.015\n"
    )
    return path


@pytest.fixture()
def mixed_csv(tmp_path):
    """CSV with all data types: detected, upper limit, lower limit, missing."""
    path = tmp_path / "mixed.csv"
    path.write_text(
        "id,redshift,sdss_u,sdss_u_err,sdss_g,sdss_g_err,sdss_r,sdss_r_err\n"
        "gal1,0.5,0.10,0.01,0.20,0.02,0.30,0.03\n"
        "gal2,1.0,0.05,-0.01,0.15,0.015,-9999,-9999\n"
        "gal3,1.5,-0.08,0.02,0.25,0.025,0.40,0.04\n"
    )
    return path


@pytest.fixture()
def custom_mapping_csv(tmp_path):
    """CSV with non-standard column names needing a filter mapping."""
    path = tmp_path / "custom.csv"
    path.write_text(
        "id,redshift,band_u,band_u_err,band_g,band_g_err\nobj1,0.8,0.20,0.02,0.30,0.03\n"
    )
    return path


@pytest.fixture()
def tsv_file(tmp_path):
    """Tab-separated file."""
    path = tmp_path / "tab.tsv"
    path.write_text("id\tredshift\tsdss_r\tsdss_r_err\ngal1\t0.5\t0.30\t0.03\n")
    return path


# ── Basic parsing ─────────────────────────────────────────────────


class TestReadCatalogBasic:
    """Tests for basic CSV parsing."""

    def test_returns_catalog(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert isinstance(cat, Catalog)

    def test_correct_dimensions(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert cat.n_galaxies == 3
        assert cat.n_filters == 3

    def test_flux_shape(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert cat.flux.shape == (3, 3)

    def test_noise_shape(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert cat.noise.shape == (3, 3)

    def test_mask_shape(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert cat.mask.shape == (3, 3)

    def test_ids_correct(self, basic_csv):
        cat = read_catalog(basic_csv)
        npt.assert_array_equal(cat.ids, ["gal1", "gal2", "gal3"])

    def test_redshifts_correct(self, basic_csv):
        cat = read_catalog(basic_csv)
        npt.assert_allclose(cat.redshifts, [0.5, 1.0, 1.5])

    def test_filter_names(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert set(cat.filter_names) == {"sdss_u", "sdss_g", "sdss_r"}

    def test_filter_names_is_tuple(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert isinstance(cat.filter_names, tuple)

    def test_flux_values(self, basic_csv):
        cat = read_catalog(basic_csv)
        # First galaxy: sdss_u=0.10, sdss_g=0.20, sdss_r=0.30
        u_idx = cat.filter_names.index("sdss_u")
        g_idx = cat.filter_names.index("sdss_g")
        r_idx = cat.filter_names.index("sdss_r")
        npt.assert_allclose(cat.flux[0, u_idx], 0.10)
        npt.assert_allclose(cat.flux[0, g_idx], 0.20)
        npt.assert_allclose(cat.flux[0, r_idx], 0.30)

    def test_noise_values(self, basic_csv):
        cat = read_catalog(basic_csv)
        u_idx = cat.filter_names.index("sdss_u")
        npt.assert_allclose(cat.noise[0, u_idx], 0.01)

    def test_all_detected(self, basic_csv):
        """All fluxes are positive with positive errors → all detected."""
        cat = read_catalog(basic_csv)
        assert np.all(cat.mask == DETECTED)

    def test_flux_unit_default(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert cat.flux_unit == "mJy"

    def test_flux_unit_custom(self, basic_csv):
        cat = read_catalog(basic_csv, flux_unit="uJy")
        assert cat.flux_unit == "uJy"


# ── Mask conventions ──────────────────────────────────────────────


class TestMaskConventions:
    """Tests for detection, upper limit, lower limit, and missing conventions."""

    def test_upper_limit(self, mixed_csv):
        """Positive flux + negative error → upper limit."""
        cat = read_catalog(mixed_csv)
        u_idx = cat.filter_names.index("sdss_u")
        assert cat.mask[1, u_idx] == UPPER_LIMIT
        npt.assert_allclose(cat.flux[1, u_idx], 0.05)
        npt.assert_allclose(cat.noise[1, u_idx], 0.01)  # abs of -0.01

    def test_lower_limit(self, mixed_csv):
        """Negative flux + positive error → lower limit."""
        cat = read_catalog(mixed_csv)
        u_idx = cat.filter_names.index("sdss_u")
        assert cat.mask[2, u_idx] == LOWER_LIMIT
        npt.assert_allclose(cat.flux[2, u_idx], 0.08)  # abs of -0.08
        npt.assert_allclose(cat.noise[2, u_idx], 0.02)

    def test_missing_data(self, mixed_csv):
        """flux=-9999, err=-9999 → masked (detected with huge noise)."""
        cat = read_catalog(mixed_csv)
        r_idx = cat.filter_names.index("sdss_r")
        assert cat.mask[1, r_idx] == DETECTED
        npt.assert_allclose(cat.flux[1, r_idx], 0.0)
        assert cat.noise[1, r_idx] == 1e30

    def test_detected(self, mixed_csv):
        """Positive flux + positive error → detected."""
        cat = read_catalog(mixed_csv)
        g_idx = cat.filter_names.index("sdss_g")
        assert cat.mask[0, g_idx] == DETECTED
        npt.assert_allclose(cat.flux[0, g_idx], 0.20)
        npt.assert_allclose(cat.noise[0, g_idx], 0.02)

    def test_zero_error_gets_huge_noise(self, tmp_path):
        """Zero error value → noise set to 1e30 (effectively infinite)."""
        path = tmp_path / "zero_err.csv"
        path.write_text("id,redshift,sdss_r,sdss_r_err\ngal1,0.5,0.30,0.0\n")
        cat = read_catalog(path)
        assert cat.noise[0, 0] == 1e30


# ── Catalog methods ───────────────────────────────────────────────


class TestCatalogMethods:
    """Tests for Catalog.galaxy() and Catalog.select_detected()."""

    def test_galaxy_returns_dict(self, basic_csv):
        cat = read_catalog(basic_csv)
        g = cat.galaxy(0)
        assert isinstance(g, dict)

    def test_galaxy_keys(self, basic_csv):
        cat = read_catalog(basic_csv)
        g = cat.galaxy(0)
        expected = {"id", "redshift", "flux", "noise", "mask", "filter_names"}
        assert set(g.keys()) == expected

    def test_galaxy_redshift(self, basic_csv):
        cat = read_catalog(basic_csv)
        g = cat.galaxy(1)
        assert g["redshift"] == 1.0

    def test_galaxy_id(self, basic_csv):
        cat = read_catalog(basic_csv)
        g = cat.galaxy(2)
        assert g["id"] == "gal3"

    def test_galaxy_filter_names_shared(self, basic_csv):
        """All galaxies share the same filter_names tuple."""
        cat = read_catalog(basic_csv)
        g0 = cat.galaxy(0)
        g1 = cat.galaxy(1)
        assert g0["filter_names"] is g1["filter_names"]

    def test_galaxy_flux_shape(self, basic_csv):
        cat = read_catalog(basic_csv)
        g = cat.galaxy(0)
        assert g["flux"].shape == (3,)

    def test_select_detected_all_detected(self, basic_csv):
        """All bands detected → returns all bands."""
        cat = read_catalog(basic_csv)
        flux, noise, names = cat.select_detected(0)
        assert len(flux) == 3
        assert len(noise) == 3
        assert len(names) == 3

    def test_select_detected_filters_upper_limit(self, mixed_csv):
        """Upper limit band excluded from detected selection."""
        cat = read_catalog(mixed_csv)
        # gal2 (idx=1): sdss_u=upper limit, sdss_r=missing (but DETECTED mask)
        _flux, _noise, names = cat.select_detected(1)
        assert "sdss_u" not in names

    def test_select_detected_filters_lower_limit(self, mixed_csv):
        """Lower limit band excluded from detected selection."""
        cat = read_catalog(mixed_csv)
        # gal3 (idx=2): sdss_u=lower limit
        _flux, _noise, names = cat.select_detected(2)
        assert "sdss_u" not in names

    def test_select_detected_flux_noise_match(self, mixed_csv):
        """Returned flux/noise arrays correspond to the returned names."""
        cat = read_catalog(mixed_csv)
        flux, noise, names = cat.select_detected(0)
        for i, name in enumerate(names):
            j = cat.filter_names.index(name)
            npt.assert_allclose(flux[i], cat.flux[0, j])
            npt.assert_allclose(noise[i], cat.noise[0, j])


# ── Properties ────────────────────────────────────────────────────


class TestCatalogProperties:
    """Tests for Catalog dataclass properties."""

    def test_n_galaxies(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert cat.n_galaxies == 3

    def test_n_filters(self, basic_csv):
        cat = read_catalog(basic_csv)
        assert cat.n_filters == 3

    def test_repr(self, basic_csv):
        cat = read_catalog(basic_csv)
        r = repr(cat)
        assert "3 galaxies" in r
        assert "3 filters" in r
        assert "mJy" in r

    def test_frozen(self, basic_csv):
        """Catalog is immutable (frozen dataclass)."""
        cat = read_catalog(basic_csv)
        with pytest.raises(AttributeError):
            cat.flux_unit = "uJy"


# ── Filter mapping ────────────────────────────────────────────────


class TestFilterMapping:
    """Tests for custom filter_mapping parameter."""

    def test_custom_mapping(self, custom_mapping_csv):
        mapping = {"band_u": "sdss_u", "band_g": "sdss_g"}
        cat = read_catalog(custom_mapping_csv, filter_mapping=mapping)
        assert cat.n_filters == 2
        assert "sdss_u" in cat.filter_names
        assert "sdss_g" in cat.filter_names

    def test_no_mapping_unrecognized_cols_ignored(self, custom_mapping_csv):
        """Without mapping, unrecognized column names yield no filters."""
        with pytest.raises(ValueError, match="No filter columns"):
            read_catalog(custom_mapping_csv)

    def test_partial_mapping(self, custom_mapping_csv):
        """Mapping only one of two columns → only that filter."""
        mapping = {"band_u": "sdss_u"}
        cat = read_catalog(custom_mapping_csv, filter_mapping=mapping)
        assert cat.n_filters == 1
        assert cat.filter_names == ("sdss_u",)


# ── Pre-built mappings ────────────────────────────────────────────


class TestPrebuiltMappings:
    """Tests for CIGALE_TO_TENGRI and CANDELS_TO_TENGRI mappings."""

    def test_cigale_mapping_nonempty(self):
        assert len(CIGALE_TO_TENGRI) > 0

    def test_candels_mapping_nonempty(self):
        assert len(CANDELS_TO_TENGRI) > 0

    def test_cigale_values_in_registry(self):
        """All tengri names in CIGALE mapping exist in FILTER_REGISTRY."""
        from tengri.observation.filters import FILTER_REGISTRY

        for cigale_name, tengri_name in CIGALE_TO_TENGRI.items():
            assert tengri_name in FILTER_REGISTRY, (
                f"CIGALE mapping '{cigale_name}' → '{tengri_name}' not found in FILTER_REGISTRY"
            )

    def test_candels_values_in_registry(self):
        """All tengri names in CANDELS mapping exist in FILTER_REGISTRY."""
        from tengri.observation.filters import FILTER_REGISTRY

        for candels_name, tengri_name in CANDELS_TO_TENGRI.items():
            assert tengri_name in FILTER_REGISTRY, (
                f"CANDELS mapping '{candels_name}' → '{tengri_name}' not found in FILTER_REGISTRY"
            )

    def test_cigale_mapping_with_csv(self, tmp_path):
        """Use CIGALE_TO_TENGRI mapping to read a CIGALE-style catalog."""
        path = tmp_path / "cigale_style.csv"
        path.write_text(
            "id,redshift,GALEX_FUV,GALEX_FUV_err,GALEX_NUV,GALEX_NUV_err\n"
            "obj1,0.3,0.01,0.001,0.02,0.002\n"
        )
        cat = read_catalog(path, filter_mapping=CIGALE_TO_TENGRI)
        assert "galex_fuv" in cat.filter_names
        assert "galex_nuv" in cat.filter_names

    def test_candels_mapping_with_csv(self, tmp_path):
        """Use CANDELS_TO_TENGRI mapping to read a CANDELS-style catalog."""
        path = tmp_path / "candels_style.csv"
        path.write_text(
            "id,redshift,ACS_F814W,ACS_F814W_err,WFC3_F160W,WFC3_F160W_err\n"
            "obj1,1.2,0.05,0.005,0.10,0.01\n"
        )
        cat = read_catalog(path, filter_mapping=CANDELS_TO_TENGRI)
        assert "hst_f814w" in cat.filter_names
        assert "hst_f160w" in cat.filter_names


# ── Custom columns ────────────────────────────────────────────────


class TestCustomColumns:
    """Tests for custom id_col, redshift_col, delimiter."""

    def test_custom_id_col(self, tmp_path):
        path = tmp_path / "custom_id.csv"
        path.write_text("objname,redshift,sdss_r,sdss_r_err\nNGC1234,0.5,0.30,0.03\n")
        cat = read_catalog(path, id_col="objname")
        assert cat.ids[0] == "NGC1234"

    def test_custom_redshift_col(self, tmp_path):
        path = tmp_path / "custom_z.csv"
        path.write_text("id,z_phot,sdss_r,sdss_r_err\ngal1,0.8,0.30,0.03\n")
        cat = read_catalog(path, redshift_col="z_phot")
        npt.assert_allclose(cat.redshifts[0], 0.8)

    def test_tab_delimiter(self, tsv_file):
        cat = read_catalog(tsv_file, delimiter="\t")
        assert cat.n_galaxies == 1
        assert "sdss_r" in cat.filter_names

    def test_custom_missing_value(self, tmp_path):
        path = tmp_path / "custom_missing.csv"
        path.write_text("id,redshift,sdss_r,sdss_r_err\ngal1,0.5,-99,-99\n")
        cat = read_catalog(path, missing_value=-99.0)
        assert cat.flux[0, 0] == 0.0
        assert cat.noise[0, 0] == 1e30


# ── Error handling ────────────────────────────────────────────────


class TestReadCatalogErrors:
    """Tests for error handling in read_catalog()."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_catalog("/nonexistent/path/catalog.csv")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("id,redshift,sdss_r,sdss_r_err\n")
        with pytest.raises(ValueError, match="Empty catalog"):
            read_catalog(path)

    def test_no_filter_columns(self, tmp_path):
        path = tmp_path / "no_filters.csv"
        path.write_text("id,redshift,random_col,random_col_err\ngal1,0.5,1.0,0.1\n")
        with pytest.raises(ValueError, match="No filter columns"):
            read_catalog(path)

    def test_accepts_path_object(self, basic_csv):
        """Works with pathlib.Path and str."""
        cat_path = read_catalog(basic_csv)
        cat_str = read_catalog(str(basic_csv))
        npt.assert_array_equal(cat_path.flux, cat_str.flux)

    def test_no_headers_in_file(self, tmp_path):
        """File without proper CSV headers."""
        path = tmp_path / "no_header.csv"
        path.write_text("")
        with pytest.raises(ValueError):
            read_catalog(path)


# ── Multi-galaxy consistency ──────────────────────────────────────


class TestMultiGalaxyConsistency:
    """Tests verifying consistency across galaxies in a catalog."""

    def test_galaxy_method_round_trips(self, basic_csv):
        """galaxy(i) data matches direct array indexing."""
        cat = read_catalog(basic_csv)
        for i in range(cat.n_galaxies):
            g = cat.galaxy(i)
            npt.assert_array_equal(g["flux"], cat.flux[i])
            npt.assert_array_equal(g["noise"], cat.noise[i])
            npt.assert_array_equal(g["mask"], cat.mask[i])
            assert g["redshift"] == cat.redshifts[i]

    def test_select_detected_subset(self, mixed_csv):
        """select_detected returns a strict subset of all bands."""
        cat = read_catalog(mixed_csv)
        for i in range(cat.n_galaxies):
            _flux_det, _noise_det, names_det = cat.select_detected(i)
            assert len(names_det) <= cat.n_filters
            for name in names_det:
                assert name in cat.filter_names
