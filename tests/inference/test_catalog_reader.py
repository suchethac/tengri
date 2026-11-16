# SPDX-License-Identifier: BSD-3-Clause
"""Photometric catalog reading, on the one reader (#3547 D4).

The CSV parsing and filter-mapping tests ported from the removed
``observation.catalog.read_catalog`` run through :func:`tengri.io.load_catalog`.
Mask and unit behavior runs through :func:`tengri.inference.catalog_ingest.ingest_catalog`,
the kept path that turns a table into arrays, with ``lim_flag="full"`` where the
table carries CIGALE limits. The sentinel-masking tests of the removed reader are not
ported: the owner ruled that a sentinel is refused unless declared.
"""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from tengri.io import load_catalog
from tengri.observation.noise import DETECTED, UPPER_LIMIT

pytestmark = pytest.mark.contract


def _ingest_bands(cat, *, flux_unit="mJy", **kw):
    """Feed ``load_catalog`` bands through the kept path: a flat table into ``ingest_catalog``."""
    from tengri.inference.catalog_ingest import ingest_catalog
    from tengri.observation import Photometry

    table = {}
    for band, cols in cat["bands"].items():
        table[band] = cols["flux"]
        table[f"{band}_err"] = cols["err"]
    phot = Photometry.from_names(list(cat["bands"]))
    return ingest_catalog(table, photometry=phot, flux_unit=flux_unit, **kw)


_SDSS = {"sdss_u": "sdss_u", "sdss_g": "sdss_g", "sdss_r": "sdss_r"}

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
    """CSV with all data types: detected, upper limit, negative detection, sentinel."""
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


class TestCatalogParsing:
    """CSV parsing through load_catalog."""

    def test_correct_dimensions(self, basic_csv):
        cat = load_catalog(basic_csv, filter_mapping=_SDSS)
        assert cat["n_rows"] == 3
        assert len(cat["bands"]) == 3

    def test_flux_and_noise_shape(self, basic_csv):
        cat = load_catalog(basic_csv, filter_mapping=_SDSS)
        for cols in cat["bands"].values():
            assert cols["flux"].shape == (3,)
            assert cols["err"].shape == (3,)

    def test_ids_correct(self, basic_csv):
        cat = load_catalog(basic_csv, filter_mapping=_SDSS)
        npt.assert_array_equal(cat["ids"], ["gal1", "gal2", "gal3"])

    def test_redshifts_correct(self, basic_csv):
        cat = load_catalog(basic_csv, filter_mapping=_SDSS)
        npt.assert_allclose(cat["redshift"], [0.5, 1.0, 1.5])

    def test_filter_names(self, basic_csv):
        cat = load_catalog(basic_csv, filter_mapping=_SDSS)
        assert set(cat["bands"]) == {"sdss_u", "sdss_g", "sdss_r"}

    def test_flux_values(self, basic_csv):
        cat = load_catalog(basic_csv, filter_mapping=_SDSS)
        npt.assert_allclose(cat["bands"]["sdss_u"]["flux"][0], 0.10)
        npt.assert_allclose(cat["bands"]["sdss_g"]["flux"][0], 0.20)
        npt.assert_allclose(cat["bands"]["sdss_r"]["flux"][0], 0.30)

    def test_noise_values(self, basic_csv):
        cat = load_catalog(basic_csv, filter_mapping=_SDSS)
        npt.assert_allclose(cat["bands"]["sdss_u"]["err"][0], 0.01)

    def test_detected_values_through_ingest(self, mixed_csv):
        """Positive flux + positive error is a detection with its own flux and error."""
        ca = _ingest_bands(
            load_catalog(mixed_csv, filter_mapping=_SDSS), flux_unit="cgs_fnu", lim_flag="full"
        )
        g_idx = ca.band_names.index("sdss_g")
        assert ca.censor[0, g_idx] == 0
        npt.assert_allclose(ca.flux[0, g_idx], 0.20)
        npt.assert_allclose(ca.noise[0, g_idx], 0.02)


# ── Mask conventions, through the kept path ───────────────────────


class TestMaskConventions:
    """Detection, upper limit and negative detection, read by ingest_catalog."""

    def test_upper_limit(self, mixed_csv):
        """Positive flux + negative error -> upper limit at the flux, sigma = |error|."""
        ca = _ingest_bands(
            load_catalog(mixed_csv, filter_mapping=_SDSS), flux_unit="cgs_fnu", lim_flag="full"
        )
        u_idx = ca.band_names.index("sdss_u")
        assert ca.censor[1, u_idx] == UPPER_LIMIT
        npt.assert_allclose(ca.flux[1, u_idx], 0.05)
        npt.assert_allclose(ca.noise[1, u_idx], 0.01)  # abs of -0.01

    def test_lower_limit(self, mixed_csv):
        """Negative flux + positive error -> detection with signed flux."""
        ca = _ingest_bands(
            load_catalog(mixed_csv, filter_mapping=_SDSS), flux_unit="cgs_fnu", lim_flag="full"
        )
        u_idx = ca.band_names.index("sdss_u")
        assert ca.censor[2, u_idx] == DETECTED
        npt.assert_allclose(ca.flux[2, u_idx], -0.08)  # signed flux
        npt.assert_allclose(ca.noise[2, u_idx], 0.02)

    @pytest.mark.parametrize(
        "flux_val,err_val,expected_flux,expected_noise,expected_mask",
        [
            (-0.004, 0.010, -0.004, 0.010, DETECTED),  # detection, below zero
            (0.05, 0.01, 0.05, 0.01, DETECTED),  # detection, positive
            (0.05, -0.01, 0.05, 0.01, UPPER_LIMIT),  # CIGALE: any negative error
            (0.0, -0.01, 0.0, 0.01, UPPER_LIMIT),  # CIGALE: zero flux + negative error
        ],
    )
    def test_flux_error_combinations(
        self, tmp_path, flux_val, err_val, expected_flux, expected_noise, expected_mask
    ):
        """Each flux/error combination without a sentinel, read against the CIGALE convention."""
        path = tmp_path / "combinations.csv"
        path.write_text(f"id,redshift,sdss_r,sdss_r_err\ngal1,0.5,{flux_val},{err_val}\n")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cat = load_catalog(path, filter_mapping=_SDSS)
            ca = _ingest_bands(cat, flux_unit="cgs_fnu", lim_flag="full")
        npt.assert_allclose(ca.flux[0, 0], expected_flux)
        npt.assert_allclose(ca.noise[0, 0], expected_noise)
        assert ca.censor[0, 0] == expected_mask

    def test_negative_detection_likelihood(self, tmp_path):
        """Negative detection contributes Gaussian likelihood with signed flux."""
        import jax.numpy as jnp

        from tengri.observation.noise import censored_neg_log_likelihood

        path = tmp_path / "neg_det.csv"
        path.write_text("id,redshift,sdss_r,sdss_r_err\ngal1,0.5,-0.004,0.010\n")
        cat = load_catalog(path, filter_mapping=_SDSS)
        ca = _ingest_bands(cat, flux_unit="cgs_fnu", lim_flag="full")

        # Verify it was read as a detection with signed flux
        assert ca.censor[0, 0] == DETECTED
        npt.assert_allclose(ca.flux[0, 0], -0.004)
        npt.assert_allclose(ca.noise[0, 0], 0.010)

        # Compute likelihood with model flux = 0 (no prediction)
        # For a detected band: E = 0.5 * ((d - m) / sigma)^2 = 0.5 * (-0.4)^2 = 0.08
        data = jnp.array(ca.flux[0])
        noise = jnp.array(ca.noise[0])
        predicted = jnp.array([0.0])  # Model prediction is zero
        mask = jnp.array(ca.censor[0], dtype=int)

        nll = censored_neg_log_likelihood(data, noise, predicted, mask, f_cal=0.0)
        # The energy should be 0.5 * ((−0.004 − 0) / 0.010)² = 0.08 (plus log(noise))
        expected_r_squared = 0.5 * ((-0.004 / 0.010) ** 2)
        expected_nll = expected_r_squared + np.log(0.010)
        npt.assert_allclose(nll, expected_nll, rtol=1e-5)


# ── Filter mapping ────────────────────────────────────────────────


class TestFilterMapping:
    """Custom filter_mapping parameter."""

    def test_custom_mapping(self, custom_mapping_csv):
        mapping = {"band_u": "sdss_u", "band_g": "sdss_g"}
        cat = load_catalog(custom_mapping_csv, filter_mapping=mapping)
        assert len(cat["bands"]) == 2
        assert "sdss_u" in cat["bands"]
        assert "sdss_g" in cat["bands"]

    def test_no_mapping_unrecognized_cols_ignored(self, custom_mapping_csv):
        """Without mapping, unrecognized column names yield no bands."""
        with pytest.raises(ValueError, match="No flux / error column pairs"):
            load_catalog(custom_mapping_csv)

    def test_partial_mapping(self, custom_mapping_csv):
        """Mapping only one of two columns -> only that band."""
        cat = load_catalog(custom_mapping_csv, filter_mapping={"band_u": "sdss_u"})
        assert tuple(cat["bands"]) == ("sdss_u",)


# ── Pre-built mappings ────────────────────────────────────────────


class TestPrebuiltMappings:
    """CIGALE_TO_TENGRI and CANDELS_TO_TENGRI."""

    def test_cigale_mapping_nonempty(self):
        from tengri.io import CIGALE_TO_TENGRI

        assert len(CIGALE_TO_TENGRI) > 0

    def test_candels_mapping_nonempty(self):
        from tengri.io import CANDELS_TO_TENGRI

        assert len(CANDELS_TO_TENGRI) > 0

    def test_cigale_values_in_registry(self):
        """All tengri names in CIGALE mapping exist in FILTER_REGISTRY."""
        from tengri.io import CIGALE_TO_TENGRI
        from tengri.observation.filters import FILTER_REGISTRY

        for cigale_name, tengri_name in CIGALE_TO_TENGRI.items():
            assert tengri_name in FILTER_REGISTRY, (
                f"CIGALE mapping '{cigale_name}' → '{tengri_name}' not found in FILTER_REGISTRY"
            )

    def test_candels_values_in_registry(self):
        """All tengri names in CANDELS mapping exist in FILTER_REGISTRY."""
        from tengri.io import CANDELS_TO_TENGRI
        from tengri.observation.filters import FILTER_REGISTRY

        for candels_name, tengri_name in CANDELS_TO_TENGRI.items():
            assert tengri_name in FILTER_REGISTRY, (
                f"CANDELS mapping '{candels_name}' → '{tengri_name}' not found in FILTER_REGISTRY"
            )

    def test_cigale_mapping_with_csv(self, tmp_path):
        """Use CIGALE_TO_TENGRI mapping to read a CIGALE-style catalog."""
        from tengri.io import CIGALE_TO_TENGRI

        path = tmp_path / "cigale_style.csv"
        path.write_text(
            "id,redshift,GALEX_FUV,GALEX_FUV_err,GALEX_NUV,GALEX_NUV_err\n"
            "obj1,0.3,0.01,0.001,0.02,0.002\n"
        )
        cat = load_catalog(path, filter_mapping=CIGALE_TO_TENGRI)
        assert "galex_fuv" in cat["bands"]
        assert "galex_nuv" in cat["bands"]

    def test_candels_mapping_with_csv(self, tmp_path):
        """Use CANDELS_TO_TENGRI mapping to read a CANDELS-style catalog."""
        from tengri.io import CANDELS_TO_TENGRI

        path = tmp_path / "candels_style.csv"
        path.write_text(
            "id,redshift,ACS_F814W,ACS_F814W_err,WFC3_F160W,WFC3_F160W_err\n"
            "obj1,1.2,0.05,0.005,0.10,0.01\n"
        )
        cat = load_catalog(path, filter_mapping=CANDELS_TO_TENGRI)
        assert "hst_f814w" in cat["bands"]
        assert "hst_f160w" in cat["bands"]


# ── Custom columns ────────────────────────────────────────────────


class TestCustomColumns:
    """Custom id_col, redshift_col, delimiter."""

    def test_custom_id_col(self, tmp_path):
        path = tmp_path / "custom_id.csv"
        path.write_text("objname,redshift,sdss_r,sdss_r_err\nNGC1234,0.5,0.30,0.03\n")
        cat = load_catalog(path, id_col="objname", filter_mapping={"sdss_r": "sdss_r"})
        assert cat["ids"][0] == "NGC1234"

    def test_custom_redshift_col(self, tmp_path):
        path = tmp_path / "custom_z.csv"
        path.write_text("id,z_phot,sdss_r,sdss_r_err\ngal1,0.8,0.30,0.03\n")
        cat = load_catalog(path, redshift_col="z_phot", filter_mapping={"sdss_r": "sdss_r"})
        npt.assert_allclose(cat["redshift"][0], 0.8)

    def test_tab_delimiter(self, tsv_file):
        cat = load_catalog(tsv_file, delimiter="\t", filter_mapping={"sdss_r": "sdss_r"})
        assert cat["n_rows"] == 1
        assert "sdss_r" in cat["bands"]


# ── Error handling ────────────────────────────────────────────────


class TestCatalogErrors:
    """Error handling in load_catalog()."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match=r"catalog\.csv"):
            load_catalog("/nonexistent/path/catalog.csv")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("id,redshift,sdss_r,sdss_r_err\n")
        with pytest.raises(ValueError, match="Empty catalog"):
            load_catalog(path, filter_mapping={"sdss_r": "sdss_r"})

    def test_no_filter_columns(self, tmp_path):
        path = tmp_path / "no_filters.csv"
        path.write_text("id,redshift,random_col,random_col_err\ngal1,0.5,1.0,0.1\n")
        with pytest.raises(ValueError, match="No flux / error column pairs"):
            load_catalog(path, filter_mapping=_SDSS)

    def test_accepts_path_object(self, basic_csv):
        """Works with pathlib.Path and str."""
        cat_path = load_catalog(basic_csv, filter_mapping=_SDSS)
        cat_str = load_catalog(str(basic_csv), filter_mapping=_SDSS)
        npt.assert_array_equal(
            cat_path["bands"]["sdss_r"]["flux"], cat_str["bands"]["sdss_r"]["flux"]
        )

    def test_no_headers_in_file(self, tmp_path):
        """File without proper CSV headers."""
        path = tmp_path / "no_header.csv"
        path.write_text("")
        with pytest.raises(ValueError):
            load_catalog(path, filter_mapping=_SDSS)
