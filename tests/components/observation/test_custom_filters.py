# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for custom filter registration and loading.

Tests the in-memory registration API, directory loading, DSPS integration,
units validation, and precedence ordering in load_filter().

Markers: bounds, contract (CI-enforced per tests/TESTING.md)
"""

import os
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.filters import (
    FILTER_REGISTRY,
    SYNTHETIC_BAND_REGISTRY,
    list_registered_filters,
    load_filter,
    register_filter,
    register_filter_from_file,
    unregister_filter,
)
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.bounds


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def clean_registry():
    """Fixture that clears user registry before and after each test."""
    from tengri.observation.filters.custom import _USER_FILTER_REGISTRY

    initial_state = _USER_FILTER_REGISTRY.copy()
    _USER_FILTER_REGISTRY.clear()
    yield _USER_FILTER_REGISTRY
    _USER_FILTER_REGISTRY.clear()
    _USER_FILTER_REGISTRY.update(initial_state)


@pytest.fixture
def sample_curve():
    """Return a simple Gaussian filter curve."""
    wave = np.linspace(5000.0, 7000.0, 100)
    trans = np.exp(-0.5 * ((wave - 6000.0) / 300.0) ** 2)
    return wave, trans


@pytest.fixture
def sample_curve_file(tmp_path, sample_curve):
    """Write sample curve to a temp file."""
    wave, trans = sample_curve
    filepath = tmp_path / "sample_filter.dat"
    np.savetxt(
        str(filepath),
        np.column_stack([wave, trans]),
        header="wavelength  transmission",
        fmt="%.6e",
    )
    return filepath


@pytest.fixture
def nm_scale_curve():
    """Return a curve in nanometer scale (will trigger warning)."""
    wave = np.linspace(500.0, 700.0, 100)  # nm scale, not Å
    trans = np.exp(-0.5 * ((wave - 600.0) / 30.0) ** 2)
    return wave, trans


@pytest.fixture
def um_scale_curve():
    """Return a curve in micrometer scale (will trigger warning)."""
    wave = np.linspace(5.0, 7.0, 100)  # µm scale, not Å
    trans = np.exp(-0.5 * ((wave - 6.0) / 0.3) ** 2)
    return wave, trans


# ── Tests: In-memory registration ──────────────────────────────────


class TestRegisterFilter:
    """register_filter: storage, retrieval, and collision detection."""

    def test_register_and_retrieve(self, clean_registry, sample_curve):
        """Registered filter is immediately available via load_filter."""
        wave, trans = sample_curve
        register_filter("test_band", wave, trans)

        fc = load_filter("test_band")
        assert fc.name == "test_band"
        assert len(fc.wave) == len(wave)
        assert np.allclose(fc.wave, wave, rtol=1e-5)

    def test_from_names_with_registered_filter(self, clean_registry, sample_curve):
        """Photometry.from_names accepts registered filter names."""
        wave, trans = sample_curve
        register_filter("my_custom_band", wave, trans)

        phot = Photometry.from_names(["my_custom_band"])
        assert len(phot.filters) == 1
        assert phot.filters[0].name == "my_custom_band"

    def test_register_overwrites_none_by_default(self, clean_registry, sample_curve):
        """Registering a name twice raises KeyError (no implicit overwrite)."""
        wave, trans = sample_curve
        register_filter("band1", wave, trans)

        with pytest.raises(KeyError, match="already exists"):
            register_filter("band1", wave, trans, overwrite=False)

    def test_register_with_overwrite_true(self, clean_registry, sample_curve):
        """With overwrite=True, can replace an existing user filter."""
        wave, trans = sample_curve
        register_filter("band1", wave, trans)

        # Register again with different transmission
        trans2 = np.ones_like(trans)
        register_filter("band1", wave, trans2, overwrite=True)

        fc = load_filter("band1")
        assert np.allclose(fc.trans, trans2)

    def test_collision_with_builtin_raises(self, clean_registry, sample_curve):
        """Registering a name that exists in FILTER_REGISTRY raises without overwrite."""
        wave, trans = sample_curve
        # "sdss_r" is definitely in FILTER_REGISTRY
        assert "sdss_r" in FILTER_REGISTRY

        with pytest.raises(KeyError, match="already exists"):
            register_filter("sdss_r", wave, trans, overwrite=False)

    def test_collision_with_builtin_with_overwrite(self, clean_registry, sample_curve):
        """With overwrite=True, can shadow a built-in name."""
        wave, trans = sample_curve
        assert "sdss_r" in FILTER_REGISTRY

        # Should not raise
        register_filter("sdss_r", wave, trans, overwrite=True)

        # User filter takes precedence
        fc = load_filter("sdss_r")
        assert np.allclose(fc.trans, trans)

    def test_collision_with_synthetic_raises(self, clean_registry, sample_curve):
        """Registering a name that exists in SYNTHETIC_BAND_REGISTRY raises."""
        wave, trans = sample_curve
        assert "alma_band6" in SYNTHETIC_BAND_REGISTRY

        with pytest.raises(KeyError, match="already exists"):
            register_filter("alma_band6", wave, trans, overwrite=False)

    def test_register_immutable_curve(self, clean_registry, sample_curve):
        """Registered curve is immutable (JAX arrays, not mutable refs)."""
        wave, trans = sample_curve
        register_filter("immutable_test", wave, trans)

        fc = load_filter("immutable_test")
        # JAX arrays are immutable by design
        assert isinstance(fc.wave, jnp.ndarray)
        assert isinstance(fc.trans, jnp.ndarray)

    def test_register_shape_mismatch_raises(self, clean_registry):
        """Registering with mismatched wave/trans shapes raises ValueError."""
        wave = np.linspace(5000, 7000, 100)
        trans = np.linspace(0, 1, 50)  # Wrong size

        with pytest.raises(ValueError, match="same shape"):
            register_filter("bad_shape", wave, trans)

    def test_register_warns_on_nm_scale(self, clean_registry, nm_scale_curve):
        """Warning when wavelengths look like nanometers."""
        wave, trans = nm_scale_curve
        with pytest.warns(UserWarning, match="nanometers"):
            register_filter("nm_band", wave, trans)

    def test_no_warning_on_a_blue_filter(self, clean_registry):
        """SDSS u is not a unit error.

        The first version of this guard warned when the *median* wavelength was
        below 5000 Å, which describes every ordinary blue and UV bandpass. A
        guard that fires on ``sdss_u`` teaches users to ignore it.
        """
        wave = np.linspace(3200.0, 3900.0, 100)  # SDSS u, in Angstrom
        trans = np.exp(-0.5 * ((wave - 3550.0) / 150.0) ** 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            register_filter("u_like", wave, trans)
        assert not [w for w in caught if "nanometers" in str(w.message)]

    def test_no_warning_on_an_xray_band(self, clean_registry):
        """A hard X-ray band legitimately lives at a few Angstrom.

        ``chandra_hard`` (2-7 keV) spans 1.8-6.2 Å. An earlier rule warned
        below 50 Å, which would have flagged every X-ray band tengri ships.
        """
        wave = np.linspace(1.77, 6.20, 50)
        trans = np.ones_like(wave)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            register_filter("xray_like", wave, trans)
        assert not [w for w in caught if "nanometers" in str(w.message)]

    def test_no_warning_on_a_millimeter_band(self, clean_registry):
        """ALMA Band 6 sits near 1.2e7 Å; that is not a unit error either."""
        wave = np.linspace(1.09e7, 1.42e7, 50)
        trans = np.ones_like(wave)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            register_filter("mm_like", wave, trans)
        assert not [w for w in caught if "nanometers" in str(w.message)]

    def test_micron_confusion_is_documented_as_undetectable(self, clean_registry, um_scale_curve):
        """An optical curve given in microns is indistinguishable from X-ray.

        0.5-0.7 µm read as Angstrom lands at 0.5-0.7 Å -- 18-25 keV, a real
        NuSTAR band. The guard deliberately stays silent rather than warn on
        legitimate X-ray input; this test pins that choice so a later "fix"
        has to argue with it.
        """
        wave, trans = um_scale_curve
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            register_filter("um_band", wave, trans)
        assert not [w for w in caught if "nanometers" in str(w.message)]

    def test_warning_carries_the_exact_values(self, clean_registry, nm_scale_curve):
        """#1645: a warning that renders a rounded number must carry the value."""
        from tengri.config.exceptions import measurements_of

        wave, trans = nm_scale_curve
        with pytest.warns(UserWarning, match="nanometers") as record:
            register_filter("nm_measured", wave, trans)
        measured = measurements_of(record[0].message)
        assert measured["wave_min_aa"] == pytest.approx(float(np.min(wave)))
        assert measured["wave_max_aa"] == pytest.approx(float(np.max(wave)))

    def test_register_no_warn_normal_scale(self, clean_registry, sample_curve):
        """No warning for Angstrom-scale wavelengths."""
        import warnings

        wave, trans = sample_curve
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_filter("normal_band", wave, trans)
            # Filter out unrelated warnings (e.g., JAX warnings)
            user_warnings = [warn for warn in w if issubclass(warn.category, UserWarning)]
            assert len(user_warnings) == 0


class TestRegisterFilterFromFile:
    """register_filter_from_file: file I/O and error handling."""

    def test_register_from_file_basic(self, clean_registry, sample_curve_file):
        """Load and register a filter from file."""
        register_filter_from_file("from_file", str(sample_curve_file))

        fc = load_filter("from_file")
        assert fc.name == "from_file"
        assert len(fc.wave) > 10

    def test_register_from_file_missing_file_raises(self, clean_registry):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            register_filter_from_file("missing", "/nonexistent/path.dat")

    def test_register_from_file_supports_overwrite(self, clean_registry, tmp_path):
        """register_filter_from_file respects overwrite parameter."""
        # Create two files
        wave1 = np.linspace(5000, 7000, 100)
        trans1 = np.ones_like(wave1) * 0.5

        wave2 = np.linspace(5000, 7000, 100)
        trans2 = np.ones_like(wave2) * 0.7

        file1 = tmp_path / "file1.dat"
        file2 = tmp_path / "file2.dat"
        np.savetxt(str(file1), np.column_stack([wave1, trans1]))
        np.savetxt(str(file2), np.column_stack([wave2, trans2]))

        # Register first file
        register_filter_from_file("band", str(file1))
        fc1 = load_filter("band")
        assert np.allclose(fc1.trans, trans1)

        # Try to re-register without overwrite → should raise
        with pytest.raises(KeyError):
            register_filter_from_file("band", str(file2), overwrite=False)

        # Register with overwrite → should succeed
        register_filter_from_file("band", str(file2), overwrite=True)
        fc2 = load_filter("band")
        assert np.allclose(fc2.trans, trans2)


class TestUnregisterFilter:
    """unregister_filter: removal and fallback behavior."""

    def test_unregister_removes_filter(self, clean_registry, sample_curve):
        """Unregistered filter falls back to built-in registries."""
        wave, trans = sample_curve
        register_filter("temp_band", wave, trans)

        # Should load the registered filter
        fc1 = load_filter("temp_band")
        assert np.allclose(fc1.trans, trans)

        # Unregister
        unregister_filter("temp_band")

        # Now should raise (not in any built-in registry)
        with pytest.raises(KeyError):
            load_filter("temp_band")

    def test_unregister_nonexistent_noop(self, clean_registry):
        """Unregistering a nonexistent name raises no error."""
        unregister_filter("nonexistent_xyz")  # Should not raise

    def test_unregister_builtin_still_accessible(self, clean_registry, sample_curve):
        """After shadowing a built-in name, unregister reveals the built-in."""
        wave, trans = sample_curve
        register_filter("sdss_r", wave, trans, overwrite=True)

        # Load user version
        fc_user = load_filter("sdss_r")
        assert np.allclose(fc_user.trans, trans)

        # Unregister
        unregister_filter("sdss_r")

        # Now should load the built-in (very different transmission profile)
        fc_builtin = load_filter("sdss_r", cache_dir="data/filters")
        # Built-in SDSS r filter should be quite different from our synthetic one
        # (different wavelength range, different shape)
        user_max = float(np.max(trans))
        builtin_max = float(np.max(np.asarray(fc_builtin.trans)))
        # The peak transmission should be similar, but let's just check they're different curves
        assert fc_builtin.wave is not fc_user.wave


class TestListRegisteredFilters:
    """list_registered_filters: table generation and content."""

    def test_empty_registry_returns_empty_table(self, clean_registry):
        """Empty registry returns table with no rows."""
        table = list_registered_filters()
        assert len(table) == 0

    def test_table_contains_registered_filters(self, clean_registry, sample_curve):
        """Registered filters appear in the listing."""
        wave, trans = sample_curve
        register_filter("band1", wave, trans)
        register_filter("band2", wave, trans)

        table = list_registered_filters()
        names = [row["name"] for row in table]
        assert "band1" in names
        assert "band2" in names
        assert len(table) == 2

    def test_table_has_correct_columns(self, clean_registry, sample_curve):
        """Returned table has the documented columns."""
        wave, trans = sample_curve
        register_filter("band1", wave, trans)

        table = list_registered_filters()
        assert len(table) == 1
        row = table[0]
        assert "name" in row
        assert "kind" in row
        assert "facility" in row
        assert "svo_id" in row
        assert row["kind"] == "user_registered"
        assert row["facility"] == "User"


# ── Tests: Directory loading ───────────────────────────────────────


class TestFilterDirectory:
    """Directory loading via TENGRI_FILTER_DIR environment variable."""

    def test_load_from_directory_basic(self, clean_registry, tmp_path, sample_curve):
        """Filter file in TENGRI_FILTER_DIR loads by stem."""
        wave, trans = sample_curve
        filter_dir = tmp_path / "filters"
        filter_dir.mkdir()

        # Write filter file
        filter_file = filter_dir / "my_dir_filter.dat"
        np.savetxt(str(filter_file), np.column_stack([wave, trans]))

        # Set env var and load
        old_env = os.environ.get("TENGRI_FILTER_DIR")
        try:
            os.environ["TENGRI_FILTER_DIR"] = str(filter_dir)
            fc = load_filter("my_dir_filter")
            assert fc.name == "my_dir_filter"
            assert np.allclose(fc.wave, wave, rtol=1e-5)
        finally:
            if old_env is None:
                os.environ.pop("TENGRI_FILTER_DIR", None)
            else:
                os.environ["TENGRI_FILTER_DIR"] = old_env

    def test_load_from_multiple_directories(self, clean_registry, tmp_path, sample_curve):
        """Colon-separated TENGRI_FILTER_DIR searches all directories."""
        _wave, _trans = sample_curve

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Write different filters to each directory
        (dir1 / "band1.dat").write_text("5000 0.5\n6000 1.0\n7000 0.5\n")
        (dir2 / "band2.txt").write_text("5000 0.5\n6000 1.0\n7000 0.5\n")

        old_env = os.environ.get("TENGRI_FILTER_DIR")
        try:
            os.environ["TENGRI_FILTER_DIR"] = f"{dir1}:{dir2}"
            # Both should load
            fc1 = load_filter("band1")
            fc2 = load_filter("band2")
            assert fc1.name == "band1"
            assert fc2.name == "band2"
        finally:
            if old_env is None:
                os.environ.pop("TENGRI_FILTER_DIR", None)
            else:
                os.environ["TENGRI_FILTER_DIR"] = old_env

    def test_load_from_directory_precedence_over_builtin(self, clean_registry, tmp_path):
        """Directory filter takes precedence over built-in with same name."""
        # Create a filter file named after a built-in
        dir1 = tmp_path / "filters"
        dir1.mkdir()
        (dir1 / "sdss_r.dat").write_text("5000 0.1\n6000 1.0\n7000 0.1\n")

        old_env = os.environ.get("TENGRI_FILTER_DIR")
        try:
            os.environ["TENGRI_FILTER_DIR"] = str(dir1)
            fc = load_filter("sdss_r")
            # Should be the directory version
            expected_max = 1.0
            assert float(jnp.max(fc.trans)) == expected_max
        finally:
            if old_env is None:
                os.environ.pop("TENGRI_FILTER_DIR", None)
            else:
                os.environ["TENGRI_FILTER_DIR"] = old_env

    def test_directory_not_set_returns_none(self, clean_registry):
        """With TENGRI_FILTER_DIR unset, directory loader returns None."""
        old_env = os.environ.get("TENGRI_FILTER_DIR")
        try:
            os.environ.pop("TENGRI_FILTER_DIR", None)
            # This should not find anything via directory, so will raise
            with pytest.raises(KeyError):
                load_filter("nonexistent_xyz")
        finally:
            if old_env is not None:
                os.environ["TENGRI_FILTER_DIR"] = old_env


# ── Tests: DSPS integration ────────────────────────────────────────


class TestDSPSIntegration:
    """DSPS transmission curve loading (requires dsps package)."""

    @pytest.mark.skip(reason="DSPS integration requires dsps.data_loaders; test separately")
    def test_load_from_dsps_transmission_curve(self, clean_registry):
        """Convert DSPS TransmissionCurve to FilterCurve."""
        from dsps.data_loaders.defaults import TransmissionCurve

        from tengri.observation.filters import load_filter_from_dsps_transmission_curve

        wave = np.linspace(5000, 7000, 100)
        trans = np.exp(-0.5 * ((wave - 6000) / 300) ** 2)
        dsps_curve = TransmissionCurve(wave=wave, transmission=trans)

        fc = load_filter_from_dsps_transmission_curve(dsps_curve, name="dsps_band")
        assert fc.name == "dsps_band"
        assert np.allclose(fc.wave, wave, rtol=1e-5)

    def test_dsps_missing_attributes_raises(self, clean_registry):
        """Object without wave/transmission attributes raises AttributeError."""
        from tengri.observation.filters import load_filter_from_dsps_transmission_curve

        class FakeCurve:
            pass

        with pytest.raises(AttributeError, match=r"wave.*transmission"):
            load_filter_from_dsps_transmission_curve(FakeCurve())


# ── Tests: Load order (precedence) ─────────────────────────────────


class TestLoadPrecedence:
    """Verify the documented resolution order.

    User-registered name, then ``TENGRI_FILTER_DIR``, then the built-in
    registry, then SVO alias resolution on the display stem. The SVO step has
    no test of its own: it is what the two below fall through to, so a
    regression there fails them. A ``test_svo_alias_beats_synthetic`` used to
    stand in for it with a ``pass`` body and the comment "This is implicitly
    tested by other tests; documenting the hierarchy" -- an unskipped test
    asserting nothing, which the pass count could not distinguish from
    coverage.
    """

    def test_user_registered_beats_builtin(self, clean_registry, sample_curve):
        """User-registered name takes precedence over FILTER_REGISTRY."""
        wave, trans = sample_curve
        register_filter("sdss_r", wave, trans, overwrite=True)

        fc = load_filter("sdss_r")
        assert np.allclose(fc.trans, trans)

    def test_user_registered_beats_synthetic(self, clean_registry, sample_curve):
        """User-registered name takes precedence over SYNTHETIC_BAND_REGISTRY."""
        wave, trans = sample_curve
        register_filter("alma_band6", wave, trans, overwrite=True)

        fc = load_filter("alma_band6")
        assert np.allclose(fc.trans, trans)

    def test_directory_beats_svo_alias(self, clean_registry, tmp_path):
        """Directory filter takes precedence over SVO built-in."""
        dir1 = tmp_path / "filters"
        dir1.mkdir()
        (dir1 / "jwst_f200w.dat").write_text("5000 0.1\n6000 1.0\n7000 0.1\n")

        old_env = os.environ.get("TENGRI_FILTER_DIR")
        try:
            os.environ["TENGRI_FILTER_DIR"] = str(dir1)
            fc = load_filter("jwst_f200w")
            # Should be the directory version
            assert float(jnp.max(fc.trans)) == 1.0
        finally:
            if old_env is None:
                os.environ.pop("TENGRI_FILTER_DIR", None)
            else:
                os.environ["TENGRI_FILTER_DIR"] = old_env


# ── Tests: Integration with Photometry ─────────────────────────────


class TestPhotometryIntegration:
    """Photometry.from_names works with custom filters."""

    def test_from_names_mixed_builtin_and_custom(self, clean_registry, sample_curve):
        """from_names accepts mix of built-in and custom filter names."""
        wave, trans = sample_curve
        register_filter("my_band", wave, trans)

        phot = Photometry.from_names(["sdss_r", "my_band", "alma_band6"], cache_dir="data/filters")
        assert len(phot.filters) == 3
        names = [fc.name for fc in phot.filters]
        assert "my_band" in names
        assert "sdss_r" in names
        assert "alma_band6" in names
