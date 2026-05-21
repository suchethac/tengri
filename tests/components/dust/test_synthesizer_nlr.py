"""Unit tests for SynthesizerNLRBackend.

Tests grid loading, interpolation, JIT compatibility, and correct
normalization from Synthesizer's per-bolometric convention to
per-ionizing-photon.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.bounds

from pathlib import Path

import jax
import jax.numpy as jnp

from tengri.components.nebular.agn_nebular import SynthesizerNLRBackend, _load_synthesizer_nlr_grid

# Test data path
_TEST_GRID_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "synthesizer_grids" / "test_grid_agn-nlr.hdf5"
)


@pytest.fixture(scope="module")
def grid_path():
    """Return path to test grid, skip if missing."""
    if not _TEST_GRID_PATH.exists():
        pytest.skip(f"Test grid not found at {_TEST_GRID_PATH}")
    return _TEST_GRID_PATH


@pytest.fixture(scope="module")
def synthesizer_backend(grid_path):
    """Initialize SynthesizerNLRBackend once per test session."""
    return SynthesizerNLRBackend(grid_path)


class TestGridLoading:
    """Test grid loading and data structure."""

    def test_grid_loads_without_error(self, grid_path):
        """Grid loads successfully from HDF5."""
        grid = _load_synthesizer_nlr_grid(grid_path)
        assert grid.log_line_per_qh.shape[-1] == 215
        assert grid.line_wavelengths_aa.shape == (215,)

    def test_grid_axes_structure(self, grid_path):
        """Grid axes have expected lengths and are finite."""
        grid = _load_synthesizer_nlr_grid(grid_path)
        assert len(grid.mass_axis) == 2
        assert len(grid.eddington_axis) == 2
        assert len(grid.cosine_axis) == 2
        assert len(grid.metallicity_axis) == 2
        assert len(grid.logU_axis) == 2
        assert len(grid.logn_axis) == 2
        assert jnp.all(jnp.isfinite(grid.mass_axis))
        assert jnp.all(jnp.isfinite(grid.eddington_axis))
        assert jnp.all(jnp.isfinite(grid.cosine_axis))

    def test_grid_log_luminosities_finite(self, grid_path):
        """Grid log_line_per_qh contains finite values (not NaN/Inf)."""
        grid = _load_synthesizer_nlr_grid(grid_path)
        assert jnp.all(jnp.isfinite(grid.log_line_per_qh))

    def test_missing_grid_raises_error(self):
        """FileNotFoundError raised if grid file does not exist."""
        with pytest.raises(FileNotFoundError):
            _load_synthesizer_nlr_grid("/nonexistent/path/grid.hdf5")


class TestBackendInitialization:
    """Test SynthesizerNLRBackend initialization and attributes."""

    def test_backend_initializes(self, grid_path):
        """Backend initializes without error."""
        backend = SynthesizerNLRBackend(grid_path)
        assert backend.name == "synthesizer_nlr"
        assert backend.has_free_params is True
        assert backend.has_continuum is False

    def test_backend_precomputes_edges(self, synthesizer_backend):
        """Edges precomputed for all 6 axes."""
        assert hasattr(synthesizer_backend, "_edges_mass")
        assert hasattr(synthesizer_backend, "_edges_edd")
        assert hasattr(synthesizer_backend, "_edges_cos")
        assert hasattr(synthesizer_backend, "_edges_met")
        assert hasattr(synthesizer_backend, "_edges_ionU")
        assert hasattr(synthesizer_backend, "_edges_nH")


class TestPrediction:
    """Test the predict_agn_nlr_lines method."""

    def test_predict_returns_expected_shapes(self, synthesizer_backend):
        """predict_agn_nlr_lines returns (215,) arrays for wavelengths and luminosities."""
        wave, lum = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=38.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        assert wave.shape == (215,)
        assert lum.shape == (215,)

    def test_predict_luminosities_finite(self, synthesizer_backend):
        """Predicted luminosities are all finite (no NaN/Inf)."""
        _, lum = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=38.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        assert jnp.all(jnp.isfinite(lum))

    def test_predict_luminosities_nonnegative(self, synthesizer_backend):
        """Predicted luminosities are non-negative."""
        _, lum = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=38.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        assert jnp.all(lum >= 0.0)

    def test_predict_respects_escape_fraction(self, synthesizer_backend):
        """Escape fraction reduces luminosities."""
        wave1, lum1 = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=38.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
            neb_fesc=0.0,
        )
        wave2, lum2 = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=38.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
            neb_fesc=0.5,
        )
        assert jnp.allclose(wave1, wave2)
        assert jnp.all(lum2 <= lum1)  # Escape fraction reduces luminosities

    def test_predict_varies_with_parameters(self, synthesizer_backend):
        """Changing grid parameters changes luminosities (except wavelengths)."""
        wave1, lum1 = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=38.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        wave2, lum2 = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=39.3,  # Change log_bh_mass (grid covers 38.3-39.3)
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        # Wavelengths should be the same (grid-independent)
        assert jnp.allclose(wave1, wave2)
        # Luminosities should differ (grid actually has different values)
        assert not jnp.allclose(lum1, lum2, atol=1e-6)


class TestJitCompatibility:
    """Test JAX JIT compilation compatibility."""

    def test_predict_is_jittable(self, synthesizer_backend):
        """The predict_agn_nlr_lines method can be JIT-wrapped."""

        @jax.jit
        def jitted_predict(
            log_bh_mass, log_eddington, cosine_inc, log_met, log_ionU, log_nH, log_qh
        ):
            return synthesizer_backend.predict_agn_nlr_lines(
                log_bh_mass=log_bh_mass,
                log_eddington=log_eddington,
                cosine_inclination=cosine_inc,
                log_metallicity=log_met,
                log_ionU=log_ionU,
                log_nH=log_nH,
                log_qh=log_qh,
            )

        wave, lum = jitted_predict(38.3, -0.3, 0.2, 0.0, -1.5, 4.0, 53.0)
        assert wave.shape == (215,)
        assert lum.shape == (215,)
        assert jnp.all(jnp.isfinite(lum))

    def test_predict_jit_consistent(self, synthesizer_backend):
        """JIT-compiled and non-JIT predictions match."""

        @jax.jit
        def jitted_predict(
            log_bh_mass, log_eddington, cosine_inc, log_met, log_ionU, log_nH, log_qh
        ):
            return synthesizer_backend.predict_agn_nlr_lines(
                log_bh_mass=log_bh_mass,
                log_eddington=log_eddington,
                cosine_inclination=cosine_inc,
                log_metallicity=log_met,
                log_ionU=log_ionU,
                log_nH=log_nH,
                log_qh=log_qh,
            )

        # Non-JIT
        wave_eager, lum_eager = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=38.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        # JIT
        wave_jit, lum_jit = jitted_predict(38.3, -0.3, 0.2, 0.0, -1.5, 4.0, 53.0)
        assert jnp.allclose(wave_eager, wave_jit)
        assert jnp.allclose(lum_eager, lum_jit)
