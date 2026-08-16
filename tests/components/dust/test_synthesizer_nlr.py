# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for SynthesizerNLRBackend.

Tests grid loading, interpolation, JIT compatibility, and correct
normalization from Synthesizer's per-bolometric convention to
per-ionizing-photon.
"""

from __future__ import annotations

import chex
import pytest

pytestmark = pytest.mark.bounds

from pathlib import Path

import jax
import jax.numpy as jnp

from tengri.components.nebular.agn_nebular import SynthesizerNLRBackend, _load_synthesizer_nlr_grid

# Test data path
_TEST_GRID_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "synthesizer_grids" / "test_grid_agn-nlr.hdf5"
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
        chex.assert_shape(grid.line_wavelengths_aa, (215,))

    def test_grid_axes_structure(self, grid_path):
        """Grid axes have expected lengths and are finite."""
        grid = _load_synthesizer_nlr_grid(grid_path)
        assert len(grid.mass_axis) == 2
        assert len(grid.eddington_axis) == 2
        assert len(grid.cosine_axis) == 2
        assert len(grid.metallicity_axis) == 2
        assert len(grid.logU_axis) == 2
        assert len(grid.logn_axis) == 2
        chex.assert_tree_all_finite(grid.mass_axis)
        chex.assert_tree_all_finite(grid.eddington_axis)
        chex.assert_tree_all_finite(grid.cosine_axis)

    def test_grid_log_luminosities_finite(self, grid_path):
        """Grid log_line_per_qh contains finite values (not NaN/Inf)."""
        grid = _load_synthesizer_nlr_grid(grid_path)
        chex.assert_tree_all_finite(grid.log_line_per_qh)

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
            log_bh_mass=8.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        chex.assert_shape(wave, (215,))
        chex.assert_shape(lum, (215,))

    def test_predict_luminosities_finite(self, synthesizer_backend):
        """Predicted luminosities are all finite (no NaN/Inf)."""
        _, lum = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=8.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        chex.assert_tree_all_finite(lum)

    def test_predict_luminosities_nonnegative(self, synthesizer_backend):
        """Predicted luminosities are non-negative."""
        _, lum = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=8.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        assert_non_negative(lum, name="lum")

    def test_predict_respects_escape_fraction(self, synthesizer_backend):
        """Escape fraction reduces luminosities."""
        wave1, lum1 = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=8.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
            neb_fesc=0.0,
        )
        wave2, lum2 = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=8.3,
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
            log_bh_mass=8.0,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        wave2, lum2 = synthesizer_backend.predict_agn_nlr_lines(
            log_bh_mass=9.0,  # Change log_bh_mass [log10(M_sun)] (grid covers 8-9)
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

        wave, lum = jitted_predict(8.3, -0.3, 0.2, 0.0, -1.5, 4.0, 53.0)
        chex.assert_shape(wave, (215,))
        chex.assert_shape(lum, (215,))
        chex.assert_tree_all_finite(lum)

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
            log_bh_mass=8.3,
            log_eddington=-0.3,
            cosine_inclination=0.2,
            log_metallicity=0.0,
            log_ionU=-1.5,
            log_nH=4.0,
            log_qh=53.0,
        )
        # JIT
        wave_jit, lum_jit = jitted_predict(8.3, -0.3, 0.2, 0.0, -1.5, 4.0, 53.0)
        assert jnp.allclose(wave_eager, wave_jit)
        assert jnp.allclose(lum_eager, lum_jit)


# Reprocessed nebular spectrum — UnifiedAGN parity path (issue #694)

import os

from tests._bounds import assert_non_negative

_SYN_LIB_DIR = Path(
    os.environ.get(
        "SYNTHESIZER_GRID_DIR",
        os.path.expanduser("~/Library/Application Support/Synthesizer/grids"),
    )
)


@pytest.fixture(scope="module")
def nebular_grid_path():
    """Path to an NLR grid carrying /spectra/nebular; skip if unavailable.

    Falls back to the Synthesizer install location so the parity assertions run
    locally even when the repo ``data/`` symlink is absent (CI skips either way).
    """
    for cand in (_TEST_GRID_PATH, _SYN_LIB_DIR / "test_grid_agn-nlr.hdf5"):
        try:
            if cand.exists():
                return cand
        except OSError:  # broken symlink loop
            continue
    pytest.skip("No Synthesizer NLR grid with /spectra found")


class TestNebularSpectrumUnifiedAGNParity:
    """The /spectra/nebular reprocessed path reproduces Synthesizer's UnifiedAGN.

    Reading ``/spectra/nebular`` (the array UnifiedAGN extracts) instead of
    re-broadening the scrambled ``/lines`` table fixes the [O III]-vs-Lyα
    inversion documented in issue #694.
    """

    def test_grid_loads_nebular_spectrum(self, nebular_grid_path):
        """Loader populates the reprocessed nebular spectrum + its wavelength axis."""
        grid = _load_synthesizer_nlr_grid(nebular_grid_path)
        assert grid.nebular_per_lbol is not None
        assert grid.spectra_wavelengths_aa is not None
        # leading 6 parameter axes + a trailing wavelength axis
        assert grid.nebular_per_lbol.ndim == 7
        assert grid.nebular_per_lbol.shape[-1] == grid.spectra_wavelengths_aa.shape[0]

    def test_reprocessed_spectrum_is_finite_nonnegative(self, nebular_grid_path):
        """Output L_nu is finite, non-negative, and on the requested grid."""
        backend = SynthesizerNLRBackend(nebular_grid_path)
        wave = jnp.linspace(1000.0, 10000.0, 2000)
        l_nu = backend.predict_agn_nebular_spectrum(
            wave, l_bol_erg=1e46, covering_fraction=0.1, log_eddington=0.0, log_metallicity=-2.0
        )
        chex.assert_shape(l_nu, (2000,))
        assert jnp.all(jnp.isfinite(l_nu))
        assert_non_negative(l_nu, name="l_nu")

    def test_oiii_dominates_over_lya(self, nebular_grid_path):
        """The reprocessed NLR is [O III] 5007-dominant, not Lyα-dominant.

        This is the qualitative signature of UnifiedAGN parity: the broken
        ``/lines`` path inverted these (issue #694).
        """
        backend = SynthesizerNLRBackend(nebular_grid_path)
        wave = jnp.linspace(1000.0, 10000.0, 6000)
        l_nu = backend.predict_agn_nebular_spectrum(
            wave, l_bol_erg=1e46, covering_fraction=0.1, log_eddington=0.0, log_metallicity=-2.0
        )

        def peak(w0, half=15.0):
            m = (wave > w0 - half) & (wave < w0 + half)
            return jnp.max(jnp.where(m, l_nu, 0.0))

        assert float(peak(5006.84)) > float(peak(1215.67))
        # and [O III] should tower over Hβ (strong forbidden AGN line)
        assert float(peak(5006.84)) > 3.0 * float(peak(4861.33))

    def test_nebular_spectrum_jit(self, nebular_grid_path):
        """predict_agn_nebular_spectrum is JIT-compatible."""
        backend = SynthesizerNLRBackend(nebular_grid_path)
        wave = jnp.linspace(1000.0, 10000.0, 512)

        @jax.jit
        def predict(lbol):
            return backend.predict_agn_nebular_spectrum(
                wave, l_bol_erg=lbol, covering_fraction=0.1
            )

        eager = backend.predict_agn_nebular_spectrum(wave, l_bol_erg=1e46, covering_fraction=0.1)
        assert jnp.allclose(eager, predict(1e46))
