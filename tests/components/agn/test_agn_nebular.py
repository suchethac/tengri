# SPDX-License-Identifier: BSD-3-Clause
"""Tests for AGN NLR emission with multiple backends.

Tests the disc -> Cue -> NLR pipeline (Chain 2) and the unified
AGN NLR dispatcher.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
from pathlib import Path

import jax.numpy as jnp

from tengri.components.nebular.agn_nebular import (
    _log_qh_from_lacc,
    agn_ionspec_from_alpha_pl,
    agn_nlr_cue,
    agn_nlr_emission,
)
from tengri.components.nebular.ionizing_spectrum import _CLIP_RANGES

# ── Fixtures and skip conditions ──────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_CUE_WEIGHTS_PATH = _DATA_DIR / "cue_weights.npz"
CUE_AVAILABLE = _CUE_WEIGHTS_PATH.exists()
requires_cue = pytest.mark.skipif(
    not CUE_AVAILABLE,
    reason="Cue weights not found at data/cue_weights.npz",
)


@pytest.fixture(scope="module")
def cue_backend():
    """Load CueBackend if weights are available."""
    if not CUE_AVAILABLE:
        pytest.skip("Cue weights not found")
    from tengri.components.nebular.cue import CueBackend

    return CueBackend(str(_CUE_WEIGHTS_PATH))


@pytest.fixture
def wavelength():
    """Standard rest-frame wavelength grid."""
    return jnp.linspace(3000.0, 10000.0, 1000)


# ── Tests: agn_ionspec_from_alpha_pl ──────────────────────────────
class TestIonspecFromAlphaPl:
    """Tests for converting AGN power-law slope to Cue ionspec params."""

    def test_single_slope_all_indices_equal(self):
        """All 4 segment indices should be equal for a pure power law."""
        result = agn_ionspec_from_alpha_pl(-1.7)
        idx1 = float(result["ionspec_index1"])
        idx2 = float(result["ionspec_index2"])
        idx3 = float(result["ionspec_index3"])
        idx4 = float(result["ionspec_index4"])
        # Before clipping they would be identical (1.7 for all).
        # After clipping, index1 has range [1.0, 42.0] so 1.7 is fine.
        assert idx1 == pytest.approx(idx2, abs=0.01)
        assert idx2 == pytest.approx(idx3, abs=0.01)
        assert idx3 == pytest.approx(idx4, abs=0.01)

    def test_typical_agn_within_cue_ranges(self):
        """alpha_pl=-1.7 should give indices within valid Cue ranges."""
        result = agn_ionspec_from_alpha_pl(-1.7)
        for key, (lo, hi) in _CLIP_RANGES.items():
            val = float(result[key])
            assert lo <= val <= hi, f"{key}={val} outside [{lo}, {hi}]"

    def test_steeper_slope_gives_larger_indices(self):
        """Steeper EUV slope (more negative alpha) -> larger wavelength index."""
        result_17 = agn_ionspec_from_alpha_pl(-1.7)
        result_20 = agn_ionspec_from_alpha_pl(-2.0)
        # wavelength_slope = -alpha_pl, so -2.0 -> 2.0 > 1.7
        # After clipping both should be within range, and idx4 has range
        # [-1.7, 8.0] so both 1.7 and 2.0 fit.
        assert float(result_20["ionspec_index4"]) > float(result_17["ionspec_index4"])

    def test_extreme_alpha_clipped(self):
        """Extreme alpha_pl values should be clipped to valid ranges."""
        # Very steep: alpha_pl = -50 -> wavelength_slope = 50
        result = agn_ionspec_from_alpha_pl(-50.0)
        assert float(result["ionspec_index1"]) <= _CLIP_RANGES["ionspec_index1"][1]
        assert float(result["ionspec_index4"]) <= _CLIP_RANGES["ionspec_index4"][1]
        # Very flat: alpha_pl = 5 -> wavelength_slope = -5
        result_flat = agn_ionspec_from_alpha_pl(5.0)
        assert float(result_flat["ionspec_index4"]) >= _CLIP_RANGES["ionspec_index4"][0]

    def test_returns_all_seven_keys(self):
        """Result should have exactly the 7 expected keys."""
        result = agn_ionspec_from_alpha_pl(-1.7)
        expected_keys = {
            "ionspec_index1",
            "ionspec_index2",
            "ionspec_index3",
            "ionspec_index4",
            "ionspec_logLratio1",
            "ionspec_logLratio2",
            "ionspec_logLratio3",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_finite(self):
        """All returned values should be finite."""
        for alpha in [-0.5, -1.0, -1.5, -1.7, -2.0, -2.5]:
            result = agn_ionspec_from_alpha_pl(alpha)
            for key, val in result.items():
                assert jnp.isfinite(val), f"{key} not finite for alpha={alpha}"


# ── Tests: _log_qh_from_lacc ──────────────────────────────────────
class TestLogQH:
    """Tests for ionizing photon rate computation."""

    def test_typical_agn_qh(self):
        """L_acc=1e44 erg/s should give log(Q_H) ~ 53-56."""
        log_qh = float(_log_qh_from_lacc(1e44, -1.7))
        assert 50.0 < log_qh < 60.0

    def test_higher_lacc_higher_qh(self):
        """More luminous AGN should produce more ionizing photons."""
        qh_low = float(_log_qh_from_lacc(1e43, -1.7))
        qh_high = float(_log_qh_from_lacc(1e45, -1.7))
        assert qh_high > qh_low

    def test_finite_for_various_slopes(self):
        """Q_H should be finite for a range of slopes."""
        for alpha in [-0.5, -1.0, -1.5, -1.7, -2.0, -3.0]:
            log_qh = _log_qh_from_lacc(1e44, alpha)
            assert jnp.isfinite(log_qh), f"Q_H not finite for alpha={alpha}"


# ── Tests: agn_nlr_cue (requires Cue weights) ─────────────────────
class TestAgnNlrCue:
    """Tests for Cue-based AGN NLR emission."""

    @requires_cue
    def test_returns_lines(self, cue_backend):
        """Should return wavelength and luminosity arrays."""
        wav, lum = agn_nlr_cue(cue_backend, l_acc_erg=1e44)
        assert len(wav) > 0
        assert len(lum) == len(wav)

    @requires_cue
    def test_nonzero_luminosity(self, cue_backend):
        """Line luminosities should be positive and nonzero."""
        _wav, lum = agn_nlr_cue(cue_backend, l_acc_erg=1e44)
        chex.assert_tree_all_finite(lum)
        assert jnp.any(lum > 0)

    @requires_cue
    def test_has_oiii_5007(self, cue_backend):
        """[OIII] 5007 should be present in the output lines."""
        wav, lum = agn_nlr_cue(cue_backend, l_acc_erg=1e44)
        # Check for a line near 5007 A (within 2 A tolerance)
        oiii_mask = jnp.abs(wav - 5007.0) < 2.0
        assert jnp.any(oiii_mask), "[OIII] 5007 not found in line list"
        oiii_lum = lum[oiii_mask]
        assert jnp.all(oiii_lum > 0), "[OIII] 5007 has zero luminosity"

    @requires_cue
    def test_covering_fraction_scales(self, cue_backend):
        """Doubling covering fraction should double luminosity."""
        _, lum_01 = agn_nlr_cue(
            cue_backend,
            l_acc_erg=1e44,
            covering_fraction=0.1,
        )
        _, lum_02 = agn_nlr_cue(
            cue_backend,
            l_acc_erg=1e44,
            covering_fraction=0.2,
        )
        ratio = jnp.sum(lum_02) / jnp.maximum(jnp.sum(lum_01), 1e-30)
        assert ratio == pytest.approx(2.0, rel=1e-4)

    @requires_cue
    def test_higher_lacc_brighter(self, cue_backend):
        """Higher L_acc should produce brighter NLR emission."""
        _, lum_low = agn_nlr_cue(cue_backend, l_acc_erg=1e43)
        _, lum_high = agn_nlr_cue(cue_backend, l_acc_erg=1e45)
        assert jnp.sum(lum_high) > jnp.sum(lum_low)

    @requires_cue
    def test_higher_logu_changes_ratios(self, cue_backend):
        """Different ionization parameter should change line ratios."""
        _wav, lum_low_u = agn_nlr_cue(
            cue_backend,
            l_acc_erg=1e44,
            neb_logU=-3.5,
        )
        _, lum_high_u = agn_nlr_cue(
            cue_backend,
            l_acc_erg=1e44,
            neb_logU=-2.0,
        )
        # Normalize both to total luminosity to compare ratios
        norm_low = lum_low_u / jnp.maximum(jnp.sum(lum_low_u), 1e-30)
        norm_high = lum_high_u / jnp.maximum(jnp.sum(lum_high_u), 1e-30)
        # Ratios should differ — not identical spectra
        assert not jnp.allclose(norm_low, norm_high, atol=1e-3)


# ── Tests: agn_nlr_emission dispatcher ────────────────────────────
# ── Tests: agn_nlr_emission dispatcher ────────────────────────────
class TestDispatcher:
    """Tests for the unified AGN NLR dispatcher."""

    @requires_cue
    def test_cue_backend(self, cue_backend):
        """backend='cue' should return (wavelengths, luminosities) tuple."""
        result = agn_nlr_emission(
            backend="cue",
            cue_backend=cue_backend,
            l_acc_erg=1e44,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        wav, lum = result
        assert len(wav) > 0
        assert len(lum) == len(wav)

    def test_cue_without_backend_raises(self, wavelength):
        """backend='cue' without cue_backend should raise ValueError."""
        with pytest.raises(ValueError, match="cue_backend must be provided"):
            agn_nlr_emission(backend="cue", l_acc_erg=1e44)

    def test_feltre_without_backend_raises(self, wavelength):
        """backend='feltre' without feltre_backend should raise ValueError."""
        with pytest.raises(ValueError, match="feltre_backend must be provided"):
            agn_nlr_emission(backend="feltre", l_acc_erg=1e44)

    def test_invalid_backend_raises(self, wavelength):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown AGN NLR backend"):
            agn_nlr_emission(backend="invalid", l_acc_erg=1e44)
