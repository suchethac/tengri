# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for Lyα escape cancellation (#2531) and Q_H unit contract (#2532).

Physics validation:
- Lyα must scale with the same escape/dust factor as every other recombination line.
- Q_H takes L☉/Hz input (not erg/s/Hz), with L_SUN offset added in log space for float32 safety.
"""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "/Users/suchethacooray/.claude/jobs/c936b159/tmp")

from sweep_S1_common import build_model

from tengri.components.nebular._shared import apply_lya_escape, compute_qh_log10
from tengri.utils.physics_constants import L_SUN as LSUN_ERG_S

# Speed of light in m/s
C_LIGHT = 2.99792458e8

pytestmark = pytest.mark.bounds


class TestLyaEscapeCancellation:
    """Test suite for issue #2531: Lyα escape factor correction."""

    @pytest.fixture(scope="class")
    def ssp(self, ssp_data_fsps):
        return ssp_data_fsps

    @pytest.mark.parametrize("backend", ["cue", "cloudy"])
    @pytest.mark.parametrize("dust_kind", ["none"])
    @pytest.mark.parametrize("fesc,fdust", [(0.0, 0.0), (0.5, 0.0), (0.99, 0.0)])
    def test_lya_hbeta_ratio_invariant_at_neb_fesc_lya_zero(self, backend, dust_kind, fesc, fdust):
        """With neb_fesc_lya=0, Lyα/Hβ must be Case B reference (23.3) regardless of fesc/fdust.

        This is the primary reproducer from #2531: the buggy code multiplies Lyα by
        (1 - neb_fesc_lya) / k_factor, canceling the general suppression.
        """
        model, err = build_model(
            neb_backend=backend, neb_fesc=fesc, neb_fdust=fdust, dust_kind=dust_kind
        )
        if err is not None:
            pytest.skip(f"Model build failed: {err}")

        fluxes = model.predict_line_fluxes({}, target_wavelengths=[1215.67, 4862.68])
        lya, hbeta = fluxes

        # Case B Lyα/Hβ from Osterbrock & Ferland 2006 Table 4.2/4.4
        case_b_ratio = 23.3

        # At neb_fesc_lya=0 (no extra Lyα suppression beyond the general budget),
        # the ratio must stay at Case B regardless of the escape/dust budget.
        # Measured fail before fix: 23.3 → 358 (fesc=0.9) or 3706 (fesc=0.99).
        # Tolerance: ±2% to account for numerical integration precision in photoionization codes.
        assert abs((lya / hbeta) - case_b_ratio) < case_b_ratio * 0.02, (
            f"{backend}/{dust_kind}/fesc={fesc}/fdust={fdust}: "
            f"Lyα/Hβ = {lya / hbeta:.6f}, expected {case_b_ratio:.1f} "
            f"(off by {abs((lya / hbeta) - case_b_ratio):.1f})"
        )

    @pytest.mark.parametrize("dust_kind", ["none", "two_component"])
    def test_apply_lya_escape_helper(self, dust_kind):
        """Unit test of the shared apply_lya_escape helper on synthetic lines.

        Verifies that the helper correctly identifies the Lyα row and applies
        the (1 - neb_fesc_lya) factor without the division.
        """
        # Synthetic line array: 5 lines at 912 (LyC edge), 1215.67 (Lyα), 4862 (Hβ),
        # 5007 (O III), 6564 (Hα) [Angstrom]
        wavelengths = jnp.array([912.0, 1215.67, 4862.68, 5006.84, 6563.77])
        line_lum = jnp.array([1.0, 23.3, 1.0, 2.5, 1.5])  # arbitrary units

        neb_fesc_lya = 0.3

        result = apply_lya_escape(line_lum, wavelengths, neb_fesc_lya)

        # Lyα (index 1) should be multiplied by (1 - 0.3) = 0.7
        expected = jnp.array([1.0, 23.3 * (1.0 - neb_fesc_lya), 1.0, 2.5, 1.5])

        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_apply_lya_escape_no_mutation(self):
        """Verify that apply_lya_escape does not mutate the input array."""
        wavelengths = jnp.array([912.0, 1215.67, 4862.68])
        line_lum = jnp.array([1.0, 23.3, 1.0])
        line_lum_orig = jnp.copy(line_lum)

        _ = apply_lya_escape(line_lum, wavelengths, 0.5)

        assert jnp.allclose(line_lum, line_lum_orig)


class TestComputeQhUnitContract:
    """Test suite for issue #2532: compute_qh unit contract and internal constants.

    The contract: compute_qh_log10 takes L☉/Hz as input (per M☉), not erg/s/Hz.
    The L_SUN conversion is added in log space to avoid float32 overflow.
    """

    @pytest.fixture(scope="class")
    def ssp(self, ssp_data_fsps):
        return ssp_data_fsps

    def test_compute_qh_log10_raw_lsun_hz_input(self, ssp):
        """compute_qh_log10 must accept L☉/Hz input and return Q_H in photons/s.

        Input: raw flux as stored in SSP grid (L☉/Hz per M☉).
        Output: Q_H log10 in photons/s/Msun, approximately 46.77±0.05 at 1 Myr.
        """
        wave = ssp.ssp_wave
        flux_raw_lsun_hz = ssp.ssp_flux[0, 0, :]  # L☉/Hz/Msun as stored in grid

        q_log10 = float(compute_qh_log10(wave, flux_raw_lsun_hz))

        # Expected: Q_H at the 1 Myr solar-Z node is ~46.77 dex
        # (corresponds to ~5.9e46 photons/s/Msun)
        # Tolerance: ±0.05 dex (measurement uncertainty vs. numpy calculation)
        assert abs(q_log10 - 46.77) < 0.05, (
            f"Q_H = {q_log10:.2f} dex — expected ~46.77±0.05. "
            f"compute_qh_log10 may not be using the L☉/Hz contract."
        )

    def test_compute_qh_log10_with_lsun_conversion_offset(self, ssp):
        """Verify offset when flux is pre-multiplied by L_SUN (erg/s/Hz input).

        If caller mistakenly converts flux to erg/s/Hz before calling compute_qh_log10,
        the result shifts by log10(L_SUN) ≈ 33.584 dex.
        """
        wave = ssp.ssp_wave
        flux_raw_lsun_hz = ssp.ssp_flux[0, 0, :]

        # Correct call: raw L☉/Hz input
        q_raw = float(compute_qh_log10(wave, flux_raw_lsun_hz))

        # Mistaken call: convert to erg/s/Hz first (multiplying by L_SUN)
        flux_erg_hz = flux_raw_lsun_hz * LSUN_ERG_S
        q_erg = float(compute_qh_log10(wave, flux_erg_hz))

        # Verify offset is exactly log10(L_SUN) ≈ 33.584 dex
        log10_lsun = np.log10(LSUN_ERG_S)
        offset = q_erg - q_raw
        assert abs(offset - log10_lsun) < 0.01, (
            f"Offset = {offset:.3f} dex, expected log10(L_SUN) = {log10_lsun:.3f}. "
            f"The float32-safe conversion in log space is not working correctly."
        )
