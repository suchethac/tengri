# SPDX-License-Identifier: BSD-3-Clause
"""Tests for shock-line atomic physics and Balmer ratios via MAPPINGS V.

These pin atomic transition ratios (Case B Balmer) and density-sensitive diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

# One assignment, not two: the second rebinds the name and the first is lost.
pytestmark = [pytest.mark.bounds, pytest.mark.conservation]

_MAPPINGS_H5 = (
    __import__("pathlib").Path(__file__).resolve().parents[3] / "data" / "mappings_templates.h5"
)
_h5_only_shock = pytest.mark.skipif(
    not _MAPPINGS_H5.exists(),
    reason="data/mappings_templates.h5 not found; build via download_mappings_templates.py",
)


class TestShockLineRatios:
    """MAPPINGS V shock models (via tengri.nebular.shock_line_ratios).

    These pin atomic-physics ratios (temperature/density insensitive) and
    Case-B Balmer ratio, which should be independent of code version.
    """

    def _r(self, v=300.0, log_n=2.0, b=1.0):
        from tengri.nebular import shock_line_ratios

        return shock_line_ratios(shock_velocity=v, shock_log_density=log_n, shock_b_over_sqrt_n=b)

    def test_oiii_5007_4959_ratio_is_atomic(self):
        """[O III] λ5007/λ4959 = 2.98 (Storey & Zeippen 2000, atomic transition)."""
        r = self._r()
        ratio = r["O3_5007A"] / r["O3_4959A"]
        assert 2.7 < ratio < 3.1, f"[OIII] 5007/4959 = {ratio:.3f}, expected 2.98"

    def test_nii_6583_6548_ratio_is_atomic(self):
        """[N II] λ6583/λ6548 = 2.96 (atomic transition, quasi-constant)."""
        r = self._r()
        ratio = r["NII_6583A"] / r["NII_6548A"]
        assert 2.7 < ratio < 3.1, f"[NII] 6583/6548 = {ratio:.3f}, expected 2.96"

    def test_halpha_hbeta_case_B_like(self):
        """Shocks at v=200 km/s give Hα/Hβ close to Case B 2.86 (T_e ≈ 1e4 K).

        Fast shocks elevate the ratio slightly; range ~2.7–3.3 covers
        the shocked-gas literature at v = 100–500 km/s.
        """
        r = self._r(v=200.0)
        balmer = r["HA_6563A"] / r["Hb_4861A"]
        assert 2.5 < balmer < 3.3, f"Hα/Hβ = {balmer:.3f}"

    @_h5_only_shock
    def test_sii_doublet_is_density_sensitive(self):
        """[SII] 6716/6731 increases from ~0.45 (high n) to ~1.45 (low n)."""
        r_low = self._r(log_n=0.0)  # 1 cm^-3
        r_hi = self._r(log_n=3.0)  # 1000 cm^-3
        sii_low = r_low["SII_6716A"] / r_low["SII_6731A"]
        sii_hi = r_hi["SII_6716A"] / r_hi["SII_6731A"]
        assert sii_hi < sii_low, (
            f"Denser gas should have smaller [SII] ratio (low_n={sii_low:.2f}, hi_n={sii_hi:.2f})"
        )
        # Bounded by atomic-physics limits (Osterbrock & Ferland AGN² Table 5.8)
        assert 0.4 < sii_hi < 1.5
        assert 0.4 < sii_low < 1.5

    def test_bpt_agn_branch_for_fast_shocks(self):
        """v_s ≥ 300 km/s lands above the Kewley+2001 AGN/SF boundary line."""

        r = self._r(v=400.0, log_n=2.0)
        log_oiii_hb = np.log10(r["O3_5007A"] / r["Hb_4861A"])
        log_nii_ha = np.log10(r["NII_6583A"] / r["HA_6563A"])
        # Kewley+2001 max-SB line: y = 0.61/(x - 0.47) + 1.19  (for log NII/Hα)
        y_kewley = 0.61 / (log_nii_ha - 0.47) + 1.19
        # Fast shocks should be above (larger [OIII]/Hβ than max-SB line)
        assert log_oiii_hb > y_kewley, (
            f"Fast shock should lie in AGN/LINER region. "
            f"log[OIII]/Hβ={log_oiii_hb:.2f}, Kewley y={y_kewley:.2f}"
        )
