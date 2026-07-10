# SPDX-License-Identifier: BSD-3-Clause
"""Tests for IMF normalization and SPS amplitudes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.bounds


def _make_ssp_if_available():
    from pathlib import Path

    import h5py

    from tengri import load_ssp_data

    p = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    if not p.exists():
        return None
    # Line amplitudes and the SSP metallicity-range check need the real grid;
    # the synthetic #613 CI fixture has an arbitrary flux scale and lgmet axis.
    with h5py.File(p, "r") as f:
        if f.attrs.get("synthetic", False):
            return None
    return load_ssp_data(str(p))


LSUN_ERG = 3.828e33  # erg/s


class TestEmissionLineAmplitudes:
    """Case B recombination: for SFR = 1 M⊙/yr (Chabrier IMF, solar Z),
    Q_H ≈ 4.2e53 photons/s (Leitherer+1999) and
    L_Hα ≈ 1.26e41 erg/s, L_Hβ = L_Hα / 2.86 ≈ 4.4e40 erg/s ≈ 1.15e7 L_sun.

    Here we pin predict_hbeta for a SFR=1 M⊙/yr 1 Gyr-old constant SFH.
    """

    def _build_const_sfh_model(self, log_sfr=0.0):
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")
        # Constant SFH: total mass = SFR x duration. Pre-#369 the fixture
        # set an SFR amplitude directly; the log_total_mass rename kept the
        # bare log_sfr value, silently building a 10^log_sfr M_sun galaxy
        # (9 dex low at SFR=1, #1013).
        duration_yr = (1.0 - 1e-3) * 1e9
        spec = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(log_sfr + float(np.log10(duration_yr))),
            sfh_const_start_gyr=Fixed(1.0),
            sfh_const_end_gyr=Fixed(1e-3),
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.0),
        )
        return SEDModel(spec, ssp)

    def test_hbeta_amplitude_for_sfr1(self):
        """Constant SFR=1 M⊙/yr ⇒ L_Hβ should be ≈1e7 L_sun (Leitherer+1999, Case B)."""
        model = self._build_const_sfh_model(log_sfr=0.0)
        L_hb = float(model.predict_hbeta({}))
        assert 1e6 < L_hb < 1e8, f"L_Hβ = {L_hb:.2e} L_sun (expected ≈1e7)"

    def test_hbeta_linear_in_sfr(self):
        """predict_hbeta must scale linearly with SFR."""
        m1 = self._build_const_sfh_model(log_sfr=0.0)
        m10 = self._build_const_sfh_model(log_sfr=1.0)
        r = float(m10.predict_hbeta({})) / float(m1.predict_hbeta({}))
        assert 9.0 < r < 11.0, f"10×SFR → {r:.2f}×L_Hβ"


class TestSSPMetallicity:
    """SSP metallicity grid ensures canonical Zsun offset."""

    def test_ssp_log_metallicity_range(self, real_ssp_only):
        """Standard SSP grids cover ~[-2.3, +0.3] in log10(Z/Z_sun).

        Asserts the real MILES grid's absolute lgmet range, so it needs the real
        grid — the synthetic #613 fixture has its own (different) lgmet axis.

        tengri stores ``ssp_lgmet`` in ABSOLUTE log10(Z), with LOG10_ZSUN = -1.848
        (MILES convention). So ssp_lgmet range should be ~[-4.1, -1.55].
        """

        ssp_path = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
        if not ssp_path.exists():
            pytest.skip("SSP file not available")
        ssp = load_ssp_data(str(ssp_path))
        lgmet = np.array(ssp.ssp_lgmet)
        assert -4.5 < lgmet.min() < -3.0
        assert -2.0 < lgmet.max() < -1.0
        # Convert to Z/Zsun (LOG10_ZSUN = -1.848)
        log_zsol = lgmet + 1.848
        assert log_zsol.min() < -1.5 and log_zsol.max() > 0.0
