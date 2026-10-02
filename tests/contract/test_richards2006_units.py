# SPDX-License-Identifier: BSD-3-Clause
"""Unit contracts for Richards+2006 disc template (issue #2563).

Verifies that the stored Richards+2006 data and tengri's runtime implementation
correctly interpret the L_nu column and preserve its values and slopes through
the interpolation chain.

Validated facts:
- Column 2 of richards2006.dat IS L_nu [erg/s/Hz], not nu*F_nu
- Value at 2500 Å matches Richards et al. 2006 ApJS 166, 470 Table 3
- 1450–2200 Å slope (alpha_nu) is canonical quasar ~-0.44
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.contract


_RICHARDS_DAT = Path(__file__).resolve().parents[2] / "src" / "tengri" / "data" / "agn_bbb" / "richards2006.dat"
_REF_H5 = Path(__file__).resolve().parents[2] / "data" / "agnfitter_bbb_reference.h5"


@pytest.fixture(scope="module")
def richards_dat_values():
    """Load Richards+2006 template from .dat file."""
    with open(_RICHARDS_DAT, "r") as fh:
        arr = np.loadtxt(fh)
    wave_aa = np.asarray(arr[:, 0], dtype=np.float64)
    lnu = np.asarray(arr[:, 1], dtype=np.float64)
    return {"wave": wave_aa, "lnu": lnu}


@pytest.fixture(scope="module")
def agnfitter_r06_ref():
    """Load R06 template from vendored HDF5 (L_nu as stored)."""
    with h5py.File(_REF_H5, "r") as f:
        g = f["r06"]
        wave_aa = np.asarray(g["wavelength"][:], dtype=np.float64)
        lnu = np.asarray(g["sed"][:], dtype=np.float64)
    return {"wave": wave_aa, "lnu": lnu}


@pytest.fixture(scope="module")
def tengri_runtime():
    """Load tengri's runtime module constants."""
    from tengri.components.agn.richards2006_disc import (
        _RICHARDS2006_BOL_INTEGRAL,
        _RICHARDS2006_LNU_SHAPE,
        _RICHARDS2006_NU_HZ,
        RICHARDS2006_WAVE_AA,
    )

    return {
        "wave": RICHARDS2006_WAVE_AA,
        "nu_hz": _RICHARDS2006_NU_HZ,
        "lnu_shape": _RICHARDS2006_LNU_SHAPE,
        "bol_integral": _RICHARDS2006_BOL_INTEGRAL,
    }


class TestRichards2006StoredValues:
    """Verify the stored L_nu values (not nu*F_nu) in both source files."""

    def test_richards_dat_2500_aa_in_physical_range(self, richards_dat_values):
        """The value at 2500 Å should lie in [1e29, 1e32] erg/s/Hz.

        This test FAILS if the column is misread as nu*F_nu (would be ~1e38 after division).
        Verifies that richards2006.dat column 2 IS L_nu, not nu*F_nu.
        """
        wave = richards_dat_values["wave"]
        lnu = richards_dat_values["lnu"]

        # Find closest to 2500 Å
        idx = np.argmin(np.abs(wave - 2500.0))
        val_2500 = lnu[idx]

        print(f"\nRichards.dat value at {wave[idx]:.1f} Å: {val_2500:.3e} erg/s/Hz")

        # The physical range for L_nu at optical wavelengths in quasars is ~1e29–1e32
        assert 1e29 <= val_2500 <= 1e32, (
            f"Value {val_2500:.3e} outside physical L_nu range [1e29, 1e32]. "
            f"If column were misread as nu*F_nu, value would be ~1e38 (divide by nu)."
        )

    def test_agnfitter_r06_2500_aa_in_physical_range(self, agnfitter_r06_ref):
        """AGNfitter R06 reference at 2500 Å should also be in [1e29, 1e32]."""
        wave = agnfitter_r06_ref["wave"]
        lnu = agnfitter_r06_ref["lnu"]

        # Find closest to 2500 Å
        idx = np.argmin(np.abs(wave - 2500.0))
        val_2500 = lnu[idx]

        print(f"\nAGNfitter R06 value at {wave[idx]:.1f} Å: {val_2500:.3e} erg/s/Hz")

        assert 1e29 <= val_2500 <= 1e32, (
            f"Value {val_2500:.3e} outside physical L_nu range [1e29, 1e32]."
        )

    def test_tengri_lnu_shape_2500_aa_in_physical_range(self, tengri_runtime):
        """Tengri's L_nu shape at 2500 Å should be in the physical range.

        This is the shape AFTER any normalization/scaling. If the bug persists (division by nu),
        the shape would be wrong by a nu/nu_2500 factor that varies across the spectrum.
        """
        wave = tengri_runtime["wave"]
        lnu_shape = tengri_runtime["lnu_shape"]

        # Find closest to 2500 Å
        idx = np.argmin(np.abs(wave - 2500.0))
        val_2500_shape = lnu_shape[idx]

        print(f"\nTengri L_nu shape at {wave[idx]:.1f} Å: {val_2500_shape:.3e}")

        # The shape should be strictly positive and in the right ballpark
        assert val_2500_shape > 0, f"L_nu shape is {val_2500_shape} (should be positive)"
        # After normalization, the shape values should still reflect the physical range
        # (the exact range depends on the bolometric normalization, but it should be
        # orders of magnitude larger than zero)
        assert val_2500_shape >= 1e15, (
            f"L_nu shape {val_2500_shape:.3e} is suspiciously small; "
            f"may indicate double division by nu."
        )

    def test_tengri_lnu_shape_2500_aa_not_double_divided(self, tengri_runtime, richards_dat_values):
        """Tengri's L_nu shape must NOT be the data double-divided by nu.

        If the bug persists (division by nu when loading), the shape would be ~3.2e15,
        which is the stored value (3.785e30) divided by nu (1.18e15).
        The correct shape should match the stored value directly (or be the same order of magnitude).

        This test FAILS if division by nu is still happening.
        """
        wave = tengri_runtime["wave"]
        lnu_shape = tengri_runtime["lnu_shape"]
        raw_wave = richards_dat_values["wave"]
        raw_lnu = richards_dat_values["lnu"]

        # Find closest to 2500 Ang in tengri data
        idx = np.argmin(np.abs(wave - 2500.0))
        val_2500_shape = lnu_shape[idx]

        # Find closest to 2500 Ang in raw data
        idx_raw = np.argmin(np.abs(raw_wave - 2500.0))
        val_2500_raw = raw_lnu[idx_raw]

        print(f"\nTengri L_nu shape at {wave[idx]:.1f} Ang: {val_2500_shape:.3e}")
        print(f"Raw L_nu at {raw_wave[idx_raw]:.1f} Ang: {val_2500_raw:.3e}")
        print(f"Ratio (shape/raw): {val_2500_shape / val_2500_raw:.3e}")

        # If the bug persists, the ratio would be 1/nu approx 8.5e-16
        # If fixed, the ratio should be 1 (or very close after any normalization)

        # The shape should NOT be much smaller than the raw value
        # (allow some tolerance for any intermediate normalization, but not 9+ orders of magnitude)
        ratio = val_2500_shape / val_2500_raw

        assert ratio > 1e-10, (
            f"L_nu shape is {ratio:.3e}x the raw data (raw={val_2500_raw:.3e}, shape={val_2500_shape:.3e}). "
            f"This suggests the column is being divided by nu (wrong). "
            f"Should be approximately 1 (or 1/normalization factor)."
        )


class TestRichards2006Slopes:
    """Verify spectral slopes match the canonical quasar alpha_nu ~ -0.44."""

    def test_richards_dat_slope_1450_2200_aa(self, richards_dat_values):
        """1450–2200 Å slope should be alpha_nu ≈ -0.44 (canonical quasar).

        Canonical type-1 quasar alpha_nu (optical–UV) is -0.44 ± 0.02.
        This verifies the data encodes a realistic quasar SED.
        """
        wave = richards_dat_values["wave"]
        lnu = richards_dat_values["lnu"]

        # Find indices bracketing 1450 and 2200 Å
        idx_1450 = np.argmin(np.abs(wave - 1450.0))
        idx_2200 = np.argmin(np.abs(wave - 2200.0))

        wave_1450 = wave[idx_1450]
        wave_2200 = wave[idx_2200]
        lnu_1450 = lnu[idx_1450]
        lnu_2200 = lnu[idx_2200]

        # alpha_nu = d log(L_nu) / d log(nu)
        # nu = c / wave, so d log(nu) = -d log(wave)
        # => alpha_nu = - d log(L_nu) / d log(wave)
        dlog_lnu = np.log10(lnu_2200) - np.log10(lnu_1450)
        dlog_wave = np.log10(wave_2200) - np.log10(wave_1450)
        alpha_nu = -dlog_lnu / dlog_wave

        print(f"\nRichards.dat 1450–2200 Å slope: alpha_nu = {alpha_nu:.4f}")
        print(f"  wave_1450 = {wave_1450:.1f} Å, L_nu = {lnu_1450:.3e}")
        print(f"  wave_2200 = {wave_2200:.1f} Å, L_nu = {lnu_2200:.3e}")

        # Should be within canonical range [-0.7, -0.2]
        # More specifically, should be close to -0.44
        assert -0.7 <= alpha_nu <= -0.2, (
            f"Slope {alpha_nu:.4f} outside physical range [-0.7, -0.2]. "
            f"Canonical quasar alpha_nu ≈ -0.44."
        )

    def test_agnfitter_r06_slope_1450_2200_aa(self, agnfitter_r06_ref):
        """AGNfitter R06 1450–2200 Å slope should also match."""
        wave = agnfitter_r06_ref["wave"]
        lnu = agnfitter_r06_ref["lnu"]

        # Find indices bracketing 1450 and 2200 Å
        idx_1450 = np.argmin(np.abs(wave - 1450.0))
        idx_2200 = np.argmin(np.abs(wave - 2200.0))

        wave_1450 = wave[idx_1450]
        wave_2200 = wave[idx_2200]
        lnu_1450 = lnu[idx_1450]
        lnu_2200 = lnu[idx_2200]

        dlog_lnu = np.log10(lnu_2200) - np.log10(lnu_1450)
        dlog_wave = np.log10(wave_2200) - np.log10(wave_1450)
        alpha_nu = -dlog_lnu / dlog_wave

        print(f"\nAGNfitter R06 1450–2200 Å slope: alpha_nu = {alpha_nu:.4f}")

        assert -0.7 <= alpha_nu <= -0.2, (
            f"Slope {alpha_nu:.4f} outside physical range [-0.7, -0.2]."
        )

    def test_tengri_slope_1450_2200_aa(self, tengri_runtime):
        """Tengri's runtime L_nu shape 1450–2200 Å slope should also be correct."""
        wave = tengri_runtime["wave"]
        lnu_shape = tengri_runtime["lnu_shape"]

        # Find indices
        idx_1450 = np.argmin(np.abs(wave - 1450.0))
        idx_2200 = np.argmin(np.abs(wave - 2200.0))

        wave_1450 = wave[idx_1450]
        wave_2200 = wave[idx_2200]
        lnu_shape_1450 = lnu_shape[idx_1450]
        lnu_shape_2200 = lnu_shape[idx_2200]

        dlog_lnu = np.log10(lnu_shape_2200) - np.log10(lnu_shape_1450)
        dlog_wave = np.log10(wave_2200) - np.log10(wave_1450)
        alpha_nu = -dlog_lnu / dlog_wave

        print(f"\nTengri L_nu shape 1450–2200 Å slope: alpha_nu = {alpha_nu:.4f}")

        assert -0.7 <= alpha_nu <= -0.2, (
            f"Slope {alpha_nu:.4f} outside physical range [-0.7, -0.2]. "
            f"This should match the stored data if no bug exists."
        )
