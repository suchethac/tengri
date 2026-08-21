# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #299 — SSP-grid-vs-burst Nyquist aliasing warning.

The forward model interpolates ``SFR(t)`` at SSP grid points (a
point-sample, not a bin-integral). A burst narrower than the local SSP
grid spacing aliases into a non-physical staircase as the peak crosses
grid boundaries. The proper fix is conservative rebinning (#299 option
1). Until that lands, ``SEDModel.build`` emits a
:class:`SFHBurstAliasingWarning` so the user hits the failure mode at
construction rather than as a visual artefact in their predictions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import tengri
from tengri.components.stellar.sfh._aliasing_warning import SFHBurstAliasingWarning

pytestmark = pytest.mark.regression_bug

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_FILE.is_file():
        pytest.skip(f"SSP file not present: {_SSP_FILE}")
    return tengri.load_ssp()


def _build_with_burst(ssp, *, width_gyr: float, peak_lbt_gyr: float):
    """Build a minimal SEDModel with a tsnorm burst of the given width/peak.

    Filters CB19 warnings so test signal stays clean.
    """
    with warnings.catch_warnings():
        # Silence orthogonal nebular warnings — we only care about the
        # aliasing warning class in these tests.
        warnings.simplefilter("ignore", category=UserWarning)
        # Re-enable the aliasing warning so the assertion can see it.
        warnings.simplefilter("always", category=SFHBurstAliasingWarning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("always", category=SFHBurstAliasingWarning)
            tengri.SEDModel.build(
                ssp,
                sfh={
                    "type": "tsnorm",
                    "*": tengri.FIXED,
                    "log_total_mass": 8.0,
                    "peak_lbt_gyr": peak_lbt_gyr,
                    "width_gyr": width_gyr,
                    "skew": 0.0,
                    "trunc": 5.0,
                    "logzsol": -0.1,
                },
                dust_attenuation={"law": "power_law", "type": "two_component", "*": tengri.FIXED},
            )
            return [x for x in w if issubclass(x.category, SFHBurstAliasingWarning)]


class TestBug299Warning:
    """The build-time aliasing warning fires iff width < grid spacing."""

    def test_narrow_burst_at_6_gyr_warns(self, ssp):
        """The exact #299 repro case: width=0.05 at peak=6 Gyr."""
        hits = _build_with_burst(ssp, width_gyr=0.05, peak_lbt_gyr=6.0)
        assert hits, "#299: narrow burst at peak=6 Gyr produced no aliasing warning"
        msg = str(hits[0].message)
        assert "staircase" in msg
        assert "sfh_tsnorm_width_gyr" in msg
        assert "0.05" in msg

    def test_wide_burst_silent(self, real_ssp_only, ssp):
        """A burst wider than local SSP grid spacing should not warn.

        Needs the real grid: "wide" is defined relative to the real SSP's
        log-age spacing; the coarse synthetic #613 grid shifts that threshold.
        ``real_ssp_only`` skips on synthetic-only CI.
        """
        # 1.0 Gyr is comfortably wider than the SSP grid step at 6 Gyr
        # (~0.6-0.8 Gyr for the standard log-spaced grid).
        hits = _build_with_burst(ssp, width_gyr=1.0, peak_lbt_gyr=6.0)
        assert not hits, (
            f"#299: wide burst (1 Gyr) at peak=6 Gyr produced a false-positive "
            f"aliasing warning: {[str(h.message) for h in hits]}"
        )

    def test_narrow_burst_at_young_peak_warns_too(self, ssp):
        """The aliasing isn't peak=6-Gyr-specific — any narrow burst aliases."""
        # 50 Myr burst at 1 Gyr peak: SSP spacing here is ~120-150 Myr,
        # so 50 Myr is also too narrow.
        hits = _build_with_burst(ssp, width_gyr=0.05, peak_lbt_gyr=1.0)
        assert hits, "narrow burst at young peak should also warn"

    def test_warning_message_names_workaround_threshold(self, ssp):
        """The warning text gives the user a concrete minimum width to use."""
        hits = _build_with_burst(ssp, width_gyr=0.05, peak_lbt_gyr=6.0)
        assert hits
        msg = str(hits[0].message)
        # Names a specific spacing in the message (the value the user
        # needs to exceed). At peak=6 Gyr on the standard grid this is
        # ~0.6-0.8 Gyr.
        assert "Gyr" in msg
        assert "#299" in msg

    def test_warning_class_is_subclass_of_userwarning(self):
        """Catchable under the default warnings filter."""
        assert issubclass(SFHBurstAliasingWarning, UserWarning)
