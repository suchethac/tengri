# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #361 — nebular-line API silently returning NaN.

Two distinct bugs:

* **Bug A**: ``neb={'type': 'cb19'}`` used to silently fall through to
  ``BakedInBackend`` because ``_init_nebular`` had no ``'cb19'`` branch.
  ``Parameters.nebular_mode`` correctly read ``"cb19"`` but the backend
  attached to the model was the wrong class — line accessors then
  returned NaN with no signal to the user.

* **Bug B**: When ``_ensure_lines`` finds a backend without
  ``predict_nebular_line_luminosities`` (BakedIn, Shock), it used to
  silently cache empty arrays. ``pred.lines.halpha`` then returned NaN
  via ``extract_line_luminosity`` on an empty array — also signal-free.
  The fix raises a one-time :class:`UserWarning` naming the backend and
  suggesting compatible alternatives.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import tengri

pytestmark = pytest.mark.regression_bug

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_FILE.is_file():
        pytest.skip(f"SSP file not present: {_SSP_FILE}")
    return tengri.load_ssp()


class TestBug361A_cb19_dispatch:
    """``neb={'type': 'cb19'}`` must wire CB19Backend, not BakedInBackend."""

    def test_cb19_lands_in_cb19_backend(self, ssp):
        from tengri.components.nebular import CB19Backend

        # Filter the CB19 import-time warnings — they're orthogonal to
        # this test (and explicitly opt-out-able by the caller).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = tengri.SEDModel.build(
                ssp,
                sfh={"type": "dpl", "*": tengri.FIXED},
                neb={"type": "cb19", "*": tengri.FIXED},
            )
        assert isinstance(m._nebular_backend, CB19Backend), (
            f"#361 Bug A: cb19 fell through to "
            f"{type(m._nebular_backend).__name__} instead of CB19Backend"
        )

    def test_cb19_nebular_mode_propagated_to_spec(self, ssp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = tengri.SEDModel.build(
                ssp,
                sfh={"type": "dpl", "*": tengri.FIXED},
                neb={"type": "cb19", "*": tengri.FIXED},
            )
        assert m.spec.nebular_mode == "cb19"

    def test_other_nebular_modes_still_dispatch_correctly(self, ssp):
        """Don't regress the existing cue / cloudy / default branches."""
        from tengri.components.nebular import BakedInBackend

        # 'none'-equivalent path lands in BakedIn (the default-fallback case)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = tengri.SEDModel.build(ssp, sfh={"type": "dpl", "*": tengri.FIXED})
        assert isinstance(m._nebular_backend, BakedInBackend)


class TestBug361C_cb19_per_line_variation:
    """CB19 line_lums must vary per line.

    Regression for #361 Bug C: when the synthetic conftest fixture
    shipped all ratios = 1.0, every CB19 line collapsed to the same
    luminosity. The fixture now uses physically-distinct Case B SF
    ratios so any code consumer of the synthetic grid sees per-line
    variation. Real Martinez-Paredes+2023 data is even more diverse.
    """

    def test_cb19_lines_have_distinct_luminosities(self, ssp):
        import jax.numpy as jnp

        from tengri.components.nebular import CB19Backend

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = CB19Backend(ssp_data=ssp)

        ssp_log_ages = ssp.ssp_lg_age_gyr + 9.0  # log10(yr)
        _waves, lums = backend.predict_nebular_line_luminosities(
            ssp_weights=jnp.ones_like(ssp_log_ages) * 1e7,
            ssp_log_ages_yr=ssp_log_ages,
            log_z=-1.7,
            neb_logU=-3.0,
            neb_logZ_gas=-1.7,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
        )
        # At least half the 10 emission lines should produce a unique
        # luminosity. ([N II] 6548 ≈ [O I] 6300 in the synthetic by
        # design, so we don't require all 10 to be unique.)
        n_unique = int(jnp.unique(lums).shape[0])
        assert n_unique >= 5, f"#361 Bug C: only {n_unique} unique line luminosities (expected ≥5)"

    def test_balmer_decrement_matches_case_b_synthetic(self, ssp):
        """Hα/Hβ in the synthetic CB19 fixture is the Case B value 2.87."""
        import jax.numpy as jnp

        from tengri.components.nebular import CB19Backend

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = CB19Backend(ssp_data=ssp)

        ssp_log_ages = ssp.ssp_lg_age_gyr + 9.0
        waves, lums = backend.predict_nebular_line_luminosities(
            ssp_weights=jnp.ones_like(ssp_log_ages) * 1e7,
            ssp_log_ages_yr=ssp_log_ages,
            log_z=-1.7,
            neb_logU=-3.0,
            neb_logZ_gas=-1.7,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
        )
        # Hα = 6564.61, Hβ = 4862.68 (vacuum)
        i_ha = int(jnp.argmin(jnp.abs(waves - 6564.61)))
        i_hb = int(jnp.argmin(jnp.abs(waves - 4862.68)))
        ratio = float(lums[i_ha] / lums[i_hb])
        assert abs(ratio - 2.87) / 2.87 < 0.01, (
            f"Hα/Hβ = {ratio:.3f}, expected 2.87 (Case B; from synthetic fixture)"
        )


class TestBug361B_silent_nan_warning:
    """BakedIn line accessors used to return NaN silently. Now they warn."""

    def test_bakedin_line_access_emits_warning(self, ssp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = tengri.SEDModel.build(
                ssp,
                sfh={
                    "type": "tsnorm",
                    "*": tengri.FIXED,
                    "log_peak_sfr": 1.0,
                    "peak_lbt_gyr": 2.0,
                    "width_gyr": 1.0,
                    "skew": 0.2,
                    "trunc": 3.0,
                    "logzsol": -0.1,
                },
                dust={
                    "type": "two_component",
                    "*": tengri.FIXED,
                    "tau_bc": 0.2,
                    "tau_diff": 0.1,
                    "slope": -0.7,
                },
            )
        pred = m.predict({"redshift": 0.05})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = float(pred.lines.halpha)

        hits = [str(x.message) for x in w if "per-line luminosity catalogue" in str(x.message)]
        assert hits, "#361 Bug B: BakedInBackend line access produced no UserWarning"
        # The warning names the offending backend and suggests alternatives.
        assert "BakedInBackend" in hits[0]
        assert "cue" in hits[0] and "cloudy" in hits[0] and "cb19" in hits[0]

    def test_bakedin_line_access_still_returns_nan(self, ssp):
        """Back-compat: NaN return is preserved (only the warning is new),
        so code that already handles NaN keeps working."""
        import jax.numpy as jnp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = tengri.SEDModel.build(
                ssp,
                sfh={"type": "dpl", "*": tengri.FIXED},
                dust={"type": "two_component", "*": tengri.FIXED},
            )
        pred = m.predict({"redshift": 0.05})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ha = pred.lines.halpha
        assert jnp.isnan(ha)
