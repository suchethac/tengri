# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #463 — dust-emission SED truncated at SSP grid edge.

Before the fix, ``SEDModel._init_multiwavelength`` only extended the rest-frame
wavelength grid past the SSP grid (~1.6e6 Å = 160 µm for BC03/MILES) when a
radio or X-ray component was active. Dust-emission templates (Dale2014 → 225 mm,
DL14 → 10 mm) interpolate onto ``state.wave`` with ``right=0.0``, so any flux
past the SSP edge was silently zero-filled — the FIR peak / Rayleigh-Jeans tail
visible in CIGALE / MAGPHYS / Prospector was discarded.

Fix: trigger ``make_panchromatic_grid`` whenever a dust-emission component is
attached (sed_model.py, _init_multiwavelength). The downstream
``nonstell.py``→``dust_ir_emission`` path already passes ``rest_wave_f64`` into
the dust template, so it auto-picks up the wider grid once the orchestrator
provides it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tengri import FIXED, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.parameters.priors import Fixed

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_DALE2014 = _DATA_DIR / "dale2014_templates.h5"

pytestmark = [
    pytest.mark.regression_bug,
    pytest.mark.skipif(
        not _SSP_FILE.is_file(),
        reason=f"SSP data not found at {_SSP_FILE}",
    ),
    pytest.mark.skipif(
        not _DALE2014.is_file(),
        reason="Dale2014 dust template not found",
    ),
]


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_FILE))


def _build_model(ssp, *, with_dust_emission: bool):
    """Tiny dusty-SFG model with or without dust IR emission."""
    dust_kwargs: dict = {
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.6),
        "tau_diff": Fixed(0.5),
        "*": FIXED,
    }
    if with_dust_emission:
        dust_kwargs["emission"] = {"type": "dale2014", "*": FIXED}

    return SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "dpl",
            "tau_gyr": Fixed(5.0),
            "alpha": Fixed(2.0),
            "beta": Fixed(1.5),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        dust_attenuation=dust_kwargs,
        redshift=Fixed(0.0),
    )


def test_dust_emission_extends_master_wavelength_grid(ssp):
    """``state.wave`` extends past the SSP grid edge when dust IR is on.

    Pre-fix: ``state.wave.max() == ssp_wave.max()`` (truncated at SSP edge).
    Post-fix: ``state.wave.max() > ssp_wave.max()``, covering the dust
    template's Rayleigh-Jeans tail (Dale2014 → 225 mm).
    """
    ssp_wave_max_aa = float(np.asarray(ssp.ssp_wave).max())

    model_no_dust_ir = _build_model(ssp, with_dust_emission=False)
    state_no_dust_ir = model_no_dust_ir.predict_state({})
    wave_no_dust_ir_max = float(np.asarray(state_no_dust_ir.wave).max())

    # Control: without dust IR emission, grid equals SSP grid (no extension).
    assert wave_no_dust_ir_max == pytest.approx(ssp_wave_max_aa, rel=1e-12), (
        "Sanity check: without dust IR the master grid should be the SSP grid."
    )

    model = _build_model(ssp, with_dust_emission=True)
    state = model.predict_state({})
    wave_max_aa = float(np.asarray(state.wave).max())

    # Post-fix: the master grid extends strictly past the SSP edge so the
    # dust template's submm/mm tail is not clipped by ``right=0.0`` in
    # ``jnp.interp``. The extension uses the panchromatic radio wing
    # (RADIO_WAVE_MAX = 3e11 Å), which covers Dale2014 (2.25e9 Å) with margin.
    assert wave_max_aa > ssp_wave_max_aa, (
        f"Master wavelength grid was not extended past the SSP edge. "
        f"Got wave.max()={wave_max_aa:.3e} Å (~{wave_max_aa / 1e4:.3g} µm); "
        f"SSP edge is {ssp_wave_max_aa:.3e} Å. "
        f"Dust-emission templates require an extended grid (#463)."
    )


def test_dust_emission_has_nonzero_flux_past_ssp_edge(ssp):
    """The FIR/submm/mm tail must carry real flux, not be zero-filled."""
    model = _build_model(ssp, with_dust_emission=True)
    result = model.predict_rest_sed({})

    wave = np.asarray(result.wavelength)
    sed = np.asarray(result.sed)

    ssp_wave_max_aa = float(np.asarray(ssp.ssp_wave).max())
    past_ssp = wave > ssp_wave_max_aa

    assert past_ssp.any(), "No grid points past the SSP edge — grid extension not triggered."
    # At least one bin past the SSP edge must have positive flux from dust IR.
    assert np.any(sed[past_ssp] > 0.0), (
        "All flux past the SSP edge is zero — dust template was clipped (#463)."
    )


def test_master_grid_unions_dale2014_native_nodes(ssp):
    """The master grid is the UNION of the SSP grid + the Dale2014 native grid.

    This locks in the architecture: ``_init_multiwavelength`` queries
    ``tengri.forward.wavelength_extension`` for the component's native template
    grid and unions every node verbatim into ``state.wave`` — not a log-spaced
    approximation. If a future refactor swaps the union for a uniform radio
    wing (the fix-(a) shortcut), this test fails.
    """
    from tengri.forward.wavelength_extension import native_wave_dust_emission

    model = _build_model(ssp, with_dust_emission=True)
    master = np.asarray(model._rest_wavelength)
    ssp_wave = np.asarray(ssp.ssp_wave)
    dale = native_wave_dust_emission("dale2014")

    assert dale is not None, "Dale2014 native grid not discoverable from the registry."

    # Every Dale2014 node past the SSP edge must appear verbatim in the
    # master grid. Use ``np.isin`` because both arrays come from ``np.sort`` /
    # ``np.unique`` and the float values are identical bit-for-bit (no
    # interpolation, no resampling).
    past_ssp = dale[dale > ssp_wave.max()]
    assert past_ssp.size > 0, "Dale2014 template should cover wavelengths past the SSP edge."

    missing = past_ssp[~np.isin(past_ssp, master)]
    assert missing.size == 0, (
        f"{missing.size}/{past_ssp.size} Dale2014 native nodes past the SSP "
        f"edge are missing from the master grid — union semantics broken. "
        f"First few missing: {missing[:5]}"
    )

    # And the master grid's upper bound matches Dale2014's, not a hardcoded
    # constant like RADIO_WAVE_MAX (which fix-(a) would have done).
    assert master.max() == pytest.approx(float(dale.max()), rel=1e-12), (
        f"Master grid max ({master.max():.3e}) doesn't track the template's "
        f"native max ({float(dale.max()):.3e}). The union architecture should "
        f"keep extension tight to the template, not pad to a global constant."
    )
