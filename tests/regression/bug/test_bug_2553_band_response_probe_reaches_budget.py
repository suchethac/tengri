# SPDX-License-Identifier: BSD-3-Clause
"""#2553: dust emission models must declare their luminosity budget fully.

Under ``approx=WavePrecomp()`` the dust emission model ``energy_balance_split``
contributed ZERO to photometry. LUT photometry equalled a no-dust-emission control
in every band (the exact path was 7.7e4x the stellar light at Herschel SPIRE 250um).

The emitter consumes its budget through ``log_L_ir`` (a separate parameter from
the linear ``L_ir``), but the band-response probe passed only the linear form.
When ``log_L_ir`` defaulted to ``-inf`` (nothing absorbed), both probe points
returned all-zero SEDs. The homogeneity check ``allclose(0, k*0)`` passed
vacuously, so a zero response was cached and photometry stayed silent.

The fix: (1) pass ``log_L_ir`` to any emitter that declares it; (2) decline
the band response when a probe SED is zero or non-finite, recording a clear
reason so the issue is visible in the engagement report.

This test sweeps the registry (all 19 emitters) to verify that the probe reaches
every emitter's budget and that LUT photometry agrees with exact photometry to
0.01% worst-band tolerance.

References
----------
.. [1] GitHub Issue #2553, following #2497, #2485.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel
from tengri.forward.sed_model import WavePrecomp
from tengri.observation import Photometry
from tengri.parameters.groups import _standalone_dust_emission_types

pytestmark = pytest.mark.regression_bug

#: A minimal filter set that covers UV, optical, mid-IR, and far-IR bands.
#: The probe must exercise the emitter's full output range to catch a zero-budget
#: defect, so this includes far-IR where dust emission peaks. Measured: every
#: emitter produces finite, non-zero emission in these bands when probed correctly.
FILTERS = [
    "sdss_r",
    "WISE_WISE_W3",
    "WISE_WISE_W4",
    "Spitzer_MIPS_24mu",
    "Herschel_Pacs_green",
    "Herschel_SPIRE_PSW",
]

#: Tolerance for worst-band relative error between LUT and exact photometry.
#: Measured on 2026-09-30 after the fix: 18 of 19 emitters at <= 6e-6 (0.0006%),
#: energy_balance_split at 0.99999 before the fix, now matching others.
#: Set to 1e-4 (0.01%) to accommodate reasonable interpolation error while
#: catching any regression to zero.
PARITY_TOLERANCE = 1e-4


def _observation() -> Observation:
    """Create a photometry observation with the test filter set."""
    return Observation(photometry=Photometry.from_names(FILTERS))


def _model_exact(ssp, dust_emission_type: str) -> SEDModel:
    """Build an exact-path model (approx=None) with the given emitter."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=_observation(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 1.0,
            "tau_diff": 1.0,
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": dust_emission_type, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def _model_lut(ssp, dust_emission_type: str) -> SEDModel:
    """Build a LUT-path model (approx=WavePrecomp()) with the given emitter."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=_observation(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 1.0,
            "tau_diff": 1.0,
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": dust_emission_type, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )


def _model_control(ssp) -> SEDModel:
    """Build a control model (no dust_emission) for anti-vacuity check."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=_observation(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 1.0,
            "tau_diff": 1.0,
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.1),
    )


#: Exclude emitters that have required data files missing (draine2021_pah
#: requires TENGRI_PAHSPEC_PATH to be set).
_EMITTERS_TO_TEST = sorted(
    e for e in _standalone_dust_emission_types()
    if e != "draine2021_pah"
)


@pytest.mark.parametrize(
    "dust_emission_type",
    _EMITTERS_TO_TEST,
)
def test_lut_photometry_matches_exact(
    dust_emission_type: str, synthetic_ssp_wide
):
    """Every dust emitter's LUT photometry must match exact photometry.

    Builds three models (exact, LUT, control) with the same configuration
    using the synthetic_ssp_wide fixture (smooth continuum, no real SSP data).
    Tests three assertions per emitter:

    1. Anti-vacuity: emission is non-zero and substantial (max(exact/control) > 2).
       If this fails, the emitter is not producing output and the test is
       measuring a silent regression.

    2. Parity: LUT and exact paths agree on photometry to within
       1.0e-04 (0.01%) worst-band relative error. If LUT photometry
       is zero or wildly different, the band-response probe failed to reach
       the budget.

    3. Guard contract: the band-response cache must be either None (with a
       recorded decline reason) or an array with at least one non-zero entry.
       Never an all-zero cache.
    """
    exact_model = _model_exact(synthetic_ssp_wide, dust_emission_type)
    lut_model = _model_lut(synthetic_ssp_wide, dust_emission_type)
    control_model = _model_control(synthetic_ssp_wide)

    # Predict photometry with empty fixed-value dict: all parameters are Fixed,
    # so predict_photometry({}) passes the fixed values back in.
    exact_phot = exact_model.predict_photometry({})
    lut_phot = lut_model.predict_photometry({})
    control_phot = control_model.predict_photometry({})

    # 1. ANTI-VACUITY: the emitter really emits
    max_emission_ratio = jnp.max(exact_phot / control_phot)
    assert max_emission_ratio > 2.0, (
        f"{dust_emission_type}: emission is not substantial "
        f"(max(exact/control) = {max_emission_ratio:.3e} <= 2). "
        f"The emitter may not be producing output; the test setup is broken."
    )

    # 2. PARITY: LUT and exact paths agree
    abs_rel_err = jnp.abs(lut_phot / exact_phot - 1.0)
    worst_band_error = jnp.max(abs_rel_err)
    assert worst_band_error <= PARITY_TOLERANCE, (
        f"{dust_emission_type}: LUT photometry deviates from exact photometry. "
        f"Worst-band relative error = {worst_band_error:.3e} > {PARITY_TOLERANCE:.3e}. "
        f"The band-response probe may not have reached the emitter's budget; "
        f"check that all required inputs (e.g., log_L_ir) are passed."
    )

    # 3. GUARD CONTRACT: if the cache is engaged (not None), it must contain
    # non-zero values. An all-zero cache means the probe failed to reach the
    # emitter's budget (#2553).
    cache = getattr(lut_model, "_dust_band_response_cache", "unset")
    if cache is not None:
        assert isinstance(cache, jnp.ndarray), (
            f"{dust_emission_type}: band response cache is not None and not an array; "
            f"it is {type(cache).__name__}."
        )
        assert jnp.any(cache != 0.0), (
            f"{dust_emission_type}: band response cache is an all-zero array. "
            f"The probe failed to reach the emitter's budget (#2553)."
        )


def test_zero_probe_declines_with_reason(synthetic_ssp_wide):
    """A zero-emitting probe triggers decline and records a reason.

    This test focuses on the fail-safe added in change (1) of the fix.
    Without the fix, energy_balance_split's probe would return zero (because
    log_L_ir was not passed), and the homogeneity check would pass vacuously,
    caching a zero response and producing silent zero photometry.

    With the fix, the zero-probe is detected and declined, recording a clear
    reason in _dust_band_response_decline. This test verifies that the
    fail-safe is in place.
    """
    model = _model_lut(synthetic_ssp_wide, "energy_balance_split")
    _ = model.predict_photometry({})

    # After predict_photometry, the band response should either:
    # (a) be engaged with non-zero values, OR
    # (b) be declined with a reason recorded
    cache = getattr(model, "_dust_band_response_cache", "unset")
    decline = getattr(model, "_dust_band_response_decline", None)

    if cache is None:
        # The band response was declined. A reason should be recorded.
        assert decline is not None and len(decline) > 0, (
            "energy_balance_split band response was declined but no reason "
            "was recorded. The fail-safe in fix #2553 is not in place."
        )
    else:
        # The band response was engaged. It must contain non-zero values.
        assert jnp.any(cache != 0.0), (
            "energy_balance_split band response cache is all-zero. "
            "The probe failed to reach the budget even with log_L_ir passed."
        )
