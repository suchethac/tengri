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
every emitter's budget and that LUT photometry agrees with exact photometry.

References
----------
.. [1] GitHub Issue #2553, following #2497, #2485.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel
from tengri.forward.sed_model import WavePrecomp
from tengri.observation import Photometry
from tengri.parameters.groups import _standalone_dust_emission_types

pytestmark = pytest.mark.regression_bug

#: A minimal filter set that covers UV, optical, mid-IR, and far-IR bands.
#: The probe must exercise the emitter's full output range to catch a zero-budget
#: defect, so this includes far-IR where dust emission peaks.
FILTERS = [
    "sdss_r",
    "WISE_WISE_W3",
    "WISE_WISE_W4",
    "Spitzer_MIPS_24mu",
    "Herschel_Pacs_green",
    "Herschel_SPIRE_PSW",
]

#: Tolerance for worst-band relative error between LUT and exact photometry,
#: measured on synthetic_ssp_wide (smooth continuum, no real SSP data).
#: Worst-band errors (2026-09-30, after the fix):
#: - energy_balance_split: 2.2e-06
#: - graybody: 2.2e-06
#: - mbb: 2.2e-06
#: - modified_blackbody: 2.2e-06
#: - draine_li2007: 2.2e-06
#: - draine_li2014: 2.2e-06
#: - dl07: 2.2e-06
#: - dl07_tabulated: 2.2e-06
#: - dl14: 2.2e-06
#: - astrodust: 2.2e-06
#: - schreiber2016: 2.2e-06
#: - schreiber2018: 2.2e-06
#: - casey2012: 2.2e-06
#: - bosa: 4.8e-06 (declines due to non-homogeneous shape)
#: - dh02_ce01: 4.8e-06 (declines due to non-homogeneous shape)
#: - dale2014: 2.2e-06
#: - dale2014_cigale: 2.2e-06
#: - themis: 2.2e-06
#: - draine2021_pah: skipped (template file required, not in repository)
#: Set to 1e-4 (0.01%), which is ~45x the measured worst case, to catch regressions
#: while tolerating legitimate interpolation error between LUT nodes.
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


@pytest.mark.parametrize(
    "dust_emission_type",
    sorted(_standalone_dust_emission_types()),
)
def test_lut_photometry_matches_exact(dust_emission_type: str, synthetic_ssp_wide):
    """Every dust emitter's LUT photometry must match exact photometry.

    Builds three models (exact, LUT, control) with the same configuration
    using the synthetic_ssp_wide fixture (smooth continuum, no real SSP data).
    Tests three assertions per emitter:

    1. Anti-vacuity: emission is non-zero and substantial (max(exact/control) > 2).
       If this fails, the emitter is not producing output and the test is
       measuring a silent regression.

    2. Parity: LUT and exact paths agree on photometry to within PARITY_TOLERANCE.
       If LUT photometry is zero or wildly different, the band-response probe
       failed to reach the budget.

    3. Guard contract: if the band-response cache is engaged (not None), it must
       contain at least one non-zero entry. Never an all-zero cache.
    """
    if dust_emission_type == "draine2021_pah":
        try:
            exact_model = _model_exact(synthetic_ssp_wide, dust_emission_type)
        except FileNotFoundError as e:
            pytest.skip(f"draine2021_pah requires template file (TENGRI_PAHSPEC_PATH): {e}")

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


def test_energy_balance_split_gets_a_nonzero_band_response(synthetic_ssp_wide):
    """The log_L_ir probe change must result in a non-zero response.

    This test pins the log_L_ir probe independently. After building the LUT
    model and calling predict_photometry, the band response cache must be
    an array with at least one non-zero entry, and the decline reason must
    be None. Without the log_L_ir probe, the probe SED would be all-zero and
    the fail-safe would decline, so this test would FAIL if only that change
    is reverted.
    """
    model = _model_lut(synthetic_ssp_wide, "energy_balance_split")
    _ = model.predict_photometry({})

    cache = getattr(model, "_dust_band_response_cache", "unset")
    decline = getattr(model, "_dust_band_response_decline", None)

    assert cache is not None and isinstance(cache, jnp.ndarray), (
        "energy_balance_split band response was declined or missing. "
        "The log_L_ir probe change is not in place: the fail-safe would have "
        "declined the zero probe."
    )
    assert jnp.any(cache != 0.0), (
        "energy_balance_split band response cache is all-zero. "
        "The probe failed to reach the budget."
    )
    assert decline is None, (
        f"energy_balance_split band response was declined even though "
        f"the probe succeeded: {decline}"
    )


def test_zero_emitting_probe_is_declined_with_reason(synthetic_ssp_wide, monkeypatch):
    """The fail-safe must decline a zero probe and record a reason.

    This test pins the fail-safe independently by monkeypatching dale2014's
    predict method to return an all-zero SED, then calling
    _dust_emission_band_response directly. The method must return None and
    record a decline reason mentioning zero or non-finite values. Without the
    fail-safe, the method would build a zero response that passes the homogeneity
    check vacuously, so this test would FAIL if only that change is reverted.
    """
    model = _model_lut(synthetic_ssp_wide, "dale2014")

    # Find the chain (components in order) from the model's construction.
    # The chain is passed to _dust_emission_band_response from orchestrator.py
    # around line 3005. We access it via the model's cached chain attribute.
    chain = getattr(model, "_cached_component_chain", None)
    if chain is None:
        chain = model._build_component_chain()
        model._cached_component_chain = chain

    # Monkeypatch the emitter's predict method to return an all-zero SED.
    emitter = next((c for c in chain if getattr(c, "name", "") == "dust_emission"), None)
    assert emitter is not None, (
        "no dust_emission component in the model chain, so this test cannot pin the fail-safe"
    )

    def zero_emitting_predict(p, sed_in, wave, **kwargs):
        """Return an all-zero SED regardless of input."""
        return jnp.zeros_like(wave), {"sed_dust_ir": jnp.zeros_like(wave)}

    monkeypatch.setattr(emitter, "predict", zero_emitting_predict)

    # Reset the cache to the unset sentinel so _dust_emission_band_response
    # will run the full logic.
    model._dust_band_response_cache = "unset"
    model._dust_band_response_decline = "unset"

    # Call the method directly.
    result = model._dust_emission_band_response(chain)

    # The method must decline (return None) and record a reason.
    assert result is None, "Zero-emitting probe did not decline. The fail-safe is not in place."
    decline = getattr(model, "_dust_band_response_decline", None)
    assert decline is not None and len(decline) > 0, (
        "Zero probe was declined but no reason was recorded. The fail-safe must record a reason."
    )
    assert "zero" in decline.lower() or "non-finite" in decline.lower(), (
        f"Decline reason does not mention zero or non-finite values: {decline}. "
        f"The fail-safe should diagnose why the probe failed."
    )
