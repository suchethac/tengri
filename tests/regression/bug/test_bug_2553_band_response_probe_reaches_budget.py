# SPDX-License-Identifier: BSD-3-Clause
"""#2553: the band-response probe must reach every dust emitter's luminosity budget.

Under ``approx=WavePrecomp()`` the dust emission model ``energy_balance_split``
contributed ZERO to photometry. LUT photometry equalled a no-dust-emission control
in every band.

The emitter consumes its budget through ``log_L_ir`` (a separate optional input from
the linear ``L_ir``), but the band-response probe passed only the linear form. When
``log_L_ir`` defaulted to ``-inf`` (nothing absorbed), both probe points returned
all-zero SEDs. The homogeneity check ``allclose(0, k * 0)`` passed vacuously, so a
zero response was cached and the emitter's photometry stayed silent.

The fix has two independent parts, each pinned by its own test here:

1. The probe passes ``log_L_ir`` to any emitter that declares it.
2. A probe SED that is identically zero or non-finite is declined with a recorded
   reason, so the exact per-call integral is used instead of a zero response.

References
----------
.. [1] GitHub Issue #2553, following #2497, #2485.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel
from tengri.forward.sed_model import WavePrecomp
from tengri.observation import Photometry
from tengri.parameters.groups import _standalone_dust_emission_types

pytestmark = pytest.mark.regression_bug

#: A filter set spanning UV, optical, mid-IR and far-IR. Far-IR is included because
#: that is where dust emission peaks and where a zero-budget defect is visible.
FILTERS = [
    "sdss_r",
    "WISE_WISE_W3",
    "WISE_WISE_W4",
    "Spitzer_MIPS_24mu",
    "Herschel_Pacs_green",
    "Herschel_SPIRE_PSW",
]

#: Worst-band relative error allowed between LUT and exact photometry. The LUT
#: interpolates between redshift nodes and quadrature sub-bands, so a small residual
#: is expected; a dropped emitter is off by orders of magnitude.
PARITY_TOLERANCE = 1e-4


def _observation() -> Observation:
    return Observation(photometry=Photometry.from_names(FILTERS))


def _build(ssp, dust_emission_type: str | None, *, approx=None) -> SEDModel:
    kwargs = {
        "ssp_data": ssp,
        "observation": _observation(),
        "sfh": {"type": "dpl", "all_params": Fixed(DEFAULT)},
        "neb": {"type": "none"},
        "dust_attenuation": {
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 1.0,
            "tau_diff": 1.0,
            "all_params": Fixed(DEFAULT),
        },
        "redshift": Fixed(0.1),
    }
    if dust_emission_type is not None:
        kwargs["dust_emission"] = {"type": dust_emission_type, "all_params": Fixed(DEFAULT)}
    if approx is not None:
        kwargs["approx"] = approx
    return SEDModel.build(**kwargs)


@pytest.mark.parametrize("dust_emission_type", sorted(_standalone_dust_emission_types()))
def test_lut_photometry_matches_exact(dust_emission_type: str, synthetic_ssp_wide):
    """Every standalone dust emitter's LUT photometry must match its exact photometry.

    Three assertions per emitter, each closing a way the check could pass vacuously:

    1. Anti-vacuity: the emitter moves the bands, max(exact / control) > 2.
    2. Parity: LUT and exact agree in the worst band to within PARITY_TOLERANCE.
    3. Guard contract: if the band-response cache is engaged, it holds a non-zero
       array. An all-zero response is never accepted.
    """
    if dust_emission_type == "draine2021_pah" and not os.environ.get("TENGRI_PAHSPEC_PATH"):
        # Without its template the component emits nothing in either arm, so the
        # anti-vacuity assertion below could not hold.
        pytest.skip("draine2021_pah requires TENGRI_PAHSPEC_PATH (template file)")
    exact_model = _build(synthetic_ssp_wide, dust_emission_type)
    lut_model = _build(synthetic_ssp_wide, dust_emission_type, approx=WavePrecomp())
    control_model = _build(synthetic_ssp_wide, None)

    exact_phot = exact_model.predict_photometry({})
    lut_phot = lut_model.predict_photometry({})
    control_phot = control_model.predict_photometry({})

    max_emission_ratio = jnp.max(exact_phot / control_phot)
    assert max_emission_ratio > 2.0, (
        f"{dust_emission_type}: emission is not substantial "
        f"(max(exact/control) = {max_emission_ratio:.3e} <= 2); the test setup is broken."
    )

    worst_band_error = jnp.max(jnp.abs(lut_phot / exact_phot - 1.0))
    assert worst_band_error <= PARITY_TOLERANCE, (
        f"{dust_emission_type}: LUT photometry deviates from exact photometry. "
        f"Worst-band relative error = {worst_band_error:.3e} > {PARITY_TOLERANCE:.3e}. "
        f"The band-response probe may not reach the emitter's luminosity budget."
    )

    cache = getattr(lut_model, "_dust_band_response_cache", "unset")
    if cache is not None:
        values = cache["values"]
        assert jnp.any(values != 0.0), (
            f"{dust_emission_type}: band response is all-zero; the probe did not reach "
            f"the emitter's luminosity budget (#2553)."
        )


def test_energy_balance_split_gets_a_nonzero_band_response(synthetic_ssp_wide):
    """Part 1: the log_L_ir probe must give energy_balance_split a non-zero response.

    Without passing log_L_ir, the probe SED is all-zero and the response is either
    zero or declined, so this test fails if only that change is reverted.
    """
    model = _build(synthetic_ssp_wide, "energy_balance_split", approx=WavePrecomp())
    _ = model.predict_photometry({})

    cache = getattr(model, "_dust_band_response_cache", "unset")
    decline = getattr(model, "_dust_band_response_decline", None)

    assert cache is not None, (
        f"energy_balance_split band response was declined ({decline}); the log_L_ir "
        f"probe is not reaching the emitter."
    )
    assert jnp.any(cache["values"] != 0.0), "energy_balance_split band response is all-zero."
    assert decline is None, f"energy_balance_split band response was declined: {decline}"


def test_zero_emitting_probe_is_declined_with_reason(synthetic_ssp_wide, monkeypatch):
    """Part 2: a zero probe SED must be declined, with a recorded reason.

    The emitter's predict is replaced with one that returns an all-zero SED. Without
    the fail-safe, the zero response passes the homogeneity check vacuously and is
    cached, so this test fails if only that change is reverted.
    """
    model = _build(synthetic_ssp_wide, "dale2014", approx=WavePrecomp())

    chain = getattr(model, "_cached_component_chain", None)
    if chain is None:
        chain = model._build_component_chain()
        model._cached_component_chain = chain

    emitter = next((c for c in chain if getattr(c, "name", "") == "dust_emission"), None)
    assert emitter is not None, "no dust_emission component in the chain"

    def zero_emitting_predict(p, sed_in, wave, **kwargs):
        return jnp.zeros_like(wave), {}

    monkeypatch.setattr(emitter, "predict", zero_emitting_predict)

    model._dust_band_response_cache = "unset"
    model._dust_band_response_decline = None

    result = model._dust_emission_band_response(chain)

    assert result is None, "a zero-emitting probe was not declined"
    decline = getattr(model, "_dust_band_response_decline", None)
    assert decline, "a zero probe was declined but no reason was recorded"
    assert "zero" in decline.lower() or "non-finite" in decline.lower(), (
        f"decline reason does not name the zero or non-finite probe: {decline}"
    )
