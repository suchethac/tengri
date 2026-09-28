# SPDX-License-Identifier: BSD-3-Clause
"""A refused dust band response must say why, and dh02_ce01 must not crash.

The band-response precompute is exact only for a dust emission template that
is degree-1 homogeneous in ``L_ir``. Two registered templates are not, and both
were invisible for different reasons:

* ``bosa`` failed the homogeneity probe, which discarded its verdict, so
  ``precompute_engagement_report`` said "unknown reason (gate conditions appear
  satisfied)" about a refusal the code could explain in one line.
* ``dh02_ce01`` never reached the probe. Its ``predict`` demanded ``log_L_ir``
  as a separate argument although ``L_ir`` carries the same quantity, so a
  caller holding only ``L_ir`` hit ``jnp.asarray(None)``. The exception was
  swallowed by the try/except around the precompute and reported as the same
  unexplained decline.

References
----------
.. [1] GitHub Issue #2497, following #2485.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.precompute_report import precompute_engagement_report
from tengri.forward.sed_model import WavePrecomp

pytestmark = pytest.mark.regression_bug

NON_HOMOGENEOUS = ("bosa", "dh02_ce01")


def _model(ssp, obs, dust_emission_type):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={"type": dust_emission_type, "all_params": Fixed(DEFAULT)},
        neb={"type": "ssp"},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )


def test_dh02_ce01_predict_derives_log_l_ir_from_l_ir():
    """The caller holding only ``L_ir`` must not meet ``jnp.asarray(None)``."""
    from tengri.components.dust.emission.templates.dh02_ce01 import DH02CE01IRSEDComponent

    wave = jnp.linspace(1.0e4, 1.0e7, 64)
    emitter = DH02CE01IRSEDComponent()
    sed, published = emitter.predict({}, jnp.zeros_like(wave), wave, L_ir=1.0e44)
    assert jnp.all(jnp.isfinite(sed))
    assert "sed_dust_ir" in published


def test_deriving_log_l_ir_agrees_with_passing_it():
    """The fallback must be the same quantity, not a different one."""
    from tengri.components.dust.emission.templates.dh02_ce01 import DH02CE01IRSEDComponent

    wave = jnp.linspace(1.0e4, 1.0e7, 64)
    emitter = DH02CE01IRSEDComponent()
    l_ir = 3.0e44
    derived, _ = emitter.predict({}, jnp.zeros_like(wave), wave, L_ir=l_ir)
    passed, _ = emitter.predict(
        {}, jnp.zeros_like(wave), wave, L_ir=l_ir, log_L_ir=float(jnp.log10(l_ir))
    )
    assert jnp.allclose(derived, passed, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dust_emission_type", NON_HOMOGENEOUS)
def test_a_refused_band_response_names_homogeneity(
    dust_emission_type, synthetic_tophat_obs, ssp_data_wne
):
    """Declining is right for these templates; declining silently is not."""
    report = precompute_engagement_report(
        _model(ssp_data_wne, synthetic_tophat_obs, dust_emission_type)
    )
    state = report.dust_band_response
    assert state.state == "declined"
    assert "homogeneous" in (state.reason or ""), state.reason
    assert "unknown reason" not in (state.reason or "")


def test_an_engaged_band_response_records_no_decline(synthetic_tophat_obs, ssp_data_wne):
    """The recorded verdict must be cleared on success, not left stale."""
    model = _model(ssp_data_wne, synthetic_tophat_obs, "dale2014")
    report = precompute_engagement_report(model)
    assert report.dust_band_response.state == "engaged"
    assert getattr(model, "_dust_band_response_decline", None) is None
