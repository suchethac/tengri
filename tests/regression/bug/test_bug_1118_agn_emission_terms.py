# SPDX-License-Identifier: BSD-3-Clause
"""The composable AGN joins the rank-1 band-response path (#1118).

Under ``WavePrecomp`` the AGN photometry used to be a dense per-call filter integral over
the full wavelength grid (about 98 MFLOPs compiled, against about 0.1 MFLOPs for radio and
X-ray). The AGN is a sum of sub-blocks (disc, torus, lines, polar), each a fixed spectral
shape times an amplitude set by ``agn_log_lbol``. When every shape parameter is fixed the
band flux factorizes exactly as it does for the additive emitters, so the same build-time
response applies.

Pinned here:
  1. with every shape parameter Fixed, the AGN takes the rank-1 path,
  2. that path is EXACT against the dense per-call integral (reference: the same model with
     one shape parameter freed, which the gate refuses, pinned back to its fixed value),
  3. a FREE shape parameter declines the path and says why,
  4. a FREE luminosity (an amplitude) does not decline it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, SEDModel, Uniform, WavePrecomp
from tengri.forward.precompute_report import precompute_engagement_report
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug

BANDS = ["sdss_g", "wise_w4"]


def _agn(**overrides):
    agn = {
        "type": "composable",
        "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "norm": "cigale_joint",
        "log_lbol": Fixed(45.5),
    }
    agn.update(overrides)
    return agn


def _model(*, agn=None, approx=WavePrecomp()):
    return SEDModel.build(
        ssp_data=tengri.load_ssp(),
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        agn=_agn() if agn is None else agn,
        approx=approx,
    )


def _params(m):
    return {k: jnp.asarray(v) for k, v in m.spec.sample(jax.random.PRNGKey(0)).items()}


def _agn_state(m):
    state = precompute_engagement_report(m).emitter_term_responses.get("agn")
    assert state is not None, "the AGN never reached the term band-response gate"
    return state


def test_agn_with_fixed_shape_takes_the_rank1_path():
    """Every shape parameter Fixed: the AGN publishes a term band response and uses it."""
    m = _model()
    m.predict_photometry(_params(m))
    state = _agn_state(m)
    assert state.state == "engaged", state


def test_agn_band_response_is_exact_against_the_dense_filter_integral():
    """The rank-1 photometry equals the dense per-call integral to 1e-6 relative.

    The reference is the same model with ``agn_cos_inc`` freed: that makes the gate refuse
    the response, so the AGN integrates its full SED through the filters on every call.
    The freed value is pinned back to the fixed one, so the two are the same physics.
    """
    fast = _model()
    dense = _model(agn=_agn(torus={"type": "skirtor", "all_params": Fixed(DEFAULT),
                                   "cos_inc": Uniform(0.0, 1.0)}))
    p_dense = _params(dense)
    p_dense["agn_cos_inc"] = jnp.asarray(float(fast.spec.get_fixed_values()["agn_cos_inc"]))

    f_fast = np.asarray(fast.predict_photometry(_params(fast)))
    f_dense = np.asarray(dense.predict_photometry(p_dense))

    assert _agn_state(fast).state == "engaged"
    assert _agn_state(dense).state == "declined"
    nz = f_dense != 0.0
    assert nz.any()
    rel = np.abs(f_fast[nz] / f_dense[nz] - 1.0).max()
    assert rel <= 1e-6, f"AGN band response differs from the dense integral by {rel:.3e}"


def test_a_free_shape_parameter_declines_the_agn_path_with_a_reason():
    """A free inclination changes the torus shape, so the response must be refused."""
    m = _model(agn=_agn(torus={"type": "skirtor", "all_params": Fixed(DEFAULT),
                               "cos_inc": Uniform(0.0, 1.0)}))
    m.predict_photometry(_params(m))
    state = _agn_state(m)
    assert state.state == "declined", state
    assert "agn_cos_inc" in (state.reason or ""), state


def test_a_free_luminosity_is_an_amplitude_and_keeps_the_agn_path():
    """``agn_log_lbol`` scales every term; freeing it must not decline the response."""
    m = _model(agn=_agn(log_lbol=Uniform(44.0, 47.0)))
    m.predict_photometry(_params(m))
    assert _agn_state(m).state == "engaged"
