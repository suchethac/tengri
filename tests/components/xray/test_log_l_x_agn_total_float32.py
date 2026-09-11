# SPDX-License-Identifier: BSD-3-Clause
"""Float32 parity for ``log_l_x_agn`` / ``log_l_x_total`` (#1206 §B).

Zero tests pinned these two properties before this PR. Both read the linear
``derived["L_agn_bol"]`` (~1e46 erg/s) and took its ``log10``, which is
``inf`` in float32 and ``log10(inf) -> nan`` after the AGN bolometric
correction — measured on the panchromatic model by the #1206 orchestrator
census: ``log_l_x_agn`` and ``log_l_x_total`` are ``nan`` in pure float32
while float64 gives 42.8055 / 42.8056. The fix reads the AGN component's own
``log_L_agn_bol`` companion instead, never materializing the linear value.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.utils.sed_quantities import compute_log_l_x_agn

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp_bare():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")


def _panchromatic_model(ssp, forward_dtype):
    """The shared AGN + X-ray parity model (stellar + dust + Cue + AGN + radio + X-ray)."""
    from tests.regression.precision.conftest import build_model

    return build_model(ssp, forward_dtype)


def test_log_l_x_agn_float64_matches_manual_log10_of_linear(ssp_bare):
    """float64 ``log_l_x_agn`` is unchanged by the switch to ``log_L_agn_bol``.

    Recomputes the pre-#1206 formula by hand from ``state.derived["L_agn_bol"]``
    -- ``compute_log_l_x_agn(log10(L_agn_bol))`` -- and checks the property
    function agrees to 1e-12 relative: reading the log companion instead of
    the linear value must not move the float64 answer.
    """
    m64 = _panchromatic_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    p["redshift"] = 0.5

    state = m64.predict_state(p)
    l_agn_bol = float(np.asarray(state.derived["L_agn_bol"]))
    assert l_agn_bol > 0.0, "fixture AGN is inactive; pick params where L_agn_bol > 0"
    expected = float(compute_log_l_x_agn(np.log10(l_agn_bol)))

    got = float(m64.predict_properties(p, names=("log_l_x_agn",))["log_l_x_agn"])
    assert np.isfinite(got), f"log_l_x_agn is non-finite in float64: {got}"
    assert_allclose(got, expected, rtol=1e-12)


def test_log_l_x_agn_and_total_pure_float32_finite_and_tracks_float64(ssp_bare):
    """``log_l_x_agn`` / ``log_l_x_total`` are finite in pure float32 and track float64.

    Before this fix both were ``nan`` in pure float32 (``log10(inf)`` chained
    into the AGN bolometric correction), with no test pinning either the
    input overflow or the output. Asserted finite, non-zero (#2100: zero is
    finite and not the same claim), and within 3e-3 relative of the float64
    reference on the same panchromatic model and parameters.
    """
    m64 = _panchromatic_model(ssp_bare, "float64")
    p = dict(m64.spec.sample(jax.random.PRNGKey(0)))
    p["redshift"] = 0.5
    ref = m64.predict_properties(p, names=("log_l_x_agn", "log_l_x_total"))

    with jax.enable_x64(False):
        m32 = _panchromatic_model(ssp_bare, "float32")
        got = m32.predict_properties(p, names=("log_l_x_agn", "log_l_x_total"))

    for name in ("log_l_x_agn", "log_l_x_total"):
        v32 = float(np.asarray(got[name]))
        v64 = float(np.asarray(ref[name]))
        assert np.isfinite(v32), f"{name} is non-finite in pure float32: {v32}"
        assert v32 != 0.0, (
            f"{name} is identically zero in pure float32 — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
        rel = abs(v32 - v64) / max(abs(v64), 1e-12)
        assert rel <= 3e-3, f"{name}: pure-f32 {v32} vs float64 {v64} — rel {rel:.2e} exceeds 3e-3"
