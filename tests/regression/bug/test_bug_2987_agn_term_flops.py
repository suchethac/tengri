# SPDX-License-Identifier: BSD-3-Clause
"""The AGN term band response must not re-evaluate the AGN per predict call (#2987).

With every AGN parameter but the luminosity fixed, the rank-1 band response already holds
the per-term filter integrals. ``_term_band_photometry`` still called ``emission_terms`` at
each term's reference wavelength on every ``predict_photometry``, and that evaluation runs
the fixed-size ledger and tie integrals (13001, 30001 and 4001 nodes), so the cost was flat
in the number of bands and the base AGN model compiled to 78.6 M FLOPs against 0.44 M with
no AGN. The per-term amplitudes at the reference wavelength are build-time constants when
the AGN is fully fixed, and the table now carries them.

Pinned here:
  1. the base model's compiled photometry stays under an absolute FLOP budget, and
  2. the cached amplitudes reproduce the per-call path: the same model with one AGN shape
     parameter freed to its fixed value takes the per-call path (the gate refuses the table),
     and must agree with the cached model to round-off.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Uniform, WavePrecomp
from tests.regression.bug import test_additive_term_band_response as band

pytestmark = pytest.mark.regression_bug

#: Compiled FLOPs of ``predict_photometry`` on the base AGN model (optical bands plus the
#: X-ray band, all AGN parameters fixed). Measured on darwin/arm64: 440,443 after the fix
#: (the no-AGN model compiles to 440,440). Before the fix the same model compiled to
#: 43,326,484 FLOPs (#2987), with the AGN's reference-wavelength evaluation paid on every
#: call. The budget is about 2x the measured value, so it tolerates the platform spread
#: that ``MAX_EMITTER_FLOPS`` above describes, and it sits ~48x below the regression.
MAX_BASE_AGN_PHOTOMETRY_FLOPS = 900_000


def _base_model():
    return band._model(None, approx=WavePrecomp(), bands="xray_yang20")


def test_base_agn_photometry_does_not_reevaluate_the_agn_per_call():
    """The base AGN model's compiled photometry stays within the budget set by #2987."""
    flops = band._flops(_base_model())
    assert flops < MAX_BASE_AGN_PHOTOMETRY_FLOPS, (
        f"base AGN photometry compiles to {flops:,.0f} FLOPs (budget "
        f"{MAX_BASE_AGN_PHOTOMETRY_FLOPS:,}): the AGN is evaluated per call at its "
        "reference wavelengths (#2987)."
    )


def test_cached_agn_amplitudes_match_the_per_call_path(monkeypatch):
    """A model with the AGN shape freed must reproduce the cached model's photometry.

    The cached model has every AGN parameter fixed, so its amplitudes are build-time constants.
    Freeing ``agn_cos_inc`` to its fixed value makes the gate refuse the term response: that
    model takes the per-call path, and it is the independent reference for the cached one.
    """
    cached = _base_model()
    fixed_cos_inc = float(cached.spec.get_fixed_values()["agn_cos_inc"])
    monkeypatch.setattr(
        band,
        "AGN",
        {**band.AGN, "cos_inc": Uniform(0.2, 0.9)},
    )
    per_call = band._model(None, approx=WavePrecomp(), bands="xray_yang20")
    assert "agn_cos_inc" in per_call.spec.free_params, "the reference must free the shape"

    p_per_call = {**band._params(per_call), "agn_cos_inc": jnp.asarray(fixed_cos_inc)}
    got = np.asarray(cached.predict_photometry(band._params(cached)))
    ref = np.asarray(per_call.predict_photometry(p_per_call))

    nz = ref != 0.0
    assert nz.any()
    rel = np.abs(got[nz] / ref[nz] - 1.0).max()
    assert rel < 1e-6, f"cached AGN amplitudes differ from the per-call path by {rel:.3e}"
