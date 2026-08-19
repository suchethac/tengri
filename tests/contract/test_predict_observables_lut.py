# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for the #281 predict-half fix (2026-07).

``ForwardModel.predict_observables`` used to run the full-resolution component
cube even for a single-population, photometry-only model built with a
``WavePrecomp`` LUT — so the LUT never accelerated it (~11-16x slower than
``predict_photometry``). It now delegates such models to the inner SEDModel's
LUT-aware ``predict_observables_jit``. Multi-population, spatial,
spectroscopy/joint, hierarchical, and exact (non-LUT) models are unaffected.

Guard: post-fix, ``forward.predict_observables(p)["phot_fnu"]`` is **bit-identical**
to ``model.predict_photometry(p)`` (both go through the LUT orchestrator). Pre-fix
they differed by the LUT approximation (~1e-4, cube vs LUT).
"""

import chex
import jax
import pytest

from tengri import FIXED, Fixed, ForwardModel, SEDModel, Uniform, WavePrecomp

pytestmark = pytest.mark.contract


def _model(ssp, obs, approx=None):
    return SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "*": FIXED},
        dust={"law": "power_law", 
            "type": "two_component",
            "*": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "emission": None,
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=approx,
    )


def test_lut_photometry_delegates_to_jit(synthetic_ssp_wide, synthetic_tophat_obs):
    """A photometry-only WavePrecomp forward serves phot_fnu from the LUT path."""
    m = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    fwd = ForwardModel.build(sed=m, observation=synthetic_tophat_obs)
    p = m.spec.sample(jax.random.PRNGKey(0))
    out = fwd.predict_observables(p)
    assert isinstance(out, dict) and "phot_fnu" in out
    # Bit-identical to predict_photometry: both now go through the LUT
    # orchestrator. (Pre-fix, predict_observables used the exact cube and
    # differed by the ~1e-4 LUT approximation.)
    chex.assert_trees_all_close(
        out["phot_fnu"],
        m.predict_photometry(p),
        rtol=0.0,
        atol=0.0,
        custom_message="predict_observables phot_fnu must equal predict_photometry (LUT path)",
    )


def test_exact_model_falls_through(synthetic_ssp_wide, synthetic_tophat_obs):
    """An exact (approx=None) model keeps the general cube path — guard excludes it."""
    m = _model(synthetic_ssp_wide, synthetic_tophat_obs)  # approx=None
    fwd = ForwardModel.build(sed=m, observation=synthetic_tophat_obs)
    p = m.spec.sample(jax.random.PRNGKey(1))
    out = fwd.predict_observables(p)
    assert isinstance(out, dict) and "phot_fnu" in out
    # Exact cube path and the exact predict_photometry agree bit-for-bit
    # (no LUT on either side here).
    chex.assert_trees_all_close(out["phot_fnu"], m.predict_photometry(p), rtol=1e-6)


def test_lut_vs_exact_within_tolerance(synthetic_ssp_wide, synthetic_tophat_obs):
    """The LUT fast-path and the exact cube path agree to the LUT tolerance."""
    p = _model(synthetic_ssp_wide, synthetic_tophat_obs).spec.sample(jax.random.PRNGKey(2))
    lut = ForwardModel.build(
        sed=_model(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp()),
        observation=synthetic_tophat_obs,
    ).predict_observables(p)["phot_fnu"]
    exact = ForwardModel.build(
        sed=_model(synthetic_ssp_wide, synthetic_tophat_obs),
        observation=synthetic_tophat_obs,
    ).predict_observables(p)["phot_fnu"]
    chex.assert_trees_all_close(lut, exact, rtol=5e-3)
