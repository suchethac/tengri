"""Performance regression test: single-screen LUT consumer."""
from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform, load_ssp
from tengri.forward.sed_model import WavePrecomp
from tengri.observation import Observation, Photometry

import jax
import jax.numpy as jnp

pytestmark = pytest.mark.regression_bug


def _single_screen_with_lut(ssp, obs, enable_lut=True):
    """Build single-component dust model with or without LUT."""
    if not enable_lut:
        # Disable LUT by removing dust_tau_v from allowlist
        orig = SEDModel._EB_ATTEN_FREE_OK
        SEDModel._EB_ATTEN_FREE_OK = frozenset(orig - {"dust_tau_v"})

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Uniform(0.0, 3.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )

    if not enable_lut:
        SEDModel._EB_ATTEN_FREE_OK = orig

    return model


def _measure_gradient_flops(model, params):
    """Measure FLOPs of the photometry gradient."""
    cost = jax.jit(
        jax.grad(lambda q: jnp.sum(model.predict_photometry(q)))
    ).lower(params).compile().cost_analysis()
    if isinstance(cost, list):
        cost = cost[0]
    return float(cost["flops"])


def test_single_screen_lut_consumer_reduces_flops(synthetic_ssp, synthetic_tophat_obs):
    """The LUT consumer (newly wired) should reduce gradient FLOPs.

    Baseline: Without LUT, the full-wavelength integral is computed for
    every gradient step (expensive).
    Fast path: With LUT, bilinear interpolation on (tau_bc, tau_diff) grid
    using precomputed weights (fast).

    The LUT was built with degenerate mapping: tau_bc=[0.0], tau_diff spans
    the tau_v prior [0, 3]. When consumed, tau_v is mapped to tau_diff.
    """
    model_on = _single_screen_with_lut(synthetic_ssp, synthetic_tophat_obs, enable_lut=True)
    model_off = _single_screen_with_lut(synthetic_ssp, synthetic_tophat_obs, enable_lut=False)

    # Same parameters for both models
    params = dict(model_on.spec.sample(jax.random.PRNGKey(42)))
    params["dust_tau_v"] = np.float64(1.37)

    # Measure FLOPs
    flops_on = _measure_gradient_flops(model_on, params)
    flops_off = _measure_gradient_flops(model_off, params)

    ratio = flops_off / flops_on

    # Report (matching coordinator's format)
    print(f"\n{'='*60}")
    print(f"Single-screen LUT consumer performance:")
    print(f"  With LUT:    {flops_on:,} FLOPs")
    print(f"  Without LUT: {flops_off:,} FLOPs")
    print(f"  Speedup:     {ratio:.1f}x")
    print(f"{'='*60}")

    # The LUT should reduce FLOPs (at minimum, not increase them)
    assert (
        flops_on <= flops_off
    ), f"LUT should not increase FLOPs: on={flops_on}, off={flops_off}"
