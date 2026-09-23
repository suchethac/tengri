# SPDX-License-Identifier: BSD-3-Clause
"""The single-screen energy-balance LUT must actually remove work.

Its accuracy is pinned in ``test_2485_single_screen_energy_balance.py``; this
file pins the other half, because the two fail independently. A LUT that is
built, threaded and numerically perfect but never read produces identical
FLOPs and identical photometry -- which is exactly the state this change was
in before the consumer was wired, and no accuracy test could see it.

Gradient FLOPs come from the compiled HLO, so they are deterministic and
machine-independent: unlike wall clock, they can carry a real threshold in CI
rather than a "not worse than" comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.sed_model import WavePrecomp

pytestmark = pytest.mark.regression_bug

#: Deliberately far below the measured saving, so ordinary variation in
#: the fixture cannot redden CI while a silently-unread LUT still fails.
MIN_SPEEDUP = 1.5

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
    cost = (
        jax.jit(jax.grad(lambda q: jnp.sum(model.predict_photometry(q))))
        .lower(params)
        .compile()
        .cost_analysis()
    )
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
    print(f"\n{'=' * 60}")
    print("Single-screen LUT consumer performance:")
    print(f"  With LUT:    {flops_on:,} FLOPs")
    print(f"  Without LUT: {flops_off:,} FLOPs")
    print(f"  Speedup:     {ratio:.1f}x")
    print(f"{'=' * 60}")

    # `<=` would pass against a LUT that is built and never read, which is the
    # defect this file exists for. Measured on the real fsps_prsc_c3k_a_chabrier
    # grid the saving is ~65x with an active nebular backend and ~86x without;
    # on the small synthetic fixture used here it is smaller, so the threshold
    # is set well below either and still far above 1.
    assert flops_off > MIN_SPEEDUP * flops_on, (
        f"the LUT removed little or no work: {flops_off:,} -> {flops_on:,} "
        f"({flops_off / max(flops_on, 1):.2f}x, needs > {MIN_SPEEDUP}x). "
        "Identical or near-identical FLOPs mean it is built but never read."
    )
