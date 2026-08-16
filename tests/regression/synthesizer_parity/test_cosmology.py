# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for cosmology utilities — synthesizer parity.

Mirrors synthesizer's ``tests/test_cosmology.py`` shape. The relations
checked here are pure geometry (no model-specific assumptions beyond
flat-FLRW), so any backend implementation is required to satisfy them.

Pitfalls guarded:
- P-19: luminosity vs angular-diameter distance confusion (factor (1+z)^2).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_paper
import jax
import jax.numpy as jnp

from tengri.utils.cosmology import (
    angular_diameter_distance_mpc,
    luminosity_distance_mpc,
)


def test_distance_relationship_dl_da_consistent_with_redshift_squared():
    """D_L / D_A = (1 + z)^2 — Etherington reciprocity for flat FLRW.

    Mirrors: synthesizer/tests/test_cosmology.py::test_distance_relationship.
    Pitfall: P-19 — angular sizes that should use D_A often get D_L.
    """
    for z in (0.5, 1.0, 2.0, 5.0, 10.0):
        d_l = float(luminosity_distance_mpc(z))
        d_a = float(angular_diameter_distance_mpc(z))
        ratio = d_l / d_a
        expected = (1.0 + z) ** 2
        rel_err = abs(ratio - expected) / expected
        assert rel_err < 1e-6, (
            f"z={z}: D_L/D_A = {ratio:.6f}, expected (1+z)^2 = {expected:.6f}, "
            f"relative error {rel_err:.2e} exceeds 1e-6 — Etherington violated."
        )


def test_zero_redshift_distances_vanish():
    """At z=0, D_L = D_A = 0 with no NaN.

    Mirrors: synthesizer/tests/test_cosmology.py::test_zero_redshift.
    Pitfall: P-24 — guards against 0/0 NaN at z=0.
    """
    d_l = float(luminosity_distance_mpc(0.0))
    d_a = float(angular_diameter_distance_mpc(0.0))
    assert jnp.isfinite(d_l), "D_L is non-finite at z=0"
    assert jnp.isfinite(d_a), "D_A is non-finite at z=0"
    assert abs(d_l) < 1e-6, f"D_L(z=0) should vanish, got {d_l} Mpc"
    assert abs(d_a) < 1e-6, f"D_A(z=0) should vanish, got {d_a} Mpc"


def test_high_redshift_distance_ordering_and_bounds():
    """At z=10 the universe age implies D_L > D_A and both are sub-Gpc-scale-Mpc.

    Mirrors: synthesizer/tests/test_cosmology.py::test_high_redshift.
    """
    d_l = float(luminosity_distance_mpc(10.0))
    d_a = float(angular_diameter_distance_mpc(10.0))
    assert d_l > d_a, f"D_L ({d_l:.0f}) must exceed D_A ({d_a:.0f}) at z=10"
    # Sanity bound on a flat-Planck18 cosmology:
    # D_L(z=10) ~ 1.06e5 Mpc, D_A(z=10) ~ 880 Mpc.
    assert 5e4 < d_l < 2e5, f"D_L(z=10)={d_l:.0f} Mpc outside expected band"
    assert 1e2 < d_a < 5e3, f"D_A(z=10)={d_a:.0f} Mpc outside expected band"


def test_distance_jit_and_grad_compatible():
    """Distances must be jittable and differentiable (for VI/HMC fits at z).

    No synthesizer parallel — tengri-specific because synthesizer is numpy-eager.
    """

    @jax.jit
    def dl_grad(z):
        return luminosity_distance_mpc(z)

    val = float(dl_grad(1.5))
    assert jnp.isfinite(val) and val > 0
    grad_fn = jax.grad(lambda z: luminosity_distance_mpc(z))
    g = float(grad_fn(1.5))
    assert jnp.isfinite(g) and g > 0, "dD_L/dz should be finite and positive"
