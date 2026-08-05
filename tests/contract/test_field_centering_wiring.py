# SPDX-License-Identifier: BSD-3-Clause
"""Partial centering of the DRW field (#1355), wired through compute_field_gp.

The funnel this addresses: ``s = L(sigma, tau) xi`` is BILINEAR in the
hyperparameters and the latents. Whitening is optimal over linear maps, so no
fixed metric reaches a multiplicative coupling — which is why preconditioning
did not fix it and why the measured R-hat splits the way it does (latents 0.994,
shared sigma 1.09, shared tau 1.64).
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.gp_sfh import drw_latent_log_prior
from tengri.components.stellar.sfh.registry import compute_field_gp

pytestmark = pytest.mark.contract


def _grid(n=16):
    return jnp.asarray(np.linspace(6.0, 10.14, n))


def test_default_centering_is_bit_identical_to_the_production_path():
    """a = 1 must not perturb a single bit of the shipped behavior."""
    n = 16
    g = _grid(n)
    xi = jnp.asarray(np.random.default_rng(0).normal(size=n))
    d = float(g[1] - g[0])
    base = compute_field_gp(xi, 0.8, 1.5e8, n, d, log_age_grid=g)
    same = compute_field_gp(xi, 0.8, 1.5e8, n, d, log_age_grid=g, centering=1.0)
    chex.assert_trees_all_close(base[0], same[0], rtol=0.0, atol=0.0)
    chex.assert_trees_all_close(base[1], same[1], rtol=0.0, atol=0.0)


def test_centering_changes_the_map():
    """a = 0 must produce a DIFFERENT field, or the knob is a no-op.

    A reparameterization that does not change the map cannot change the
    geometry — the exact error that made #1301's OU-innovations "fix" a no-op.
    """
    n = 16
    g = _grid(n)
    xi = jnp.asarray(np.random.default_rng(1).normal(size=n))
    d = float(g[1] - g[0])
    a1 = compute_field_gp(xi, 0.8, 1.5e8, n, d, log_age_grid=g, centering=1.0)[0]
    a0 = compute_field_gp(xi, 0.8, 1.5e8, n, d, log_age_grid=g, centering=0.0)[0]
    assert not np.allclose(np.asarray(a1), np.asarray(a0)), (
        "centering=0 produced the same field as centering=1; the knob is a no-op"
    )


def test_marginal_variance_is_invariant_to_centering():
    """Both ends realize the SAME prior; only the coordinates differ.

    Drawing zeta from its own a-dependent prior and mapping through must give
    the same marginal field variance at every a. If it does not, the knob is
    changing the model rather than the parameterization.
    """
    n, sigma, tau = 12, 0.8, 1.5e8
    g = _grid(n)
    d = float(g[1] - g[0])
    sigma_s = sigma * np.log(10.0)
    rng = np.random.default_rng(2)
    for a in (1.0, 0.5, 0.0):
        scale = sigma_s ** (1.0 - a)  # zeta ~ N(0, sigma_s^(2-2a) I)
        draws = []
        for _ in range(400):
            zeta = jnp.asarray(rng.normal(size=n) * scale)
            draws.append(
                np.asarray(
                    compute_field_gp(zeta, sigma, tau, n, d, log_age_grid=g, centering=a)[0]
                )
            )
        var = float(np.var(np.stack(draws)))
        assert 0.5 * sigma_s**2 < var < 2.0 * sigma_s**2, (
            f"a={a}: marginal field variance {var:.3f} is far from sigma_s^2="
            f"{sigma_s**2:.3f}; centering changed the model, not the coordinates"
        )


def test_latent_log_prior_carries_the_normalizer():
    """The -n(1-a) log sigma_s term is not optional.

    Without it a sampler runs cleanly, reports nothing, and targets a different
    posterior at every a — visible only as a recovered sigma that drifts with a
    knob meant to be a pure change of coordinates.
    """
    n = 8
    zeta = jnp.zeros(n)
    lp1 = float(drw_latent_log_prior(zeta, 0.8, centering=1.0))
    lp0 = float(drw_latent_log_prior(zeta, 0.8, centering=0.0))
    sigma_s = 0.8 * np.log(10.0)
    # At zeta = 0 the quadratic form vanishes and only the normalizer remains,
    # so the difference is exactly the -n(1-a) log sigma_s term.
    chex.assert_trees_all_close(lp1 - lp0, n * np.log(sigma_s), rtol=1e-10)
