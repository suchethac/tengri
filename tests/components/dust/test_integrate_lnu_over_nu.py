# SPDX-License-Identifier: BSD-3-Clause
"""Analytic-limit tests for the canonical frequency-integral helper.

One implementation of :math:`\\int L_\\nu\\,d\\nu` lives in
``components/dust/emission/_physics.py``; the former per-module copies in
``draine2021_pah.py`` and ``astrodust_hd23.py`` were bit-identical duplicates
consolidated here (2026-07 audit).
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from tengri.components.dust.emission._physics import integrate_lnu_over_nu
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.limit


def test_exact_for_lnu_linear_in_nu():
    """L_nu = a + b nu is integrated exactly by the trapezoid in nu.

    The helper is the quadrature every template closure normalizes with on
    the evaluation grid, so it must be the plain trapezoid in nu: that is
    what makes a normalized ``sed_dust_ir`` integrate back to ``L_ir`` to
    round-off (the energy-balance contract). The former nu*L_nu-in-ln(lambda)
    form was exact for L_nu = A/nu instead and differed from this one by its
    discretization error (1.4e-5 on the Draine et al. 2021 PAH grid).
    """
    wave_aa = jnp.geomspace(1e3, 1e7, 401)
    nu = C_AA / wave_aa
    a, b = 3.7e18, 2.5e4
    l_nu = a + b * nu

    result = integrate_lnu_over_nu(l_nu, wave_aa)
    nu_max, nu_min = nu[0], nu[-1]
    expected = a * (nu_max - nu_min) + 0.5 * b * (nu_max**2 - nu_min**2)

    chex.assert_trees_all_close(result, expected, rtol=1e-12)


def test_batched_leading_axes():
    """Broadcasts over leading axes; integrates the last axis only."""
    wave_aa = jnp.geomspace(1e3, 1e6, 201)
    nu = C_AA / wave_aa
    l_nu = jnp.stack([1.0 / nu, 2.0 / nu])

    result = integrate_lnu_over_nu(l_nu, wave_aa)

    chex.assert_shape(result, (2,))
    chex.assert_trees_all_close(result[1], 2.0 * result[0], rtol=1e-12)


def test_consumers_share_the_one_implementation():
    """Both former call sites import THE canonical helper (no local copies)."""
    from tengri.components.dust import draine2021_pah_ir
    from tengri.components.dust.emission.templates import astrodust

    assert draine2021_pah_ir.integrate_lnu_over_nu is integrate_lnu_over_nu
    assert astrodust.integrate_lnu_over_nu is integrate_lnu_over_nu
