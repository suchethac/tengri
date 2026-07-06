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


def test_exact_for_lnu_inverse_nu():
    """L_nu = A/nu makes nu*L_nu constant in ln(lambda) — trapezoid is exact:
    integral of L_nu d nu = A * ln(nu_max/nu_min)."""
    wave_aa = jnp.geomspace(1e3, 1e7, 401)
    nu = C_AA / wave_aa
    amplitude = 3.7e18
    l_nu = amplitude / nu

    result = integrate_lnu_over_nu(l_nu, wave_aa)
    expected = amplitude * jnp.log(nu[0] / nu[-1])

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
