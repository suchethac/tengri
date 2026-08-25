# SPDX-License-Identifier: BSD-3-Clause
r"""Gray geometric Type-1/Type-2 visibility mask for the composable AGN runner.

A dusty torus hides the accretion disc and broad-line region (the compact
"central engine") along edge-on sightlines while the spatially-extended
narrow-line region stays visible at all inclinations. This module provides the
**gray** (wavelength-independent) visibility mask used by the composable runner
for torus blocks that do not carry their own wavelength-dependent dusty screen
(see :mod:`tengri.components.agn.blocks.torus_screen` for that complementary,
λ-dependent model).

The mask is the same function the monolithic ``unified_nlr_blr`` model uses, so a
composable disc+torus+NLR+BLR config reproduces its Type-1/2 geometry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["sigmoid_visibility_mask", "split_lines_result"]


def split_lines_result(result: Array | tuple[Array, Array]) -> tuple[Array, Array]:
    """Normalize an NLR/BLR block return to ``(anisotropic, isotropic)`` L_lambda.

    An NLR or BLR block may return either a single ``L_lambda`` array (treated as
    fully anisotropic / maskable, the back-compatible default) or a
    ``(L_maskable, L_isotropic)`` tuple (e.g. BLR maskable, NLR isotropic). This
    collapses both forms to a fixed ``(aniso, iso)`` pair so the runner can mask
    only the anisotropic central engine. The branch is on the Python return type,
    which is static per block name; JIT-safe.

    Parameters
    ----------
    result : Array or tuple of (Array, Array)
        A lines block's return value.

    Returns
    -------
    tuple of (Array, Array)
        ``(L_anisotropic, L_isotropic)`` [erg/s/Å].
    """
    if isinstance(result, tuple):
        aniso, iso = result
        return jnp.asarray(aniso), jnp.asarray(iso)
    arr = jnp.asarray(result)
    return arr, jnp.zeros_like(arr)


def sigmoid_visibility_mask(
    cos_inc: Array | float,
    theta_torus: Array | float,
    width: float = 2.0,
) -> Array:
    r"""Smooth disc/BLR visibility as a function of inclination.

    Returns ~1 (visible, Type-1) for face-on orientations and ~0 (obscured,
    Type-2) once the sightline grazes the torus edge. The critical inclination is
    ``inc_crit = 90° − theta_torus``; Synthesizer implements this as a hard binary
    step (``inclination + theta_torus > 90°`` → zeroed), replaced here by a smooth
    sigmoid so the inclination stays a differentiable fit parameter.

    .. math::

        \sigma(i, \theta_t) = \mathrm{sigmoid}\!\left(
            -\frac{\arccos(\cos i) - (90^\circ - \theta_t)}{w}\right)

    with :math:`i` the inclination [deg], :math:`\theta_t` the torus half-opening
    angle [deg], and :math:`w` the transition half-width [deg].

    Parameters
    ----------
    cos_inc : array_like or float
        Cosine of inclination (0 = edge-on, 1 = face-on).
    theta_torus : array_like or float
        Torus half-opening angle [deg].
    width : float, optional
        Sigmoid transition half-width [deg]. Default ``2.0``. As ``width → 0``
        this converges to Synthesizer's hard binary mask.

    Returns
    -------
    Array
        Visibility fraction in [0, 1].

    Notes
    -----
    **JIT/grad/vmap compatible.**

    References
    ----------
    .. [1] Synthesizer ``torus_edgeon_condition`` (Lovell et al. 2025,
           OJA 8, doi:10.33232/001c.145766; Roper et al. 2026, JOSS 11, 9436,
           doi:10.21105/joss.09436: cite both):
           https://github.com/synthesizer-project/synthesizer/blob/main/src/synthesizer/emission_models/agn/unified_agn.py
    """
    inc_deg = jnp.degrees(jnp.arccos(jnp.clip(cos_inc, 0.0, 1.0)))
    inc_crit = 90.0 - jnp.clip(theta_torus, 0.0, 90.0)
    return jax.nn.sigmoid(-(inc_deg - inc_crit) / jnp.maximum(width, 0.1))
